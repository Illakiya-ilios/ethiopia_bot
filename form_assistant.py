"""Registration Form Assistant — bot #3 of the Ethiopia Tourism suite.

Helps a signed-in traveller register for a package by driving a website's
registration form from chat. This is "World A": the bot and the website share
one UI. The bot can NAVIGATE the site and FILL form fields, under three hard,
code-enforced boundaries:

    1. NO database writes.  The register path opens no DB session; it only
       emits a FormFillCommand that the frontend applies.
    2. NO auto-submit.      FormFillCommand has no submit capability. The user
       reviews every field and submits manually.
    3. CONSENT-GATED fill.  Passport OCR and chat-history reuse only run AFTER
       the bot asks and the user explicitly agrees.

Auto-fill data sources (both consent-gated except live conversation):
    - conversation : values the user dictates in the current message
    - passport_ocr : GCP Cloud Vision OCR over an uploaded passport image
    - chat_history : reuse of details the user already provided

Auth is keyless via Application Default Credentials (ADC), consistent with the
rest of the app. Passport images are transient and never persisted.

Run standalone (text-only flows; OCR needs an image + GCP creds):

    python form_assistant.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI


# ============================================================
# 0. EXCEPTIONS
# ============================================================


class FormAssistantError(Exception):
    """Base error for the form assistant."""


class ConfigurationError(FormAssistantError):
    """Raised when configuration is missing or invalid."""


class OcrError(FormAssistantError):
    """Raised when the OCR service call fails (transport / ADC)."""


class OcrQualityError(OcrError):
    """Raised when an image is unreadable / too sparse to be a passport."""


class ExtractionError(FormAssistantError):
    """Raised when the LLM value extraction fails."""


# ============================================================
# 1. LOGGING
# ============================================================


logger = logging.getLogger("form_assistant")


def configure_logging(level: str = "WARNING") -> None:
    """Configure root logging (quiet by default; LOG_LEVEL=INFO for detail)."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("urllib3", "google.auth", "grpc", "sqlalchemy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ============================================================
# 2. CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings for the form assistant."""

    gcp_project_id: str
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-2.0-flash-lite"
    llm_temperature: float = 0.0

    # OCR
    ocr_backend: str = "vision"                 # "vision" | "document_ai"
    ocr_location: str = "us"                     # Document AI processor location
    doc_ai_processor_id: str = ""                # required only for document_ai
    max_image_bytes: int = 8 * 1024 * 1024       # reject oversized uploads

    # Demo site — placeholders, swapped for the real form later.
    known_pages: tuple[str, ...] = ("registration", "home", "packages", "visa")
    registration_path: str = "/register"

    log_level: str = "WARNING"

    def validate(self) -> None:
        if not self.gcp_project_id:
            raise ConfigurationError("GCP_PROJECT_ID is required.")
        if self.ocr_backend not in {"vision", "document_ai"}:
            raise ConfigurationError(
                "OCR_BACKEND must be 'vision' or 'document_ai'."
            )
        if self.ocr_backend == "document_ai" and not self.doc_ai_processor_id:
            raise ConfigurationError(
                "DOC_AI_PROCESSOR_ID is required when OCR_BACKEND=document_ai."
            )
        if self.max_image_bytes <= 0:
            raise ConfigurationError("MAX_IMAGE_BYTES must be positive.")


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        gcp_project_id=os.getenv("GCP_PROJECT_ID", ""),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        ocr_backend=os.getenv("OCR_BACKEND", "vision"),
        ocr_location=os.getenv("OCR_LOCATION", "us"),
        doc_ai_processor_id=os.getenv("DOC_AI_PROCESSOR_ID", ""),
        max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024))),
        log_level=os.getenv("LOG_LEVEL", "WARNING"),
    )

    settings.validate()
    return settings


# ============================================================
# 3. DATA STRUCTURES
# ============================================================


class DataSource(str, Enum):
    CONVERSATION = "conversation"
    PASSPORT_OCR = "passport_ocr"
    CHAT_HISTORY = "chat_history"
    NONE = "none"


class ConsentStatus(str, Enum):
    NOT_ASKED = "not_asked"
    AWAITING = "awaiting"
    GRANTED = "granted"
    DENIED = "denied"


@dataclass
class ConsentState:
    """Per-source consent, carried in supervisor state across turns."""

    passport_ocr: ConsentStatus = ConsentStatus.NOT_ASKED
    chat_history: ConsentStatus = ConsentStatus.NOT_ASKED

    def awaiting_source(self) -> Optional[DataSource]:
        if self.passport_ocr == ConsentStatus.AWAITING:
            return DataSource.PASSPORT_OCR
        if self.chat_history == ConsentStatus.AWAITING:
            return DataSource.CHAT_HISTORY
        return None


@dataclass(frozen=True)
class NavigationTarget:
    page_id: str
    url_path: str
    confident: bool


@dataclass
class FormFillCommand:
    """Instruction the frontend applies to the website.

    BOUNDARY: there is intentionally NO submit field and NO DB handle. The
    frontend opens ``navigate_to`` and writes ``field_values`` into inputs; the
    user submits manually.
    """

    navigate_to: Optional[NavigationTarget] = None
    field_values: dict[str, str] = field(default_factory=dict)
    source: Optional[DataSource] = None

    def is_empty(self) -> bool:
        return self.navigate_to is None and not self.field_values


@dataclass
class FormAssistantResult:
    """What the register node returns to the supervisor."""

    answer: str
    command: Optional[FormFillCommand] = None
    consent: ConsentState = field(default_factory=ConsentState)


# ============================================================
# 4. FIELD MAPPING (canonical keys -> frontend form field ids)
# ============================================================


CANONICAL_KEYS = {
    # identity / passport
    "surname", "given_name", "full_name", "date_of_birth", "gender",
    "nationality", "citizenship", "place_of_birth", "country_of_birth",
    "passport_no", "passport_issue_date", "passport_expiry_date",
    "passport_issuing_country", "passport_issuing_authority", "passport_type",
    # contact
    "email", "phone", "whatsapp",
    # trip
    "arrival_date", "departure_date", "airline", "flight_number",
    "destinations", "purpose_of_visit", "port_of_entry",
}


# canonical_key -> frontend form field id (placeholders; swap for real ids).
FIELD_MAP: dict[str, str] = {
    "surname": "reg_surname",
    "given_name": "reg_given_name",
    "full_name": "reg_full_name",
    "date_of_birth": "reg_dob",
    "gender": "reg_gender",
    "nationality": "reg_nationality",
    "citizenship": "reg_citizenship",
    "place_of_birth": "reg_place_of_birth",
    "country_of_birth": "reg_country_of_birth",
    "passport_no": "reg_passport_no",
    "passport_issue_date": "reg_passport_issue_date",
    "passport_expiry_date": "reg_passport_expiry_date",
    "passport_issuing_country": "reg_passport_issuing_country",
    "passport_issuing_authority": "reg_passport_issuing_authority",
    "passport_type": "reg_passport_type",
    "email": "reg_email",
    "phone": "reg_phone",
    "whatsapp": "reg_whatsapp",
    "arrival_date": "reg_arrival_date",
    "departure_date": "reg_departure_date",
    "airline": "reg_airline",
    "flight_number": "reg_flight_no",
    "destinations": "reg_destinations",
    "purpose_of_visit": "reg_purpose_of_visit",
    "port_of_entry": "reg_port_of_entry",
}


def map_to_form_fields(canonical: dict[str, str]) -> dict[str, str]:
    """Deterministically map canonical keys to frontend form field ids.

    Pure function: no LLM, no DB, no I/O. Unknown keys are skipped.
    """

    return {
        FIELD_MAP[key]: value
        for key, value in canonical.items()
        if key in FIELD_MAP and value
    }


# ============================================================
# 5. NAVIGATION INTENT
# ============================================================


NAV_PROMPT = """You detect website navigation requests for a tourism site.

Known pages: {pages}

If the user's message asks to open / go to / navigate to a page, respond with
ONLY that page id (one of the known pages). If the request is about
registering or filling the registration form, respond with: registration
If no navigation is requested, respond with exactly: none

USER MESSAGE:
{message}

PAGE:"""


def detect_navigation_intent(
    llm: ChatVertexAI,
    message: str,
    settings: Settings,
) -> Optional[NavigationTarget]:
    """Return a NavigationTarget if the user asked to open a page, else None."""

    prompt = NAV_PROMPT.format(
        pages=", ".join(settings.known_pages),
        message=message,
    )

    try:
        response = llm.invoke(prompt)
        label = (response.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.error("Navigation detection failed: %s", exc)
        return None

    label = re.sub(r"[^a-z_]", "", label)

    if label in ("", "none"):
        return None

    if label in settings.known_pages:
        path = (
            settings.registration_path
            if label == "registration"
            else f"/{label}"
        )
        return NavigationTarget(page_id=label, url_path=path, confident=True)

    # Model returned something unexpected -> ask the user to confirm.
    return NavigationTarget(page_id="registration",
                            url_path=settings.registration_path,
                            confident=False)


# ============================================================
# 6. FIELD EXTRACTION (conversation + history)
# ============================================================


_EXTRACT_KEYS_LIST = ", ".join(sorted(CANONICAL_KEYS))

EXTRACT_PROMPT = """You extract registration form values from text.

Return ONLY a JSON object mapping canonical keys to string values. Use ONLY
these keys (omit any you cannot find): {keys}

Rules:
- Do NOT invent values. Only include what is clearly stated.
- Dates must be ISO format YYYY-MM-DD.
- gender: "Male" or "Female".
- Return {{}} if nothing is found. No prose, no markdown.

TEXT:
{text}

JSON:"""


def _parse_json_object(raw: str) -> dict[str, str]:
    """Extract a JSON object from an LLM response, tolerant of code fences."""

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Grab the first {...} block if there's surrounding text.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    # Keep only known canonical keys with non-empty string values.
    result: dict[str, str] = {}
    for key, value in data.items():
        if key in CANONICAL_KEYS and isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


def extract_fields_from_message(llm: ChatVertexAI, message: str) -> dict[str, str]:
    """LLM extracts canonical field values from the current message."""

    prompt = EXTRACT_PROMPT.format(keys=_EXTRACT_KEYS_LIST, text=message)

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Field extraction failed: {exc}") from exc

    return _parse_json_object(response.content or "")


def extract_fields_from_history(
    llm: ChatVertexAI,
    history: list[dict],
) -> dict[str, str]:
    """Consent-gated: mine prior chat turns for reusable field values."""

    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in history
    )

    prompt = EXTRACT_PROMPT.format(keys=_EXTRACT_KEYS_LIST, text=transcript)

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"History extraction failed: {exc}") from exc

    return _parse_json_object(response.content or "")


# ============================================================
# 7. OCR PIPELINE (GCP Cloud Vision + passport parsing)
# ============================================================


def build_ocr_client(settings: Settings):
    """Create the GCP OCR client using ADC (keyless)."""

    if settings.ocr_backend != "vision":
        raise ConfigurationError(
            "Only the 'vision' OCR backend is implemented in this POC."
        )

    from google.cloud import vision  # imported lazily to keep startup light

    logger.info("Initializing Cloud Vision OCR client")
    return vision.ImageAnnotatorClient()


def run_passport_ocr(client, image_bytes: bytes, settings: Settings) -> str:
    """Invoke Cloud Vision document_text_detection and return the full text.

    Does NOT persist the image. Raises OcrQualityError if the text is too
    sparse to be a passport, OcrError on a service failure.
    """

    if not image_bytes:
        raise OcrQualityError("No image provided.")

    if len(image_bytes) > settings.max_image_bytes:
        raise OcrError("Image exceeds the maximum allowed size.")

    from google.cloud import vision

    try:
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
    except Exception as exc:  # noqa: BLE001
        raise OcrError(f"OCR service call failed: {exc}") from exc

    if response.error.message:
        raise OcrError(response.error.message)

    text = response.full_text_annotation.text or ""

    if len(text.strip()) < 20:
        raise OcrQualityError("OCR text too sparse to be a passport.")

    return text


# ---- passport / MRZ parsing --------------------------------------------


_MRZ_LINE = re.compile(r"^[A-Z0-9<]{30,44}$")


def _mrz_date_to_iso(yymmdd: str, *, expiry: bool = False) -> Optional[str]:
    """Convert an MRZ YYMMDD date to ISO YYYY-MM-DD (best-effort century)."""

    if not re.fullmatch(r"\d{6}", yymmdd):
        return None

    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    # Expiry dates are future-leaning; birth dates are past-leaning.
    century = 2000 if (expiry or yy <= (datetime.now().year % 100)) else 1900
    if not expiry and 2000 + yy > datetime.now().year:
        century = 1900
    return f"{century + yy:04d}-{mm:02d}-{dd:02d}"


def _parse_mrz(lines: list[str]) -> dict[str, str]:
    """Parse a TD3 (passport) MRZ: two 44-char lines."""

    mrz = [ln for ln in lines if _MRZ_LINE.match(ln)]
    if len(mrz) < 2:
        return {}

    line1, line2 = mrz[-2], mrz[-1]
    result: dict[str, str] = {}

    # Line 1: P<ISSUER SURNAME<<GIVEN<NAMES
    if line1.startswith("P"):
        issuer = line1[2:5].replace("<", "")
        if issuer:
            result["passport_issuing_country"] = issuer
        name_field = line1[5:]
        if "<<" in name_field:  # surname<<given separator
            surname_raw, given_raw = name_field.split("<<", 1)
            surname = surname_raw.replace("<", " ").strip()
            given = given_raw.replace("<", " ").strip()
            if surname:
                result["surname"] = surname.title()
            if given:
                result["given_name"] = given.title()
        else:
            names = name_field.replace("<", " ").strip()
            if names:
                result["full_name"] = names.title()

    # Line 2: passport_no(9) chk nat(3) dob(6) chk sex expiry(6) ...
    if len(line2) >= 28:
        passport_no = line2[0:9].replace("<", "")
        nationality = line2[10:13].replace("<", "")
        dob = line2[13:19]
        sex = line2[20:21]
        expiry = line2[21:27]

        if passport_no:
            result["passport_no"] = passport_no
        if nationality:
            result["nationality"] = nationality
        dob_iso = _mrz_date_to_iso(dob)
        if dob_iso:
            result["date_of_birth"] = dob_iso
        if sex in ("M", "F"):
            result["gender"] = "Male" if sex == "M" else "Female"
        exp_iso = _mrz_date_to_iso(expiry, expiry=True)
        if exp_iso:
            result["passport_expiry_date"] = exp_iso

    return result


def parse_passport_fields(ocr_text: str) -> dict[str, str]:
    """Parse OCR text into canonical passport fields (MRZ preferred)."""

    if not ocr_text or not ocr_text.strip():
        return {}

    lines = [ln.strip().upper() for ln in ocr_text.splitlines() if ln.strip()]

    result = _parse_mrz(lines)

    # Fallback: labeled-line heuristics for anything MRZ didn't yield.
    if "passport_no" not in result:
        for ln in lines:
            m = re.search(r"(?:PASSPORT\s*(?:NO|NUMBER)|DOCUMENT\s*NO)\D*([A-Z0-9]{6,})", ln)
            if m:
                result["passport_no"] = m.group(1)
                break

    return {k: v for k, v in result.items() if k in CANONICAL_KEYS}


# ============================================================
# 8. CONSENT HANDLING
# ============================================================


_CONSENT_PROMPTS = {
    DataSource.PASSPORT_OCR: (
        "I can read your passport photo to fill in the form. Your image is "
        "used only to extract the details and is not saved. Shall I go ahead? "
        "(yes/no)"
    ),
    DataSource.CHAT_HISTORY: (
        "I can reuse the details you already shared in our chat (name, dates, "
        "etc.) to fill the form. Would you like me to do that? (yes/no)"
    ),
}


def request_consent(source: DataSource) -> str:
    return _CONSENT_PROMPTS.get(source, "Do you consent? (yes/no)")


def interpret_consent_reply(message: str) -> Optional[bool]:
    """Return True (yes), False (no), or None (not a clear yes/no)."""

    text = message.strip().lower()
    if re.search(r"\b(yes|yeah|yep|sure|ok|okay|go ahead|please do|do it)\b", text):
        return True
    if re.search(r"\b(no|nope|don't|do not|stop|cancel|nah)\b", text):
        return False
    return None


def update_consent(
    state: ConsentState,
    source: DataSource,
    granted: bool,
) -> ConsentState:
    status = ConsentStatus.GRANTED if granted else ConsentStatus.DENIED
    if source == DataSource.PASSPORT_OCR:
        state.passport_ocr = status
    elif source == DataSource.CHAT_HISTORY:
        state.chat_history = status
    return state


# ============================================================
# 9. FILL-SOURCE INTENT
# ============================================================


FILL_SOURCE_PROMPT = """Classify what the user wants for filling a form.

Respond with ONE word:
- passport_ocr : they want to use / read their passport image
- chat_history : they want to reuse what they already told you in the chat
- conversation : they are stating field values right now in this message
- none         : none of the above

USER MESSAGE:
{message}

ANSWER:"""


def classify_fill_source(llm: ChatVertexAI, message: str) -> DataSource:
    try:
        response = llm.invoke(FILL_SOURCE_PROMPT.format(message=message))
        label = (response.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fill-source classification failed: %s", exc)
        return DataSource.NONE

    for src in (DataSource.PASSPORT_OCR, DataSource.CHAT_HISTORY,
                DataSource.CONVERSATION):
        if src.value in label:
            return src
    return DataSource.NONE


# ============================================================
# 10. FORM ASSISTANT SERVICE
# ============================================================


def summarize_command(command: FormFillCommand) -> str:
    """Human-friendly summary of what the bot filled (never mentions submit)."""

    parts = []
    if command.navigate_to is not None:
        parts.append(f"opened the {command.navigate_to.page_id} page")
    if command.field_values:
        n = len(command.field_values)
        parts.append(f"filled in {n} field{'s' if n != 1 else ''}")

    if not parts:
        return "I didn't catch any details to fill. Could you restate them?"

    action = " and ".join(parts)
    return (
        f"I've {action}. Please review everything and submit the form yourself "
        "when it looks right."
    )


@dataclass
class FormAssistant:
    """Drives the registration form from chat. Never writes DB, never submits."""

    settings: Settings
    llm: ChatVertexAI
    ocr_client: object = None

    @classmethod
    def bootstrap(cls, settings: Optional[Settings] = None) -> "FormAssistant":
        settings = settings or load_settings()

        logger.info("Initializing Gemini model '%s'", settings.llm_model)
        llm = ChatVertexAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

        # OCR client is built lazily on first passport use to keep startup fast
        # and avoid requiring the Vision API unless it's actually needed.
        return cls(settings=settings, llm=llm, ocr_client=None)

    def _ensure_ocr_client(self):
        if self.ocr_client is None:
            self.ocr_client = build_ocr_client(self.settings)
        return self.ocr_client

    def handle(
        self,
        message: str,
        *,
        user_id: Optional[int] = None,
        history: Optional[list[dict]] = None,
        image_bytes: Optional[bytes] = None,
        consent: Optional[ConsentState] = None,
    ) -> FormAssistantResult:
        """Route a form-fill turn. Returns text + FormFillCommand.

        Never writes to the DB and never submits the form.
        """

        history = history or []
        consent = consent or ConsentState()
        command = FormFillCommand()

        # 1) Navigation (independent of consent).
        nav = detect_navigation_intent(self.llm, message, self.settings)
        if nav is not None:
            if not nav.confident:
                return FormAssistantResult(
                    "Which page did you mean? For example: "
                    f"{', '.join(self.settings.known_pages)}.",
                    command,
                    consent,
                )
            command.navigate_to = nav

        # 2) If awaiting a consent answer, interpret yes/no first.
        awaiting = consent.awaiting_source()
        if awaiting is not None:
            decision = interpret_consent_reply(message)
            if decision is None:
                return FormAssistantResult(
                    "Please answer yes or no.", command, consent
                )
            consent = update_consent(consent, awaiting, decision)
            if not decision:
                return FormAssistantResult(
                    "Okay, I won't use that. You can fill it in yourself, or "
                    "tell me the details directly.",
                    command,
                    consent,
                )
            # Granted -> perform that source's fill now.
            return self._perform_fill(awaiting, message, history,
                                      image_bytes, command, consent)

        # 3) Determine the requested fill source from the message.
        intent = classify_fill_source(self.llm, message)

        if intent == DataSource.PASSPORT_OCR:
            if consent.passport_ocr != ConsentStatus.GRANTED:
                consent.passport_ocr = ConsentStatus.AWAITING
                return FormAssistantResult(
                    request_consent(DataSource.PASSPORT_OCR), command, consent
                )
            return self._perform_fill(DataSource.PASSPORT_OCR, message,
                                      history, image_bytes, command, consent)

        if intent == DataSource.CHAT_HISTORY:
            if consent.chat_history != ConsentStatus.GRANTED:
                consent.chat_history = ConsentStatus.AWAITING
                return FormAssistantResult(
                    request_consent(DataSource.CHAT_HISTORY), command, consent
                )
            return self._perform_fill(DataSource.CHAT_HISTORY, message,
                                      history, image_bytes, command, consent)

        if intent == DataSource.CONVERSATION:
            return self._perform_fill(DataSource.CONVERSATION, message,
                                      history, image_bytes, command, consent)

        # No fill intent — just navigation, or nothing actionable.
        if command.navigate_to is not None:
            return FormAssistantResult(
                summarize_command(command), command, consent
            )
        return FormAssistantResult(
            "I can open the registration page and help fill the form. You can "
            "dictate details, upload your passport, or reuse our chat history.",
            command,
            consent,
        )

    def _perform_fill(
        self,
        source: DataSource,
        message: str,
        history: list[dict],
        image_bytes: Optional[bytes],
        command: FormFillCommand,
        consent: ConsentState,
    ) -> FormAssistantResult:
        """Execute the fill for a source. Consent for gated sources is GRANTED."""

        if source == DataSource.PASSPORT_OCR:
            assert consent.passport_ocr == ConsentStatus.GRANTED
            if not image_bytes:
                return FormAssistantResult(
                    "Please upload your passport image first, then ask again.",
                    command,
                    consent,
                )
            try:
                text = run_passport_ocr(
                    self._ensure_ocr_client(), image_bytes, self.settings
                )
                canonical = parse_passport_fields(text)
            except OcrQualityError:
                return FormAssistantResult(
                    "I couldn't read the passport clearly. Please try a "
                    "sharper, well-lit photo.",
                    command,
                    consent,
                )
            except OcrError as exc:
                logger.error("OCR error: %s", exc)
                return FormAssistantResult(
                    "Passport reading is unavailable right now. You can type "
                    "the details instead.",
                    command,
                    consent,
                )
            command.field_values = map_to_form_fields(canonical)
            command.source = DataSource.PASSPORT_OCR

        elif source == DataSource.CHAT_HISTORY:
            assert consent.chat_history == ConsentStatus.GRANTED
            try:
                canonical = extract_fields_from_history(self.llm, history)
            except ExtractionError as exc:
                logger.error("History extraction error: %s", exc)
                return FormAssistantResult(
                    "I couldn't pull details from our chat. Please restate "
                    "them and I'll fill the form.",
                    command,
                    consent,
                )
            command.field_values = map_to_form_fields(canonical)
            command.source = DataSource.CHAT_HISTORY

        elif source == DataSource.CONVERSATION:
            try:
                canonical = extract_fields_from_message(self.llm, message)
            except ExtractionError as exc:
                logger.error("Message extraction error: %s", exc)
                return FormAssistantResult(
                    "I couldn't process those details. Could you restate them?",
                    command,
                    consent,
                )
            command.field_values = map_to_form_fields(canonical)
            command.source = DataSource.CONVERSATION

        return FormAssistantResult(summarize_command(command), command, consent)


# ============================================================
# 11. CLI
# ============================================================


def main() -> int:
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings.log_level)

    try:
        assistant = FormAssistant.bootstrap(settings)
    except FormAssistantError as exc:
        logger.error("Failed to start: %s", exc)
        return 1

    print("\n===================================")
    print(" Registration Form Assistant")
    print("===================================")
    print("Text-only demo (passport OCR needs the Streamlit UI for uploads).")

    consent = ConsentState()
    history: list[dict] = []

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if message.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            return 0

        if not message:
            continue

        history.append({"role": "user", "content": message})

        try:
            result = assistant.handle(
                message, history=history, consent=consent
            )
            consent = result.consent
            print(f"\nAssistant: {result.answer}")
            if result.command and result.command.field_values:
                print(f"  [form-fill] {result.command.field_values}")
            if result.command and result.command.navigate_to:
                print(f"  [navigate]  {result.command.navigate_to.url_path}")
            history.append({"role": "assistant", "content": result.answer})
        except FormAssistantError as exc:
            logger.error("Request failed: %s", exc)
            print("\nAssistant: Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    raise SystemExit(main())
