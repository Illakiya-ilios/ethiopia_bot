"""Registration Form Assistant — bot #3 of the Ethiopia Tourism suite.

Helps a signed-in traveller register for a package by driving the website's
registration form from chat. Operates under three HARD constraints:

    1. NO database writes. It only emits FormFillCommand values the frontend
       applies to form fields.
    2. NO auto-submit. The user reviews every field and submits manually; the
       command schema has no submit capability.
    3. CONSENT-GATED auto-fill. Passport OCR and chat-history reuse only run
       AFTER the bot asks and the user explicitly agrees.

Auto-fill data sources:
    - conversation : values the user dictates in the current message
    - passport_ocr : GCP Cloud Vision OCR over an uploaded passport image
                     (keyless ADC), with MRZ + labeled-text parsing
    - chat_history : reuse details already provided earlier in the chat

Fits the existing enterprise single-file style: frozen Settings, typed
exception hierarchy, structured logging, lazy bootstrap(), a service class.

Auth uses Application Default Credentials (ADC); no API keys.
"""

from __future__ import annotations

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

from endpoints import SITE_PAGES, festival_register_url, match_package_slug


# ============================================================
# 0. EXCEPTIONS
# ============================================================


class FormAssistantError(Exception):
    """Base error for the form assistant."""


class ConfigurationError(FormAssistantError):
    """Raised when configuration is missing or invalid."""


class OcrError(FormAssistantError):
    """Raised when the OCR service call fails (transport/ADC)."""


class OcrQualityError(OcrError):
    """Raised when the image is unreadable / too sparse to parse."""


class ExtractionError(FormAssistantError):
    """Raised when LLM value extraction fails."""


# ============================================================
# 1. LOGGING
# ============================================================


logger = logging.getLogger("form_assistant")


def configure_logging(level: str = "WARNING") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("urllib3", "google.auth", "grpc", "sqlalchemy", "httpx"):
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
    ocr_backend: str = "vision"          # "vision" (only backend for POC)
    max_image_bytes: int = 8 * 1024 * 1024

    log_level: str = "WARNING"

    def validate(self) -> None:
        if not self.gcp_project_id:
            raise ConfigurationError("GCP_PROJECT_ID is required.")
        if self.ocr_backend not in {"vision"}:
            raise ConfigurationError(
                f"Unsupported OCR_BACKEND: {self.ocr_backend}"
            )


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        gcp_project_id=os.getenv("GCP_PROJECT_ID", ""),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        ocr_backend=os.getenv("OCR_BACKEND", "vision"),
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
    """Per-source consent, carried across turns in supervisor state."""

    passport_ocr: ConsentStatus = ConsentStatus.NOT_ASKED
    chat_history: ConsentStatus = ConsentStatus.NOT_ASKED

    def status_for(self, source: DataSource) -> ConsentStatus:
        if source == DataSource.PASSPORT_OCR:
            return self.passport_ocr
        if source == DataSource.CHAT_HISTORY:
            return self.chat_history
        return ConsentStatus.GRANTED  # conversation needs no consent

    def awaiting_source(self) -> Optional[DataSource]:
        if self.passport_ocr == ConsentStatus.AWAITING:
            return DataSource.PASSPORT_OCR
        if self.chat_history == ConsentStatus.AWAITING:
            return DataSource.CHAT_HISTORY
        return None


@dataclass
class FormFillCommand:
    """Instruction the frontend applies to the website form.

    BOUNDARIES (structural): no submit action, no DB handles. The frontend
    only opens a page and/or fills fields; the USER submits.
    """

    navigate_to: Optional[dict] = None            # {"page": str, "url": str}
    field_values: dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None                  # provenance for the UI note
    # deliberately: NO `submit` field.

    def to_dict(self) -> dict:
        return {
            "navigate_to": self.navigate_to,
            "field_values": self.field_values,
            "source": self.source,
        }


@dataclass
class FormAssistantResult:
    """What the register node returns to the supervisor / API."""

    answer: str
    command: Optional[FormFillCommand] = None
    consent: ConsentState = field(default_factory=ConsentState)


# ============================================================
# 4. FIELD MAPPING
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


# canonical_key -> frontend form field id (PLACEHOLDER ids; swap for real ones)
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
# 5. NAVIGATION
# ============================================================


_PAGE_KEYWORDS = {
    "packages": ["packages", "package", "itinerary", "itineraries", "trip",
                 "tour", "browse packages"],
    "calendar": ["calendar", "festival calendar", "dates", "schedule"],
    "flagship": ["flagship", "flagship festival", "main festival"],
    "heritage": ["heritage", "history", "historic", "culture"],
    "stopover": ["stopover", "stop over", "layover"],
    "closing": ["closing", "strategy", "plan"],
    "home": ["home", "homepage", "main page", "top", "landing"],
}


def detect_navigation_intent(message: str) -> Optional[dict]:
    """Return {"page", "url"} if the message names a known page, else None.

    First tries to deep-link to a specific festival registration page
    (/packages/<slug>); otherwise resolves a landing-page section.
    """

    lowered = message.lower()

    # 1) Specific package/festival deep-link (path-based /packages/<slug>).
    slug = match_package_slug(lowered)
    if slug:
        return {"page": f"package_{slug}", "url": festival_register_url(slug)}

    # 2) Landing-page section.
    for page_id, keywords in _PAGE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords) and page_id in SITE_PAGES:
            return {"page": page_id, "url": SITE_PAGES[page_id]}
    return None


# ============================================================
# 6. FILL-SOURCE + FIELD EXTRACTION (LLM)
# ============================================================


_FILL_SOURCE_PROMPT = """Classify how the user wants to fill the registration form.

Respond with ONLY one word:
- "passport_ocr" : they want to use / upload / scan their passport image.
- "chat_history" : they want to reuse what they already told you earlier.
- "conversation" : they are dictating field values in this message.
- "none"         : no form-fill request.

USER MESSAGE:
{message}

ANSWER:"""


def classify_fill_source(llm: ChatVertexAI, message: str) -> DataSource:
    """Decide which data source the user is asking to fill from."""

    lowered = message.lower()

    # Fast keyword hints.
    if any(k in lowered for k in ["passport", "scan", "upload my passport"]):
        return DataSource.PASSPORT_OCR
    if any(k in lowered for k in ["already told", "what i said", "from our chat",
                                  "use my details", "reuse"]):
        return DataSource.CHAT_HISTORY

    try:
        response = llm.invoke(_FILL_SOURCE_PROMPT.format(message=message))
        label = (response.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fill-source classification failed: %s", exc)
        return DataSource.NONE

    for source in DataSource:
        if source.value in label:
            return source
    return DataSource.NONE


_EXTRACT_KEYS = ", ".join(sorted(CANONICAL_KEYS))

_EXTRACT_MESSAGE_PROMPT = """Extract registration field values from the user's message.

Return ONLY a compact JSON object mapping canonical keys to string values.
Allowed keys: {keys}

Rules:
- Only include keys the user actually provided. Omit everything else.
- Normalize dates to ISO YYYY-MM-DD when possible.
- Do NOT invent values. Output {{}} if nothing is present.

USER MESSAGE:
{message}

JSON:"""

_EXTRACT_HISTORY_PROMPT = """Extract reusable registration field values from the
conversation so far.

Return ONLY a compact JSON object mapping canonical keys to string values.
Allowed keys: {keys}

Rules:
- Only include values the user ALREADY provided in the conversation.
- Do NOT invent values. Output {{}} if nothing is present.
- Normalize dates to ISO YYYY-MM-DD when possible.

CONVERSATION:
{history}

JSON:"""


def _parse_json_object(raw: str) -> dict[str, str]:
    """Best-effort parse of an LLM JSON object response into str->str."""

    import json

    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Extract the first {...} block if the model added prose.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text or "{}")
    except (ValueError, TypeError):
        return {}

    if not isinstance(data, dict):
        return {}

    # Keep only canonical keys with non-empty string values.
    result: dict[str, str] = {}
    for key, value in data.items():
        if key in CANONICAL_KEYS and value not in (None, ""):
            result[key] = str(value).strip()
    return result


def extract_fields_from_message(llm: ChatVertexAI, message: str) -> dict[str, str]:
    """LLM extracts canonical field values from the current message."""

    try:
        response = llm.invoke(
            _EXTRACT_MESSAGE_PROMPT.format(keys=_EXTRACT_KEYS, message=message)
        )
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Field extraction failed: {exc}") from exc

    return _parse_json_object(response.content)


def extract_fields_from_history(
    llm: ChatVertexAI, history: list[dict]
) -> dict[str, str]:
    """Consent-gated: mine prior chat turns for reusable field values."""

    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in (history or [])
    )

    if not transcript.strip():
        return {}

    try:
        response = llm.invoke(
            _EXTRACT_HISTORY_PROMPT.format(keys=_EXTRACT_KEYS, history=transcript)
        )
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"History extraction failed: {exc}") from exc

    return _parse_json_object(response.content)


# ============================================================
# 7. OCR PIPELINE (GCP Cloud Vision + passport parsing)
# ============================================================


def build_ocr_client(settings: Settings):
    """Create a Cloud Vision client (keyless ADC). Imported lazily."""

    from google.cloud import vision

    return vision.ImageAnnotatorClient()


def run_passport_ocr(client, image_bytes: bytes, settings: Settings) -> str:
    """Run Cloud Vision document_text_detection; return full text incl. MRZ.

    Preconditions: consent granted (enforced by caller); non-empty image within
    the size limit. Does NOT persist the image.
    """

    if not image_bytes:
        raise OcrQualityError("No image provided.")
    if len(image_bytes) > settings.max_image_bytes:
        raise OcrQualityError("Image is too large.")

    from google.cloud import vision

    try:
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
    except Exception as exc:  # noqa: BLE001
        raise OcrError(f"OCR service error: {exc}") from exc

    if response.error.message:
        raise OcrError(response.error.message)

    text = response.full_text_annotation.text if response.full_text_annotation else ""

    if len(text.strip()) < 20:
        raise OcrQualityError("Could not read enough text from the image.")

    return text


# --- MRZ (TD3 passport) parsing -------------------------------------------

_MRZ_COUNTRY_TO_NAME = {
    "IND": "Indian", "USA": "American", "GBR": "British", "DEU": "German",
    "FRA": "French", "KEN": "Kenyan", "ETH": "Ethiopian",
}


def _mrz_date_to_iso(yymmdd: str, expiry: bool = False) -> Optional[str]:
    """Convert MRZ YYMMDD to ISO YYYY-MM-DD with a century heuristic."""

    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:6]
    # DOB: 20xx if <= current 2-digit year else 19xx. Expiry: always 20xx.
    now_yy = datetime.now().year % 100
    century = 2000 if (expiry or yy <= now_yy) else 1900
    year = century + yy
    try:
        datetime(year, int(mm), int(dd))
    except ValueError:
        return None
    return f"{year:04d}-{mm}-{dd}"


def parse_passport_fields(ocr_text: str) -> dict[str, str]:
    """Parse OCR text into canonical passport fields (MRZ preferred)."""

    fields: dict[str, str] = {}

    # Find TD3 MRZ: two lines of ~44 chars using A-Z0-9<.
    mrz_lines = [
        ln.replace(" ", "")
        for ln in ocr_text.splitlines()
        if re.fullmatch(r"[A-Z0-9<]{30,44}", ln.replace(" ", ""))
    ]

    if len(mrz_lines) >= 2:
        line1, line2 = mrz_lines[-2], mrz_lines[-1]

        # Line 1: P<ISSUING<SURNAME<<GIVEN<NAMES
        m = re.match(r"P[A-Z<]?([A-Z]{3})(.+)", line1)
        if m:
            issuing = m.group(1)
            fields["passport_issuing_country"] = _MRZ_COUNTRY_TO_NAME.get(
                issuing, issuing
            )
            names = m.group(2)
            if "<<" in names:
                surname_part, given_part = names.split("<<", 1)
                surname = surname_part.replace("<", " ").strip()
                given = given_part.replace("<", " ").strip()
                if surname:
                    fields["surname"] = surname.title()
                if given:
                    fields["given_name"] = given.title()
                if surname or given:
                    fields["full_name"] = f"{given.title()} {surname.title()}".strip()

        # Line 2: passport_no(9) chk(1) nationality(3) dob(6) chk sex(1) exp(6)...
        if len(line2) >= 28:
            passport_no = line2[0:9].replace("<", "").strip()
            nationality = line2[10:13]
            dob = line2[13:19]
            sex = line2[20:21]
            expiry = line2[21:27]

            if passport_no:
                fields["passport_no"] = passport_no
            if re.fullmatch(r"[A-Z]{3}", nationality):
                fields["nationality"] = _MRZ_COUNTRY_TO_NAME.get(
                    nationality, nationality
                )
            dob_iso = _mrz_date_to_iso(dob)
            if dob_iso:
                fields["date_of_birth"] = dob_iso
            if sex in ("M", "F"):
                fields["gender"] = "Male" if sex == "M" else "Female"
            exp_iso = _mrz_date_to_iso(expiry, expiry=True)
            if exp_iso:
                fields["passport_expiry_date"] = exp_iso

    # Fallback: labeled-line heuristics for anything MRZ didn't yield.
    if "passport_no" not in fields:
        m = re.search(r"passport\s*(?:no|number)[:\s]+([A-Z0-9]{6,10})",
                      ocr_text, re.IGNORECASE)
        if m:
            fields["passport_no"] = m.group(1).upper()

    return fields


# ============================================================
# 8. CONSENT
# ============================================================


def request_consent(source: DataSource) -> str:
    if source == DataSource.PASSPORT_OCR:
        return (
            "I can read your passport image to fill the form. Your image is "
            "used only to extract the details and is not saved. Do you consent? "
            "(yes/no)"
        )
    if source == DataSource.CHAT_HISTORY:
        return (
            "I can reuse the details you already shared in our chat (like name "
            "and travel dates) to fill the form. Is that okay? (yes/no)"
        )
    return "Do you want me to proceed? (yes/no)"


def interpret_consent_reply(message: str) -> Optional[bool]:
    """Return True (yes), False (no), or None (not a clear yes/no)."""

    lowered = message.strip().lower()
    if lowered in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "please do",
                   "go ahead", "confirm"}:
        return True
    if lowered in {"no", "n", "nope", "don't", "dont", "cancel", "stop"}:
        return False
    if re.search(r"\b(yes|sure|okay|ok|go ahead|proceed)\b", lowered):
        return True
    if re.search(r"\b(no|don't|dont|cancel|stop|nope)\b", lowered):
        return False
    return None


def update_consent(
    state: ConsentState, source: DataSource, granted: bool
) -> ConsentState:
    status = ConsentStatus.GRANTED if granted else ConsentStatus.DENIED
    if source == DataSource.PASSPORT_OCR:
        return ConsentState(passport_ocr=status, chat_history=state.chat_history)
    if source == DataSource.CHAT_HISTORY:
        return ConsentState(passport_ocr=state.passport_ocr, chat_history=status)
    return state


# ============================================================
# 9. FORM ASSISTANT SERVICE
# ============================================================


def _summarize_command(command: FormFillCommand) -> str:
    parts = []
    if command.navigate_to:
        page = command.navigate_to["page"].replace("_", " ")
        parts.append(f"opening the {page} page")
    if command.field_values:
        n = len(command.field_values)
        src = f" from your {command.source.replace('_', ' ')}" if command.source else ""
        parts.append(f"filling {n} field(s){src}")

    if not parts:
        return (
            "Tell me what you'd like to enter, or say 'use my passport' or "
            "'use what I already told you' and I'll help fill the form."
        )

    return (
        "I've " + " and ".join(parts) + ". Please review the details and submit "
        "the form yourself when you're happy with them."
    )


@dataclass
class FormAssistant:
    """Navigates + fills the registration form. Never writes DB, never submits."""

    settings: Settings
    llm: ChatVertexAI
    ocr_client: object

    @classmethod
    def bootstrap(cls, settings: Optional[Settings] = None) -> "FormAssistant":
        settings = settings or load_settings()

        logger.info("Initializing form assistant (Gemini + Vision, ADC)")
        llm = ChatVertexAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        ocr_client = build_ocr_client(settings)

        return cls(settings=settings, llm=llm, ocr_client=ocr_client)

    def handle(
        self,
        message: str,
        *,
        user_id: Optional[int] = None,
        history: Optional[list[dict]] = None,
        image_bytes: Optional[bytes] = None,
        consent: Optional[ConsentState] = None,
    ) -> FormAssistantResult:
        """See design pseudocode. Returns text + FormFillCommand.

        Never opens a DB session; never includes a submit action.
        """

        consent = consent or ConsentState()
        history = history or []
        command = FormFillCommand()

        # 1) Navigation (independent of consent).
        nav = detect_navigation_intent(message)
        if nav:
            command.navigate_to = nav

        # 2) If awaiting a consent answer, resolve yes/no first.
        awaiting = consent.awaiting_source()
        if awaiting is not None:
            decision = interpret_consent_reply(message)
            if decision is None:
                return FormAssistantResult(
                    "Please answer yes or no so I know how to proceed.",
                    command if command.navigate_to else None,
                    consent,
                )
            consent = update_consent(consent, awaiting, decision)
            if not decision:
                return FormAssistantResult(
                    "Okay, I won't use that. You can fill the form in yourself, "
                    "or tell me the details and I'll enter them.",
                    command if command.navigate_to else None,
                    consent,
                )
            # granted -> perform that source's fill below.
            source = awaiting
        else:
            source = classify_fill_source(self.llm, message)

        # 3a) Passport OCR (consent gate).
        if source == DataSource.PASSPORT_OCR:
            if consent.passport_ocr != ConsentStatus.GRANTED:
                consent = ConsentState(
                    passport_ocr=ConsentStatus.AWAITING,
                    chat_history=consent.chat_history,
                )
                return FormAssistantResult(
                    request_consent(DataSource.PASSPORT_OCR),
                    command if command.navigate_to else None,
                    consent,
                )
            assert consent.passport_ocr == ConsentStatus.GRANTED
            if not image_bytes:
                return FormAssistantResult(
                    "Please upload your passport image and I'll read it.",
                    command if command.navigate_to else None,
                    consent,
                )
            try:
                text = run_passport_ocr(self.ocr_client, image_bytes, self.settings)
                canonical = parse_passport_fields(text)
            except OcrQualityError:
                return FormAssistantResult(
                    "I couldn't read the passport clearly. Please try a sharper, "
                    "well-lit photo.",
                    command if command.navigate_to else None,
                    consent,
                )
            except OcrError:
                return FormAssistantResult(
                    "Passport reading is unavailable right now. You can type the "
                    "details instead.",
                    command if command.navigate_to else None,
                    consent,
                )
            finally:
                image_bytes = None  # transient: never persisted
            command.field_values = map_to_form_fields(canonical)
            command.source = DataSource.PASSPORT_OCR.value

        # 3b) Chat history (consent gate).
        elif source == DataSource.CHAT_HISTORY:
            if consent.chat_history != ConsentStatus.GRANTED:
                consent = ConsentState(
                    passport_ocr=consent.passport_ocr,
                    chat_history=ConsentStatus.AWAITING,
                )
                return FormAssistantResult(
                    request_consent(DataSource.CHAT_HISTORY),
                    command if command.navigate_to else None,
                    consent,
                )
            assert consent.chat_history == ConsentStatus.GRANTED
            try:
                canonical = extract_fields_from_history(self.llm, history)
            except ExtractionError:
                return FormAssistantResult(
                    "I couldn't pull details from our chat. Please tell me what "
                    "to enter.",
                    command if command.navigate_to else None,
                    consent,
                )
            command.field_values = map_to_form_fields(canonical)
            command.source = DataSource.CHAT_HISTORY.value

        # 3c) Conversational (no consent needed).
        elif source == DataSource.CONVERSATION:
            try:
                canonical = extract_fields_from_message(self.llm, message)
            except ExtractionError:
                canonical = {}
            command.field_values = map_to_form_fields(canonical)
            if command.field_values:
                command.source = DataSource.CONVERSATION.value

        # 4) Compose reply. NEVER submit; NEVER write DB.
        has_command = bool(command.navigate_to or command.field_values)
        answer = _summarize_command(command)
        return FormAssistantResult(
            answer,
            command if has_command else None,
            consent,
        )


# ============================================================
# 10. CLI (standalone smoke test)
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
    print("Try: 'take me to registration', 'use my passport', "
          "'use what I already told you', or dictate fields.")

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
        result = assistant.handle(message, history=history, consent=consent)
        consent = result.consent

        print(f"\nAssistant: {result.answer}")
        if result.command:
            if result.command.navigate_to:
                print(f"  [navigate] {result.command.navigate_to['url']}")
            if result.command.field_values:
                print(f"  [fill] {result.command.field_values}")

        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    raise SystemExit(main())
