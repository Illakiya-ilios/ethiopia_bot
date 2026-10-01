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

from langchain_google_genai import ChatGoogleGenerativeAI

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

    google_api_key: str
    llm_model: str = "gemini-2.0-flash-lite"
    # A vision-capable model for passport OCR (Gemini reads the image directly).
    ocr_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    max_image_bytes: int = 8 * 1024 * 1024

    log_level: str = "WARNING"

    def validate(self) -> None:
        if not self.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required.")


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        ocr_model=os.getenv("OCR_MODEL", "gemini-2.0-flash"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
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
    # party composition
    "adults_male", "adults_female", "children_count", "infants",
    "travellers_total",
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
    # party composition (PLACEHOLDER ids; map to the real form field names)
    "adults_male": "reg_adults_male",
    "adults_female": "reg_adults_female",
    "children_count": "reg_children_count",
    "infants": "reg_infants",
    "travellers_total": "reg_travellers_total",
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
    "packages": ["packages", "itineraries", "browse packages"],
    "calendar": ["festival calendar", "calendar"],
    "flagship": ["flagship festival", "flagship"],
    "heritage": ["heritage page", "history page", "heritage"],
    "stopover": ["stopover", "stop over", "layover"],
    "closing": ["closing section", "strategy page"],
    "home": ["homepage", "home page", "landing page"],
}

# Explicit navigation verbs — navigation is only triggered when the user
# clearly asks to go somewhere, not on every form-fill message.
_NAV_VERBS = ("navigate", "take me to", "open ", "go to", "show me the",
              "redirect", "bring me to", "visit the")


def detect_navigation_intent(message: str) -> Optional[dict]:
    """Return {"page", "url"} only when the user clearly wants to navigate.

    A specific package/festival name always deep-links. Otherwise a landing
    section only matches when an explicit navigation verb is present, so
    phrases like "we are planning..." don't accidentally navigate.
    """

    lowered = message.lower()

    # 1) Specific package/festival deep-link (path-based /packages/<slug>).
    slug = match_package_slug(lowered)
    if slug:
        return {"page": f"package_{slug}", "url": festival_register_url(slug)}

    # 2) Landing-page section — ONLY if the user used a navigation verb.
    if not any(verb in lowered for verb in _NAV_VERBS):
        return None

    for page_id, keywords in _PAGE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords) and page_id in SITE_PAGES:
            return {"page": page_id, "url": SITE_PAGES[page_id]}
    return None


# ============================================================
# 6. FILL-SOURCE + FIELD EXTRACTION (LLM)
# ============================================================


_FILL_SOURCE_PROMPT = """Classify how the user wants to fill the registration form.

Respond with ONLY one word:
- "passport_ocr" : they explicitly want to use/upload/scan a passport image.
- "chat_history" : they explicitly want to reuse what they told you earlier.
- "conversation" : they ask to fill the form AND/OR give any traveller details
                   in this message (names, number of people, dates, nationality,
                   etc.). This is the default whenever they want help filling it.
- "none"         : the message is unrelated to filling a form.

Examples:
- "can you fill this form for me, we are 3 people" -> conversation
- "fill the form, 1 man and 2 women" -> conversation
- "use my passport" -> passport_ocr
- "use what I told you earlier" -> chat_history

USER MESSAGE:
{message}

ANSWER:"""


def classify_fill_source(llm: ChatGoogleGenerativeAI, message: str) -> DataSource:
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
- For party / group size, map counts to:
    adults_male      = number of adult men
    adults_female    = number of adult women
    children_count   = number of children
    infants          = number of infants
    travellers_total = total number of travellers
  Example: "3 people, 1 man and 2 adult women" ->
    {{"adults_male": "1", "adults_female": "2", "travellers_total": "3"}}
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


def extract_fields_from_message(llm: ChatGoogleGenerativeAI, message: str) -> dict[str, str]:
    """LLM extracts canonical field values from the current message."""

    try:
        response = llm.invoke(
            _EXTRACT_MESSAGE_PROMPT.format(keys=_EXTRACT_KEYS, message=message)
        )
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Field extraction failed: {exc}") from exc

    return _parse_json_object(response.content)


def extract_fields_from_history(
    llm: ChatGoogleGenerativeAI, history: list[dict]
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
# 7. OCR PIPELINE (Gemini vision — reads the passport image directly)
# ============================================================


_PASSPORT_OCR_PROMPT = f"""You are reading a passport image. Extract the holder's
details and return ONLY a compact JSON object mapping canonical keys to string
values.

Allowed keys: {", ".join(sorted(CANONICAL_KEYS))}

Rules:
- Prefer the Machine Readable Zone (MRZ, the two lines of <<< at the bottom) for
  accuracy, but also use the printed labels.
- Normalize all dates to ISO YYYY-MM-DD.
- Use full country/nationality names (e.g. "Indian", not "IND").
- gender must be "Male" or "Female".
- Only include keys you can read with confidence. Omit anything unclear.
- If the image is not a readable passport, return {{}}.
- Output ONLY the JSON object, no prose, no markdown.
"""


def run_passport_ocr(
    llm: ChatGoogleGenerativeAI, image_bytes: bytes, settings: Settings
) -> dict[str, str]:
    """Read a passport image with Gemini vision and return canonical fields.

    Preconditions: consent granted (enforced by caller); non-empty image within
    the size limit. Does NOT persist the image.
    """

    if not image_bytes:
        raise OcrQualityError("No image provided.")
    if len(image_bytes) > settings.max_image_bytes:
        raise OcrQualityError("Image is too large.")

    import base64

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": _PASSPORT_OCR_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64}",
            },
        ],
    }

    try:
        response = llm.invoke([message])
    except Exception as exc:  # noqa: BLE001
        raise OcrError(f"OCR (Gemini vision) failed: {exc}") from exc

    fields = _parse_json_object(response.content)

    if not fields:
        raise OcrQualityError("Could not read passport details from the image.")

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
        parts.append(f"opened the {page} page")
    if command.field_values:
        n = len(command.field_values)
        src = f" from your {command.source.replace('_', ' ')}" if command.source else ""
        parts.append(f"filled {n} field(s){src}")

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
    llm: ChatGoogleGenerativeAI
    ocr_llm: ChatGoogleGenerativeAI

    @classmethod
    def bootstrap(cls, settings: Optional[Settings] = None) -> "FormAssistant":
        settings = settings or load_settings()

        logger.info("Initializing form assistant (Gemini API key)")
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
        # A vision-capable Gemini model for reading passport images.
        ocr_llm = ChatGoogleGenerativeAI(
            model=settings.ocr_model,
            temperature=0.0,
            google_api_key=settings.google_api_key,
        )

        return cls(settings=settings, llm=llm, ocr_llm=ocr_llm)

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
                canonical = run_passport_ocr(
                    self.ocr_llm, image_bytes, self.settings
                )
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
