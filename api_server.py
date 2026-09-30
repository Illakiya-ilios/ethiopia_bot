"""FastAPI server exposing the Ethiopia Tourism AI Assistant to the website.

The bot is deployed as a backend service and embedded on every page of the
React site (http://4.240.116.35:3000). The frontend calls this API; the bot
returns an answer plus an optional navigation directive the frontend uses to
open a page.

Endpoints:
    GET  /health          -> liveness probe
    POST /chat            -> { answer, route, navigation }

The supervisor is bootstrapped once at startup and reused across requests.

Run (dev):
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

Auth/trust model:
    ``user_id`` is supplied by the website's authenticated session and passed
    with each request. It is used only for personal (bookings) answers and
    never influences routing. The bot performs no DB writes and never submits
    forms — navigation only.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from endpoints import SITE_BASE_URL, SITE_PAGES
from main import Supervisor, build_supervisor, configure_logging


logger = logging.getLogger("api_server")


# ============================================================
# 1. REQUEST / RESPONSE MODELS
# ============================================================


class ChatRequest(BaseModel):
    """Incoming chat request from the website."""

    message: str = Field(..., description="The user's message.")
    user_id: Optional[int] = Field(
        default=None,
        description=(
            "Authenticated user id from the website session. Required for "
            "personal booking questions; omit for anonymous/discovery use."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque client session id (reserved for future memory).",
    )
    history: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Prior chat turns [{role, content}], used for the register route's "
            "chat-history form-fill. The frontend maintains this."
        ),
    )
    consent: Optional[dict] = Field(
        default=None,
        description=(
            "Consent state {passport_ocr, chat_history} echoed from the last "
            "response; the frontend persists it across turns."
        ),
    )
    image_base64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded passport image for OCR (register route). Transient: "
            "used only to extract fields, never stored."
        ),
    )


class NavigationDirective(BaseModel):
    page: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    navigation: Optional[NavigationDirective] = None
    # register-route extras:
    command: Optional[dict] = None       # {navigate_to, field_values, source}
    consent: Optional[dict] = None       # {passport_ocr, chat_history}


# ============================================================
# 2. APP + LIFECYCLE
# ============================================================


app = FastAPI(
    title="Ethiopia Tourism AI Assistant",
    version="1.0.0",
    description="Chat API for the Ethiopia tourism website assistant.",
)


# Allow the React site (and configurable extra origins) to call the API from
# the browser. Defaults to the deployed site origin.
_allowed = os.getenv("CORS_ORIGINS", SITE_BASE_URL)
_origins = [o.strip() for o in _allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# The supervisor is expensive to build (models + vector store), so build once.
_supervisor: Optional[Supervisor] = None


@app.on_event("startup")
def _startup() -> None:
    global _supervisor

    load_dotenv()
    configure_logging(os.getenv("LOG_LEVEL", "WARNING"))

    logger.info("Bootstrapping supervisor (models + vector store)...")
    _supervisor = build_supervisor()
    logger.info("Supervisor ready. Allowed CORS origins: %s", _origins)


def _get_supervisor() -> Supervisor:
    if _supervisor is None:  # pragma: no cover - should not happen post-startup
        raise RuntimeError("Supervisor not initialized yet.")
    return _supervisor


# ============================================================
# 3. ROUTES
# ============================================================


@app.get("/health")
def health() -> dict:
    """Liveness probe. Returns ready=false until the supervisor is built."""
    return {
        "status": "ok",
        "ready": _supervisor is not None,
        "site": SITE_BASE_URL,
        "pages": list(SITE_PAGES.keys()),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main entry point: route the message and return answer + navigation."""

    message = (request.message or "").strip()
    if not message:
        return ChatResponse(
            answer="Please type a message so I can help.",
            route="discovery",
            navigation=None,
        )

    # Decode a transient passport image if provided (register route only).
    image_bytes = None
    if request.image_base64:
        import base64
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to decode image_base64; ignoring.")

    try:
        result = _get_supervisor().chat(
            message,
            user_id=request.user_id,
            history=request.history,
            image_bytes=image_bytes,
            consent=request.consent,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat request failed: %s", exc)
        return ChatResponse(
            answer="Sorry, something went wrong. Please try again.",
            route="discovery",
            navigation=None,
        )
    finally:
        image_bytes = None  # do not retain image bytes

    navigation = (
        NavigationDirective(**result.navigation) if result.navigation else None
    )

    return ChatResponse(
        answer=result.answer,
        route=result.route,
        navigation=navigation,
        command=result.command,
        consent=result.consent,
    )
