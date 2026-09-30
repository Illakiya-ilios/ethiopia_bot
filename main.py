"""Supervisor agent — routes a user's question to the right assistant.

This is the orchestration layer (bot #4) that connects the two assistants
built so far using a LangGraph supervisor:

    - Discovery  -> rag.py         (general Ethiopia tourism knowledge, RAG)
    - Bookings   -> query_assistant (the user's OWN booking data, per-user SQL)

Flow:

    User question ─▶ Supervisor (Gemini classifies intent + language)
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Discovery node          Bookings node
      (RagService.ask)     (QueryAssistant.ask, user-scoped)
             │                     │
             └──────────┬──────────┘
                        ▼
                       END

Security note:
    The authenticated ``user_id`` is supplied by the trusted caller at invoke
    time and flows through graph state to the bookings node. The supervisor
    never derives or changes it, so per-user data isolation is preserved. In
    production the id comes from a validated session token, never user input.

Run standalone:

    python main.py 2        # session scoped to user_id=2
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, StateGraph

from rag import RagService
from query_assistant import QueryAssistant, load_settings as load_query_settings
from form_assistant import (
    ConsentState,
    ConsentStatus,
    FormAssistant,
    FormAssistantResult,
)
from endpoints import SITE_PAGES, festival_register_url, match_package_slug


def _consent_status(value) -> "ConsentStatus":
    """Coerce a stored consent string back into a ConsentStatus enum."""
    try:
        return ConsentStatus(value) if value else ConsentStatus.NOT_ASKED
    except ValueError:
        return ConsentStatus.NOT_ASKED


logger = logging.getLogger("supervisor")


# ============================================================
# 1. GRAPH STATE
# ============================================================


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    question: str
    user_id: int
    route: str                    # discovery | bookings | navigate | register
    answer: str
    navigation: Optional[dict]    # {"page": str, "url": str} for the frontend

    # --- form-fill (register route) state, carried across turns ---
    history: list                 # prior chat turns [{role, content}]
    image_bytes: Optional[bytes]  # transient passport upload (never persisted)
    consent: object               # ConsentState
    command: Optional[dict]       # FormFillCommand.to_dict() for the frontend


Route = Literal["discovery", "bookings", "navigate", "register"]


@dataclass
class ChatResult:
    """Structured result returned to the API/frontend.

    - ``navigation`` is non-None when the user asked to open a page.
    - ``command`` is non-None on the register route: it carries navigation +
      field values the frontend applies to the form (never a submit, no DB).
    - ``consent`` is echoed back so the frontend can persist it across turns.
    """

    answer: str
    route: str
    navigation: Optional[dict] = None
    command: Optional[dict] = None
    consent: Optional[dict] = None


# ============================================================
# 2. INTENT CLASSIFIER (SUPERVISOR BRAIN)
# ============================================================


ROUTER_PROMPT = """You are a router for an Ethiopia tourism assistant.

Classify the user's message into exactly ONE category:

- "register": the user wants help FILLING OUT the registration form — using
  their passport image, reusing details they already gave, or dictating field
  values. Also when replying yes/no to a form-fill consent question.
  Signals: "fill the form", "use my passport", "scan my passport",
  "use what I already told you", "fill in my details", "help me register the form".

- "navigate": the user wants to GO TO / OPEN a page on the website — e.g.
  "take me to registration", "open the visa page", "go to my packages",
  "show me the registration form".
  Signals: "take me to", "open", "go to", "navigate", "show me the ... page".

- "bookings": questions about the user's OWN account data — their package
  requests, their trips, who is travelling with them (their family/
  passengers), their costs/budget, or the status of their visa applications.
  Signals: "my", "I booked", "our trip", "my visa status", "how much did I".

- "discovery": general questions about Ethiopia tourism — destinations,
  attractions, culture, history, visa rules in general, travel tips,
  itineraries, "what can I do / see / visit", recommendations.

Respond with ONLY one word: register OR navigate OR bookings OR discovery.

USER MESSAGE:
{question}

CATEGORY:"""


def classify_intent(llm: ChatVertexAI, question: str) -> Route:
    """Ask Gemini to pick a route; default to discovery if unclear."""

    try:
        response = llm.invoke(ROUTER_PROMPT.format(question=question))
        label = (response.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.error("Intent classification failed: %s", exc)
        return "discovery"

    if "register" in label:
        return "register"

    if "navigate" in label:
        return "navigate"

    if "booking" in label:
        return "bookings"

    return "discovery"


# ============================================================
# 2b. NAVIGATION TARGET RESOLUTION
# ============================================================


# Keywords mapped to a known page id in endpoints.SITE_PAGES.
_PAGE_KEYWORDS = {
    "packages": ["packages", "package", "itinerary", "itineraries", "trip",
                 "tour", "browse packages"],
    "calendar": ["calendar", "festival calendar", "dates", "schedule"],
    "flagship": ["flagship", "flagship festival", "main festival"],
    "heritage": ["heritage", "history", "historic", "culture"],
    "stopover": ["stopover", "stop over", "layover"],
    "closing": ["closing", "strategy", "plan"],
    "home": ["home", "homepage", "main page", "start", "top", "landing"],
}


NAV_PROMPT = """The user wants to open a page on the Ethiopia tourism website.

Available pages: {pages}

Given the user's message, respond with ONLY the single page id that best
matches, or "unknown" if none clearly matches.

USER MESSAGE:
{question}

PAGE ID:"""


def resolve_navigation_target(
    llm: ChatVertexAI, question: str
) -> Optional[dict]:
    """Map a navigation request to a known page dict {"page", "url"}.

    Tries fast keyword matching first, then falls back to the LLM. Returns
    None if no page can be confidently determined.
    """

    lowered = question.lower()

    # 0) Specific package/festival deep-link (path-based /packages/<slug>).
    slug = match_package_slug(lowered)
    if slug:
        return {"page": f"package_{slug}", "url": festival_register_url(slug)}

    # 1) Fast path: keyword match against landing-page sections.
    for page_id, keywords in _PAGE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords) and page_id in SITE_PAGES:
            return {"page": page_id, "url": SITE_PAGES[page_id]}

    # 2) LLM fallback for less obvious phrasing.
    try:
        response = llm.invoke(
            NAV_PROMPT.format(
                pages=", ".join(SITE_PAGES.keys()), question=question
            )
        )
        page_id = (response.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.error("Navigation resolution failed: %s", exc)
        return None

    if page_id in SITE_PAGES:
        return {"page": page_id, "url": SITE_PAGES[page_id]}

    return None


# ============================================================
# 3. GRAPH BUILDER
# ============================================================


class Supervisor:
    """Builds and runs the LangGraph routing graph over both assistants."""

    def __init__(
        self,
        router_llm: ChatVertexAI,
        rag_service: RagService,
        query_assistant: QueryAssistant,
        form_assistant: FormAssistant,
    ) -> None:
        self.router_llm = router_llm
        self.rag_service = rag_service
        self.query_assistant = query_assistant
        self.form_assistant = form_assistant
        self.graph = self._build_graph()

    # ---- node implementations ------------------------------------------

    def _supervisor_node(self, state: AgentState) -> AgentState:
        route = classify_intent(self.router_llm, state["question"])
        logger.info("Routed to: %s", route)
        return {"route": route}

    def _discovery_node(self, state: AgentState) -> AgentState:
        answer = self.rag_service.ask(state["question"])
        return {"answer": answer}

    def _bookings_node(self, state: AgentState) -> AgentState:
        user_id = state.get("user_id")

        if not user_id:
            # No authenticated identity -> cannot serve personal data.
            return {
                "answer": (
                    "I can only look up your bookings when you're signed in. "
                    "Please log in and try again."
                )
            }

        answer = self.query_assistant.ask(state["question"], user_id=user_id)
        return {"answer": answer}

    def _navigate_node(self, state: AgentState) -> AgentState:
        target = resolve_navigation_target(self.router_llm, state["question"])

        if not target:
            available = ", ".join(SITE_PAGES.keys())
            return {
                "answer": (
                    "I'm not sure which page you mean. I can take you to: "
                    f"{available}. Which one would you like?"
                ),
                "navigation": None,
            }

        return {
            "answer": (
                f"Sure — opening the {target['page'].replace('_', ' ')} page "
                "for you now."
            ),
            "navigation": target,
        }

    def _register_node(self, state: AgentState) -> AgentState:
        result: FormAssistantResult = self.form_assistant.handle(
            state["question"],
            user_id=state.get("user_id"),
            history=state.get("history", []),
            image_bytes=state.get("image_bytes"),
            consent=state.get("consent") or ConsentState(),
        )

        command = result.command
        return {
            "answer": result.answer,
            "command": command.to_dict() if command else None,
            "navigation": command.navigate_to if command else None,
            "consent": result.consent,
            "image_bytes": None,   # clear transient image after handling
        }

    # ---- routing edge ---------------------------------------------------

    @staticmethod
    def _route_selector(state: AgentState) -> Route:
        return state.get("route", "discovery")  # type: ignore[return-value]

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("discovery", self._discovery_node)
        graph.add_node("bookings", self._bookings_node)
        graph.add_node("navigate", self._navigate_node)
        graph.add_node("register", self._register_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_selector,
            {
                "discovery": "discovery",
                "bookings": "bookings",
                "navigate": "navigate",
                "register": "register",
            },
        )
        graph.add_edge("discovery", END)
        graph.add_edge("bookings", END)
        graph.add_edge("navigate", END)
        graph.add_edge("register", END)

        return graph.compile()

    # ---- public API -----------------------------------------------------

    def chat(
        self,
        question: str,
        user_id: Optional[int] = None,
        *,
        history: Optional[list] = None,
        image_bytes: Optional[bytes] = None,
        consent: Optional[dict] = None,
    ) -> ChatResult:
        """Route and answer, returning a structured result for the frontend.

        ``user_id`` must come from a trusted/authenticated session; it is only
        used by the bookings path and never influences routing.

        Form-fill context (register route) is passed through and echoed back:
        - ``history``: prior chat turns, for chat-history extraction.
        - ``image_bytes``: transient passport upload (never persisted).
        - ``consent``: dict form of ConsentState, persisted by the frontend
          across turns.
        """

        initial: AgentState = {"question": question}
        if user_id:
            initial["user_id"] = user_id
        if history:
            initial["history"] = history
        if image_bytes:
            initial["image_bytes"] = image_bytes
        if consent:
            initial["consent"] = ConsentState(
                passport_ocr=_consent_status(consent.get("passport_ocr")),
                chat_history=_consent_status(consent.get("chat_history")),
            )

        result = self.graph.invoke(initial)

        consent_obj = result.get("consent")
        consent_out = (
            {
                "passport_ocr": consent_obj.passport_ocr.value,
                "chat_history": consent_obj.chat_history.value,
            }
            if consent_obj is not None
            else None
        )

        return ChatResult(
            answer=result.get("answer", "Sorry, I couldn't produce an answer."),
            route=result.get("route", "discovery"),
            navigation=result.get("navigation"),
            command=result.get("command"),
            consent=consent_out,
        )

    def ask(self, question: str, user_id: Optional[int] = None) -> str:
        """Convenience wrapper returning just the answer text (CLI/back-compat)."""

        return self.chat(question, user_id=user_id).answer


# ============================================================
# 4. BOOTSTRAP
# ============================================================


def build_supervisor() -> Supervisor:
    """Wire up the router LLM and both underlying assistants."""

    load_dotenv()

    settings = load_query_settings()

    router_llm = ChatVertexAI(
        model=settings.llm_model,
        temperature=0.0,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )

    logger.info("Bootstrapping discovery (RAG) service...")
    rag_service = RagService.bootstrap()

    logger.info("Bootstrapping bookings (query) assistant...")
    query_assistant = QueryAssistant.bootstrap(settings)

    logger.info("Bootstrapping registration (form-fill) assistant...")
    form_assistant = FormAssistant.bootstrap()

    return Supervisor(
        router_llm=router_llm,
        rag_service=rag_service,
        query_assistant=query_assistant,
        form_assistant=form_assistant,
    )


# ============================================================
# 5. LOGGING
# ============================================================


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
# 6. CLI
# ============================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ethiopia Tourism Assistant — supervisor over both bots."
    )
    parser.add_argument(
        "user_id",
        type=int,
        nargs="?",
        help=(
            "Authenticated user id for personal booking questions. Omit to run "
            "in discovery-only mode. In production this comes from the auth "
            "layer, not user input."
        ),
    )
    args = parser.parse_args()

    import os

    configure_logging(os.getenv("LOG_LEVEL", "WARNING"))

    try:
        supervisor = build_supervisor()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start: %s", exc)
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    print("\n===================================")
    print(" Ethiopia Tourism AI Assistant")
    print("===================================")
    if args.user_id:
        print(f"Signed in as user_id={args.user_id}.")
        print("Ask about Ethiopia tourism OR your own bookings.")
    else:
        print("Discovery mode (not signed in). Ask about Ethiopia tourism.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            return 0

        if not question:
            continue

        try:
            answer = supervisor.ask(question, user_id=args.user_id)
            print(f"\nAssistant: {answer}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed: %s", exc)
            print("\nAssistant: Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    raise SystemExit(main())
