"""Streamlit UI for the Ethiopia Tourism AI Assistant.

A presentation-friendly chat front-end in the Ethiopian flag palette that
wraps the LangGraph supervisor (main.py). It routes between:

    - Discovery : general Ethiopia tourism knowledge (rag.py)
    - Bookings  : the signed-in user's own data (query_assistant.py)

A lightweight "login" (pick a traveller) establishes the authenticated
``user_id`` so personal booking questions are scoped correctly — mirroring how
a real auth layer would supply the id. Guests get discovery-only.

Run:

    streamlit run app_ui.py
"""

from __future__ import annotations

import streamlit as st

from model import SessionLocal, User, UserRole
from main import build_supervisor


# ============================================================
# 1. PAGE CONFIG + THEME (Ethiopian flag palette)
# ============================================================

# Flag colors: green / yellow / red, with the blue national disc.
GREEN = "#078930"
YELLOW = "#FCDD09"
RED = "#DA121A"
BLUE = "#0F47AF"

st.set_page_config(
    page_title="Ethiopia Tourism AI Assistant",
    page_icon="🇪🇹",
    layout="centered",
)

st.markdown(
    f"""
    <style>
    /* App background: subtle flag-inspired gradient */
    .stApp {{
        background: linear-gradient(160deg, #f4fbf4 0%, #fffdf0 50%, #fdf4f4 100%);
        color: #1a1a1a;
    }}

    /* Force readable dark text everywhere (guards against dark theme) */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] * {{
        color: #1a1a1a !important;
    }}

    /* Header banner in flag stripes */
    .flag-banner {{
        display: flex;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        margin-bottom: 0.5rem;
    }}
    .flag-banner div {{ height: 10px; flex: 1; }}

    .app-title {{
        text-align: center;
        font-weight: 800;
        font-size: 2rem;
        margin: 0.2rem 0 0 0;
        background: linear-gradient(90deg, {GREEN}, {BLUE}, {RED});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .app-sub {{
        text-align: center;
        color: #444;
        margin-bottom: 0.6rem;
    }}

    /* Chat bubbles — light background with dark text */
    [data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
        background: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        color: #1a1a1a;
    }}
    /* Assistant bubble gets a subtle green tint to match the theme */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {{
        background: #f3faf4;
        border-color: {GREEN}33;
    }}

    /* Primary buttons in Ethiopian green */
    .stButton > button {{
        background: {GREEN};
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background: {BLUE};
        color: white;
    }}

    /* Sidebar accent */
    [data-testid="stSidebar"] {{
        border-right: 4px solid {YELLOW};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def flag_banner() -> None:
    st.markdown(
        f"""
        <div class="flag-banner">
            <div style="background:{GREEN}"></div>
            <div style="background:{YELLOW}"></div>
            <div style="background:{RED}"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 2. CACHED RESOURCES
# ============================================================


@st.cache_resource(show_spinner="Warming up the assistant...")
def get_supervisor():
    """Bootstrap the LangGraph supervisor once and reuse across reruns."""
    return build_supervisor()


@st.cache_data(show_spinner=False)
def get_travellers() -> list[dict]:
    """Fetch selectable traveller accounts (non-admin) for the demo login."""
    with SessionLocal() as session:
        users = (
            session.query(User)
            .filter(User.role == UserRole.USER)
            .order_by(User.full_name)
            .all()
        )
        return [
            {"id": u.id, "name": u.full_name or f"User {u.id}", "email": u.email}
            for u in users
        ]


# ============================================================
# 3. SIDEBAR — LOGIN / SESSION
# ============================================================


def render_sidebar() -> None:
    with st.sidebar:
        st.header("👤 Session")
        st.caption(
            "Sign in as a traveller to ask about your own bookings, or "
            "continue as a guest for general tourism questions."
        )

        travellers = get_travellers()
        options = {"Guest (discovery only)": None}
        for t in travellers:
            options[f"{t['name']} · {t['email']}"] = t["id"]

        choice = st.selectbox("Signed in as", list(options.keys()))
        user_id = options[choice]

        # Reset chat when identity changes.
        if st.session_state.get("user_id") != user_id:
            st.session_state.user_id = user_id
            st.session_state.messages = []

        if user_id:
            st.success(f"Signed in (user_id={user_id})")
        else:
            st.info("Guest mode — bookings questions are disabled.")

        if st.button("Clear conversation"):
            st.session_state.messages = []

        st.divider()
        st.caption("🇪🇹 Ethiopia Tourism AI · POC")


# ============================================================
# 4. MAIN CHAT
# ============================================================


def render_header() -> None:
    flag_banner()
    st.markdown('<div class="app-title">Ethiopia Tourism AI Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Discover Ethiopia · Manage your journey</div>',
        unsafe_allow_html=True,
    )


SUGGESTIONS = [
    "What are the top attractions in Lalibela?",
    "Suggest a 5-day itinerary in Ethiopia",
    "What packages have I booked?",
    "Who is travelling with me?",
]


def main() -> None:
    render_header()
    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggestion chips (only before the first message).
    if not st.session_state.messages:
        st.write("Try asking:")
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            if cols[i % 2].button(suggestion, key=f"sugg_{i}"):
                st.session_state.pending = suggestion

    # Render history.
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🇪🇹"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input: either a typed prompt or a clicked suggestion.
    prompt = st.chat_input("Ask about Ethiopia or your bookings...")
    if not prompt and st.session_state.get("pending"):
        prompt = st.session_state.pop("pending")

    if not prompt:
        return

    # Show the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Get the assistant's answer via the supervisor.
    with st.chat_message("assistant", avatar="🇪🇹"):
        with st.spinner("Thinking..."):
            try:
                supervisor = get_supervisor()
                answer = supervisor.ask(
                    prompt, user_id=st.session_state.get("user_id")
                )
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "Sorry, something went wrong reaching the assistant. "
                    f"({exc})"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
