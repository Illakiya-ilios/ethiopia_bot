"""Database Query Assistant — natural-language questions over tourism.db.

Bot #2 of the Ethiopia Tourism assistant suite. It answers analytical
questions about platform data (users, package requests, costs, passengers,
admin notifications, visa applications) by:

    1. Presenting the live DB schema to Vertex AI Gemini
    2. Asking Gemini to write a single read-only SQL SELECT
    3. Validating that SQL is safe (SELECT-only, no mutations)
    4. Executing it against the SQLite database
    5. Asking Gemini to turn the rows into a natural-language answer

Safety is the priority: the generated SQL is guarded so the model can never
INSERT/UPDATE/DELETE/DROP or run multiple statements. Execution additionally
uses a read-only connection where supported.

Runs standalone for testing:

    python query_assistant.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from langchain_google_vertexai import ChatVertexAI

from model import engine as default_engine


# ============================================================
# 0. EXCEPTIONS
# ============================================================


class QueryAssistantError(Exception):
    """Base error for the query assistant."""


class ConfigurationError(QueryAssistantError):
    """Raised when configuration is missing or invalid."""


class UnsafeQueryError(QueryAssistantError):
    """Raised when generated SQL fails the safety checks."""


class SqlGenerationError(QueryAssistantError):
    """Raised when the LLM fails to produce usable SQL."""


class QueryExecutionError(QueryAssistantError):
    """Raised when the SQL fails to execute."""


# ============================================================
# 1. LOGGING
# ============================================================


logger = logging.getLogger("query_assistant")


def configure_logging(level: str = "WARNING") -> None:
    """Configure root logging (quiet by default; set LOG_LEVEL=INFO for detail)."""

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
    """Environment-driven settings for the query assistant."""

    gcp_project_id: str
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-2.0-flash-lite"
    llm_temperature: float = 0.0
    max_rows: int = 100
    log_level: str = "WARNING"

    def validate(self) -> None:
        if not self.gcp_project_id:
            raise ConfigurationError("GCP_PROJECT_ID is required.")
        if self.max_rows <= 0:
            raise ConfigurationError("MAX_ROWS must be positive.")


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        gcp_project_id=os.getenv("GCP_PROJECT_ID", ""),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_rows=int(os.getenv("MAX_ROWS", "100")),
        log_level=os.getenv("LOG_LEVEL", "WARNING"),
    )

    settings.validate()
    return settings


# ============================================================
# 3. SCHEMA INTROSPECTION
# ============================================================


# Tables a user is NEVER allowed to see, even indirectly.
#   - users               : other people's accounts + password hashes
#   - admin_notifications : internal admin-facing data
_BLOCKED_TABLES = {"users", "admin_notifications"}

# The only "tables" the LLM is allowed to reference are these per-user secured
# views. They are materialized as CTEs (see build_secured_cte) that are
# pre-filtered to the authenticated user's own data via a bound parameter.
_SECURED_VIEWS = ("my_requests", "my_costs", "my_passengers", "my_visas")


def build_secured_cte() -> str:
    """Return the WITH clause that scopes all data to the current user.

    Every downstream query can only reference the ``my_*`` views produced
    here, and each is filtered by ``:current_user_id`` (a bound parameter, so
    it is injection-safe). This is the real security boundary — enforced in
    SQL, not in the prompt.
    """

    return (
        "WITH my_requests AS (\n"
        "    SELECT * FROM package_requests WHERE user_id = :current_user_id\n"
        "),\n"
        "my_costs AS (\n"
        "    SELECT c.* FROM package_request_costs c\n"
        "    JOIN my_requests r ON c.package_request_id = r.id\n"
        "),\n"
        "my_passengers AS (\n"
        "    SELECT p.* FROM package_request_passengers p\n"
        "    JOIN my_requests r ON p.package_request_id = r.id\n"
        "),\n"
        "my_visas AS (\n"
        "    SELECT v.* FROM visa_applications v\n"
        "    JOIN my_requests r ON v.package_request_id = r.id\n"
        ")"
    )


def build_user_schema_description(db_engine: Engine) -> str:
    """Render the schema of ONLY the per-user secured views for the prompt.

    The LLM is shown ``my_requests`` / ``my_costs`` / ``my_passengers`` /
    ``my_visas`` (which carry the same columns as their base tables) and is
    told these are the only tables it may reference. It never sees the raw
    tables, the users table, or admin data.
    """

    inspector = inspect(db_engine)

    # Map each secured view to its underlying base table for column info.
    view_to_base = {
        "my_requests": "package_requests",
        "my_costs": "package_request_costs",
        "my_passengers": "package_request_passengers",
        "my_visas": "visa_applications",
    }

    lines: List[str] = []

    for view_name, base_table in view_to_base.items():
        columns = inspector.get_columns(base_table)
        col_parts = [f"{col['name']} {col['type']}" for col in columns]
        lines.append(
            f"VIEW {view_name} (\n  " + ",\n  ".join(col_parts) + "\n)"
        )

    lines.append(
        "\nRELATIONSHIPS: my_costs, my_passengers, and my_visas each link to "
        "my_requests via package_request_id = my_requests.id."
    )

    return "\n\n".join(lines)


# ============================================================
# 4. SQL SAFETY GUARDS
# ============================================================


# Statements/keywords that must never appear in generated SQL.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|GRANT|REVOKE|MERGE)\b",
    re.IGNORECASE,
)


def _strip_sql_markdown(raw: str) -> str:
    """Remove ```sql fences and surrounding whitespace from an LLM response."""

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned.rstrip(";").strip()


# Base tables the model must never name directly — it must use the my_* views.
_PROTECTED_TABLE_REF = re.compile(
    r"\b(users|admin_notifications|package_requests|package_request_costs|"
    r"package_request_passengers|visa_applications)\b",
    re.IGNORECASE,
)


def sanitize_sql(raw_sql: str, max_rows: int) -> str:
    """Validate an LLM SELECT and wrap it in the per-user secured CTE.

    The returned SQL is a single read-only statement whose only data sources
    are the ``my_*`` views, which are filtered to ``:current_user_id``. Any
    reference to a raw base table is rejected.

    Raises:
        UnsafeQueryError: if the SQL is unsafe or escapes the secured views.
    """

    sql = _strip_sql_markdown(raw_sql)

    if not sql:
        raise UnsafeQueryError("Empty SQL generated.")

    # Must be a single statement (no stacked queries).
    if ";" in sql:
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")

    # The model should return a bare SELECT (we add the WITH clause ourselves).
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise UnsafeQueryError("Only a single SELECT is permitted.")

    # No mutating/DDL keywords anywhere.
    if _FORBIDDEN.search(sql):
        raise UnsafeQueryError("Query contains forbidden keywords.")

    # Reject any attempt to read raw base tables — the model may only use the
    # secured my_* views. This is the enforcement that keeps a user inside
    # their own data even if the prompt is manipulated.
    protected = _PROTECTED_TABLE_REF.search(sql)
    if protected:
        raise UnsafeQueryError(
            f"Access to table '{protected.group(0)}' is not permitted."
        )

    # Enforce a row cap if the model didn't include one.
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        sql = f"{sql} LIMIT {max_rows}"

    # Prepend the secured, user-scoped CTE. The final query can only read
    # from the my_* views defined here.
    return f"{build_secured_cte()}\n{sql}"


# ============================================================
# 5. LLM
# ============================================================


def build_llm(settings: Settings) -> ChatVertexAI:
    logger.info("Initializing Vertex AI Gemini model '%s'", settings.llm_model)

    return ChatVertexAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )


SQL_SYSTEM_PROMPT = """
You are a careful SQL analyst for an Ethiopia tourism platform, answering on
behalf of a single logged-in traveller about THEIR OWN bookings only.

You are given the available VIEWS and a user question. Produce ONE read-only
SQL SELECT statement that answers the question.

STRICT RULES:
1. Output ONLY the SQL query. No explanation, no markdown, no comments.
2. Output a single bare SELECT statement. Do NOT write a WITH/CTE clause and
   do NOT end with a semicolon.
3. You may ONLY reference these views: my_requests, my_costs, my_passengers,
   my_visas. These are already scoped to the current traveller.
4. NEVER reference base tables such as users, package_requests,
   admin_notifications, or any table not listed as a view.
5. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
6. Prefer explicit JOINs on package_request_id = my_requests.id.
7. Add a reasonable LIMIT if the result could be large.
8. If the question cannot be answered from these views, output exactly:
   SELECT 'UNANSWERABLE' AS note
"""


def generate_sql(
    llm: ChatVertexAI,
    schema: str,
    question: str,
) -> str:
    prompt = f"""{SQL_SYSTEM_PROMPT}

========================
AVAILABLE VIEWS
========================
{schema}

========================
USER QUESTION
========================
{question}

========================
SQL QUERY
========================
"""

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise SqlGenerationError(f"LLM failed to generate SQL: {exc}") from exc

    return response.content


ANSWER_SYSTEM_PROMPT = """
You are a data assistant for an Ethiopia tourism platform.

Given the user's question, the SQL that was run, and the result rows (JSON),
write a clear, concise natural-language answer.

RULES:
1. Base the answer only on the provided rows. Do not invent data.
2. If the rows are empty, say no matching records were found.
3. For counts/aggregates, state the number plainly.
4. For lists, summarize; show a small table only if it aids clarity.
5. Do not mention SQL unless the user asked about it.
"""


def synthesize_answer(
    llm: ChatVertexAI,
    question: str,
    sql: str,
    rows: List[dict],
) -> str:
    prompt = f"""{ANSWER_SYSTEM_PROMPT}

USER QUESTION:
{question}

SQL EXECUTED:
{sql}

RESULT ROWS (JSON):
{json.dumps(rows, default=str, indent=2)}

ANSWER:
"""

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise SqlGenerationError(f"LLM failed to synthesize answer: {exc}") from exc

    return response.content


# ============================================================
# 6. QUERY EXECUTION
# ============================================================


def execute_query(db_engine: Engine, sql: str, user_id: int) -> List[dict]:
    """Run a validated, user-scoped SELECT and return rows as dicts.

    ``user_id`` is bound as the ``:current_user_id`` parameter used by the
    secured CTE, so it is never string-interpolated (injection-safe). The
    connection never commits, so even a slipped-through mutation cannot
    persist.
    """

    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(sql), {"current_user_id": user_id})
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            conn.rollback()  # ensure no writes are ever committed
    except Exception as exc:  # noqa: BLE001
        raise QueryExecutionError(f"Failed to execute query: {exc}") from exc

    return rows


# ============================================================
# 7. QUERY ASSISTANT SERVICE
# ============================================================


@dataclass
class QueryAssistant:
    """Answers natural-language questions over the tourism database."""

    settings: Settings
    llm: ChatVertexAI
    db_engine: Engine
    schema: str

    @classmethod
    def bootstrap(
        cls,
        settings: Optional[Settings] = None,
        db_engine: Optional[Engine] = None,
    ) -> "QueryAssistant":
        settings = settings or load_settings()
        db_engine = db_engine or default_engine

        # The LLM only ever sees the per-user secured views, never raw tables.
        schema = build_user_schema_description(db_engine)
        logger.info("Loaded user-scoped schema (%d views)", len(_SECURED_VIEWS))

        llm = build_llm(settings)

        return cls(
            settings=settings,
            llm=llm,
            db_engine=db_engine,
            schema=schema,
        )

    def ask(self, question: str, user_id: int) -> str:
        """Answer a question scoped to ``user_id``'s own data.

        ``user_id`` MUST come from a trusted/authenticated session (e.g. the
        app or supervisor), never from the end user's chat text. The bot will
        only ever return data owned by this user (their package requests and
        the costs, passengers, and visa applications derived from them).
        """

        if not isinstance(user_id, int) or user_id <= 0:
            raise ConfigurationError("A valid authenticated user_id is required.")

        logger.info("Question (user_id=%s): %s", user_id, question)

        raw_sql = generate_sql(self.llm, self.schema, question)
        sql = sanitize_sql(raw_sql, self.settings.max_rows)
        logger.info("Secured SQL: %s", sql)

        rows = execute_query(self.db_engine, sql, user_id)
        logger.info("Rows returned: %d", len(rows))

        return synthesize_answer(self.llm, question, sql, rows)


# ============================================================
# 8. CLI
# ============================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-user tourism database query assistant."
    )
    parser.add_argument(
        "user_id",
        type=int,
        help=(
            "Authenticated user id to scope the session to. In production this "
            "comes from the trusted auth layer, not from user input."
        ),
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings.log_level)

    try:
        assistant = QueryAssistant.bootstrap(settings)
    except QueryAssistantError as exc:
        logger.error("Failed to start: %s", exc)
        return 1

    print("\n===================================")
    print(" My Bookings Assistant")
    print("===================================")
    print(f"Session scoped to user_id={args.user_id}.")
    print("Ask about your package requests, travellers, costs, and visas.")

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
            answer = assistant.ask(question, user_id=args.user_id)
            print(f"\nAssistant: {answer}")
        except UnsafeQueryError as exc:
            print(f"\nAssistant: I can only look up your own booking data. ({exc})")
        except QueryAssistantError as exc:
            logger.error("Request failed: %s", exc)
            print("\nAssistant: Sorry, I couldn't answer that. Please rephrase.")


if __name__ == "__main__":
    raise SystemExit(main())
