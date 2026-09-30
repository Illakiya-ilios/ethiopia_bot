"""Ethiopia Tourism AI Assistant — GCP-native, enterprise-grade RAG.

A single-file Retrieval-Augmented Generation application backed entirely by
Google Cloud Platform managed services:

    - Embeddings : Gemini API Text Embeddings  (models/text-embedding-004)
    - Generation : Gemini API                  (gemini-2.0-flash-lite)
    - Retrieval  : Hybrid (BM25 + dense vector) over a persisted vector store

Enterprise concerns addressed here:
    - Fail-fast, validated configuration sourced from the environment
    - Structured logging (JSON-capable) instead of bare prints
    - Typed exception hierarchy for clean error handling
    - Lazy, dependency-injected initialization (no import-time side effects)
    - Clear, testable functions and a thin CLI entrypoint

Authentication uses Application Default Credentials (ADC). In production run on
a GCP identity (service account attached to Cloud Run / GKE / GCE). Locally use:

    gcloud auth application-default login
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from pydantic import Field


# ============================================================
# 0. EXCEPTIONS
# ============================================================


class RagError(Exception):
    """Base class for all application errors."""


class ConfigurationError(RagError):
    """Raised when configuration is missing or invalid."""


class IngestionError(RagError):
    """Raised when document loading, parsing, or indexing fails."""


class RetrievalError(RagError):
    """Raised when context retrieval fails."""


class GenerationError(RagError):
    """Raised when the LLM fails to generate an answer."""


# ============================================================
# 1. LOGGING
# ============================================================


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
) -> None:
    """Configure root logging once, in a container-friendly way."""

    handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("urllib3", "google.auth", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("ethiopia_tourism_rag")


# ============================================================
# 2. CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class Settings:
    """Validated, environment-driven application settings.

    Values are loaded from environment variables (and an optional ``.env``
    file). Validation happens once at startup so the app never boots into a
    half-configured state.
    """

    # Gemini API (Google AI Studio key)
    google_api_key: str

    # Embeddings (Gemini API). The Google AI API exposes embeddings under
    # 'models/gemini-embedding-001' (v1beta); 'text-embedding-*' are Vertex-only.
    embedding_model: str = "models/gemini-embedding-001"

    # Generation (Gemini API)
    llm_model: str = "gemini-2.0-flash-lite"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048

    # Vector store (persisted Chroma; embeddings served by Vertex AI)
    chroma_dir: str = "chroma_db"
    collection_name: str = "ethiopia_tourism"

    # Ingestion
    #   - pdf_path : a single PDF (backward compatible)
    #   - data_dirs: one or more folders scanned recursively for PDFs
    # If data_dirs is set it takes precedence; otherwise pdf_path is used.
    pdf_path: str = "Ethiopia_Tourism_RAG_Expanded_Guide.pdf"
    data_dirs: tuple[str, ...] = ()
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    retrieval_top_k: int = 8
    vector_fetch_k: int = 20
    mmr_lambda: float = 0.7

    # Observability
    #   Default WARNING keeps the console quiet (only warnings/errors show).
    #   Set LOG_LEVEL=INFO in the environment for verbose diagnostics.
    log_level: str = "WARNING"
    json_logs: bool = False

    def validate(self) -> None:
        """Fail fast on invalid configuration."""

        if not self.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required.")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        if not (0.0 <= self.mmr_lambda <= 1.0):
            raise ConfigurationError("MMR_LAMBDA must be between 0 and 1.")

        # Validate the configured ingestion sources.
        if self.data_dirs:
            missing = [d for d in self.data_dirs if not os.path.isdir(d)]
            if missing:
                raise ConfigurationError(
                    f"DATA_DIRS folder(s) not found: {', '.join(missing)}"
                )
        elif not os.path.exists(self.pdf_path):
            raise ConfigurationError(f"PDF not found: {self.pdf_path}")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into a tuple of trimmed values."""

    raw = os.getenv(name)

    if not raw:
        return ()

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_settings() -> Settings:
    """Load and validate settings from the environment."""

    load_dotenv()

    settings = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048")),
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "ethiopia_tourism"),
        pdf_path=os.getenv("PDF_PATH", "Ethiopia_Tourism_RAG_Expanded_Guide.pdf"),
        data_dirs=_get_list("DATA_DIRS"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
        vector_fetch_k=int(os.getenv("VECTOR_FETCH_K", "20")),
        mmr_lambda=float(os.getenv("MMR_LAMBDA", "0.7")),
        log_level=os.getenv("LOG_LEVEL", "WARNING"),
        json_logs=_get_bool("JSON_LOGS", False),
    )

    settings.validate()

    return settings


# ============================================================
# 3. EMBEDDINGS (Vertex AI)
# ============================================================


def build_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini API embeddings client (uses GOOGLE_API_KEY)."""

    logger.info(
        "Initializing Gemini embeddings model '%s'",
        settings.embedding_model,
    )

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


# ============================================================
# 4. DISCOVER + LOAD PDFs
# ============================================================


def discover_pdfs(settings: Settings) -> List[str]:
    """Resolve the list of PDF files to ingest.

    If ``data_dirs`` is configured, every ``*.pdf`` under those folders is
    collected recursively. Otherwise the single ``pdf_path`` is used.
    """

    if settings.data_dirs:
        pdfs: List[str] = []

        for directory in settings.data_dirs:
            found = sorted(str(p) for p in Path(directory).rglob("*.pdf"))
            logger.info("Found %d PDF(s) in '%s'", len(found), directory)
            pdfs.extend(found)

        if not pdfs:
            raise IngestionError(
                f"No PDF files found in: {', '.join(settings.data_dirs)}"
            )

        return pdfs

    return [settings.pdf_path]


def load_pdf(settings: Settings) -> List[Document]:
    """Load all configured PDFs into LangChain documents (one per page).

    Every page is tagged with its originating ``source`` file so retrieved
    context can be attributed back to the correct document.
    """

    pdf_files = discover_pdfs(settings)

    logger.info("Loading %d PDF file(s)", len(pdf_files))

    documents: List[Document] = []

    for pdf_file in pdf_files:
        logger.info("Loading PDF: %s", pdf_file)

        try:
            loader = PyPDFLoader(pdf_file)
            file_docs = loader.load()
        except Exception as exc:  # noqa: BLE001 - convert to domain error
            raise IngestionError(
                f"Failed to load PDF '{pdf_file}': {exc}"
            ) from exc

        for doc in file_docs:
            doc.metadata["source"] = pdf_file

        documents.extend(file_docs)

    logger.info(
        "Loaded %d pages across %d file(s)", len(documents), len(pdf_files)
    )

    return documents


# ============================================================
# 5. SPLIT DOCUMENT
# ============================================================


def split_documents(
    settings: Settings,
    documents: List[Document],
) -> List[Document]:
    """Split page-level documents into overlapping retrieval chunks."""

    logger.info("Splitting %d documents into chunks", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata.setdefault("source", settings.pdf_path)

    logger.info("Created %d chunks", len(chunks))

    return chunks


# ============================================================
# 6. CREATE VECTOR DATABASE
# ============================================================


def create_vector_store(
    settings: Settings,
    embeddings: GoogleGenerativeAIEmbeddings,
    chunks: List[Document],
) -> Chroma:
    """Build and persist the vector store from freshly embedded chunks."""

    logger.info("Creating vector store at '%s'", settings.chroma_dir)

    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_dir,
        )
    except Exception as exc:  # noqa: BLE001
        # Ingestion is atomic: if embedding/persisting fails partway through
        # (e.g. expired credentials), remove the partial store so the next run
        # re-ingests cleanly instead of loading an empty/corrupt database.
        _remove_store(settings.chroma_dir)
        raise IngestionError(f"Failed to create vector store: {exc}") from exc

    logger.info("Vector store created successfully")

    return db


def _remove_store(chroma_dir: str) -> None:
    """Best-effort deletion of a (possibly partial) vector store directory."""

    if not os.path.isdir(chroma_dir):
        return

    logger.warning("Removing incomplete vector store at '%s'", chroma_dir)

    try:
        shutil.rmtree(chroma_dir)
    except OSError as exc:
        logger.error("Could not remove '%s': %s", chroma_dir, exc)


# ============================================================
# 7. LOAD EXISTING VECTOR DATABASE
# ============================================================


def load_vector_store(
    settings: Settings,
    embeddings: GoogleGenerativeAIEmbeddings,
) -> Chroma:
    """Open an already-persisted vector store."""

    logger.info("Loading existing vector store at '%s'", settings.chroma_dir)

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


# ============================================================
# 8. BUILD / LOAD VECTOR DATABASE
# ============================================================


def get_vector_database(
    settings: Settings,
    embeddings: GoogleGenerativeAIEmbeddings,
) -> Chroma:
    """Return a ready-to-use vector store, building it on first run."""

    if os.path.exists(settings.chroma_dir):
        db = load_vector_store(settings, embeddings)

        # A store left behind by a failed ingest can exist but be empty.
        # Detect that, discard it, and rebuild instead of failing later.
        if _store_is_populated(db):
            logger.info("Vector store already exists; reusing it")
            return db

        logger.warning(
            "Existing vector store is empty; rebuilding from source documents"
        )
        _remove_store(settings.chroma_dir)

    documents = load_pdf(settings)
    chunks = split_documents(settings, documents)

    return create_vector_store(settings, embeddings, chunks)


def _store_is_populated(db: Chroma) -> bool:
    """Return True if the vector store contains at least one document."""

    try:
        return db._collection.count() > 0
    except Exception:  # noqa: BLE001 - treat any probe failure as empty
        return False


# ============================================================
# 9. LOAD DOCUMENTS FROM VECTOR STORE
# ============================================================


def get_all_documents(db: Chroma) -> List[Document]:
    """Materialize all stored documents (needed to seed the BM25 index)."""

    data = db.get(include=["documents", "metadatas"])

    documents: List[Document] = []

    for content, metadata in zip(data["documents"], data["metadatas"]):
        documents.append(
            Document(page_content=content, metadata=metadata or {})
        )

    return documents


# ============================================================
# 10. HYBRID RETRIEVER
# ============================================================


class HybridRetriever(BaseRetriever):
    """Combines BM25 (lexical) and dense vector (semantic) retrieval.

    Results are merged and de-duplicated by (source, page, chunk_id) so the
    same passage is never returned twice.
    """

    bm25_retriever: BaseRetriever = Field(description="BM25 retriever")
    vector_retriever: BaseRetriever = Field(description="Vector retriever")
    k: int = Field(default=8)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:

        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        combined = bm25_docs + vector_docs

        unique_documents: List[Document] = []
        seen = set()

        for doc in combined:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.metadata.get("chunk_id"),
            )

            if key not in seen:
                seen.add(key)
                unique_documents.append(doc)

            if len(unique_documents) >= self.k:
                break

        return unique_documents


def build_retriever(
    settings: Settings,
    db: Chroma,
    all_documents: List[Document],
) -> HybridRetriever:
    """Assemble the hybrid retriever from BM25 + vector retrievers."""

    if not all_documents:
        raise RetrievalError(
            "No documents available to build the retriever."
        )

    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = settings.retrieval_top_k

    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_top_k,
            "fetch_k": settings.vector_fetch_k,
            "lambda_mult": settings.mmr_lambda,
        },
    )

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        k=settings.retrieval_top_k,
    )


# ============================================================
# 11. FORMAT CONTEXT
# ============================================================


def format_context(documents: List[Document]) -> str:
    """Render retrieved documents into a source-attributed context block."""

    context_parts = []

    for doc in documents:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"SOURCE: {source}\nPAGE: {page}\n\n{doc.page_content}"
        )

    return "\n\n--------------------\n\n".join(context_parts)


# ============================================================
# 12. GEMINI LLM (Vertex AI)
# ============================================================


def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the Gemini API chat model (uses GOOGLE_API_KEY)."""

    logger.info("Initializing Gemini model '%s'", settings.llm_model)

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        google_api_key=settings.google_api_key,
    )


# ============================================================
# 13. RAG PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are the Ethiopia Tourism AI Assistant.

Your job is to answer questions about Ethiopia tourism
using the provided knowledge base.

RULES:

1. Use the retrieved context as your primary source.

2. Do not invent tourism information.

3. If the answer is not supported by the retrieved
   context, say:

   "I don't have enough information in the current
   Ethiopia tourism knowledge base."

4. Do not fabricate:
   - prices
   - hotel availability
   - flight availability
   - visa approval
   - visa processing time
   - current security conditions
   - current health requirements

5. If the user asks for dynamic information that is
   not contained in the knowledge base, clearly say
   that current information needs to be checked using
   a live source.

6. Answer naturally and clearly.

7. For simple questions, keep the answer concise.

8. For itinerary questions, provide useful structured
   recommendations based only on the retrieved context.

9. Do not mention the internal RAG system unless the
   user asks about it.

10. Do not claim that information is current unless
    the context explicitly supports that.
"""


def generate_answer(
    llm: ChatGoogleGenerativeAI,
    question: str,
    context: str,
) -> str:
    """Invoke Gemini with the system prompt, retrieved context, and question."""

    prompt = f"""
{SYSTEM_PROMPT}

========================
RETRIEVED KNOWLEDGE
========================

{context}

========================
USER QUESTION
========================

{question}

========================
ANSWER
========================
"""

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"LLM generation failed: {exc}") from exc

    return response.content


# ============================================================
# 14. RAG SERVICE
# ============================================================


@dataclass
class RagService:
    """Orchestrates retrieval + generation.

    Built once via :meth:`bootstrap` and reused across requests, which keeps
    heavy clients (embeddings, LLM, vector store) warm.
    """

    settings: Settings
    llm: ChatGoogleGenerativeAI
    retriever: HybridRetriever

    @classmethod
    def bootstrap(cls, settings: Optional[Settings] = None) -> "RagService":
        """Wire up all dependencies for the service."""

        settings = settings or load_settings()

        embeddings = build_embeddings(settings)
        db = get_vector_database(settings, embeddings)

        all_documents = get_all_documents(db)
        logger.info(
            "Documents available for retrieval: %d", len(all_documents)
        )

        retriever = build_retriever(settings, db, all_documents)
        llm = build_llm(settings)

        return cls(settings=settings, llm=llm, retriever=retriever)

    def ask(self, question: str) -> str:
        """Answer a single question through the full RAG pipeline."""

        logger.info("Handling question: %s", question)

        try:
            documents = self.retriever.invoke(question)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        if not documents:
            return (
                "I don't have enough information in the current "
                "Ethiopia tourism knowledge base."
            )

        context = format_context(documents)

        return generate_answer(self.llm, question, context)


# ============================================================
# 15. CLI
# ============================================================


def main() -> int:
    """Interactive CLI entrypoint."""

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        # Logging may not be configured yet; print the config error plainly.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    try:
        service = RagService.bootstrap(settings)
    except RagError as exc:
        logger.error("Failed to start service: %s", exc)
        return 1

    print("\n===================================")
    print(" Ethiopia Tourism AI Assistant")
    print("===================================")

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
            answer = service.ask(question)
            print(f"\nAssistant: {answer}")
        except RagError as exc:
            logger.error("Request failed: %s", exc)
            print(
                "\nAssistant: Sorry, something went wrong handling that "
                "request. Please try again."
            )


if __name__ == "__main__":
    raise SystemExit(main())
