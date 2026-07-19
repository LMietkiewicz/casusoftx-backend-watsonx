"""backend.py - Flask application for document-processing tasks and RAG queries.

Exposes two endpoints:

* ``POST /api/task``  - accepts a document upload plus a list of tasks, returns
  generated task ids immediately, and processes the document in the background,
  delivering each task result via the success/error callback URLs.
* ``POST /api/query`` - synchronous RAG query over previously-ingested documents.
* ``GET  /health``    - liveness probe.

Ingestion (``upload_to_milvus``), retrieval (``search_vectors``), conversion
(``convert_to_pdf``) and the per-task LLM operations live in their own modules;
this file is the HTTP layer, task routing, and background orchestration.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List

import fitz  # PyMuPDF
import hashlib
import requests
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, CrossEncoder
from waitress import serve

import config
from logging_utils import preview, setup_logging
from milvus import ensure_collection
from tasks import (
    base_extraction,
    category_subcategory,
    check_confidential,
    department_assignment,
    other,
    suggested_action,
    summary,
)
from utils import (
    call_llm, 
    convert_to_pdf, 
    search_vectors, 
    upload_to_milvus, 
    fetch_document_text,
)

# Configure logging before anything else so module-level events are formatted.
setup_logging()
logger = logging.getLogger(__name__)

# Two separate encoders (same model), by access pattern:
#   * encoder_ingest - used only by the single-threaded ingestion worker, so it
#     is never accessed concurrently and needs no lock.
#   * encoder_rag    - used by concurrent /api/query request threads; its forward
#     pass is serialized by a short lock inside search_vectors.
# Keeping them separate means a RAG query's embed never waits behind a document's
# (much larger) ingestion embed. Costs one extra model in memory by design.
logger.info("Loading encoders '%s' (ingest + rag)", config.ENCODER_MODEL)
encoder_ingest = SentenceTransformer(config.ENCODER_MODEL)
encoder_rag = SentenceTransformer(config.ENCODER_MODEL)

# Cross-encoder reranker, shared across query threads (serialized in search_vectors).
reranker = None
if config.RERANK_ENABLED:
    logger.info("Loading reranker '%s'", config.RERANKER_MODEL)
    reranker = CrossEncoder(config.RERANKER_MODEL)

# Accepted upload content types (zip handling has been removed).
CONTENT_TYPES = (
    "application/pdf",
    "application/doc",
    "application/msword",
    "application/docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/txt",
    "text/plain",
    "application/xml",
    "text/xml",
)

# Office/text formats that must be converted to PDF before processing.
_OFFICE_CONTENT_TYPES = frozenset(CONTENT_TYPES) - {"application/pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Single-worker ingestion: documents are processed strictly one after another
# (ingest + summary + tasks all run inline on this one worker). This serializes
# the encoder-heavy work, keeps Milvus inserts ordered, and makes the
# existence-check/insert in upload_to_milvus race-free (no concurrent ingest).
ingest_executor = ThreadPoolExecutor(max_workers=1)

# Admission control: bound the number of documents queued/in-flight so a burst
# cannot grow the queue (or the on-disk temp files) without limit. Over the cap,
# /api/task returns 503 so the caller applies backpressure.
_pending_lock = threading.Lock()
_pending_docs = 0


# --------------------------------------------------------------------------- #
# Task ids & callbacks
# --------------------------------------------------------------------------- #
def generate_task_id() -> str:
    """Return a collision-free task id."""
    return uuid.uuid4().hex


def send_callback(url: str, payload: Dict[str, Any]) -> None:
    """POST a result payload to a callback URL, with timeout and bounded retry.

    Args:
        url: Callback URL.
        payload: JSON-serializable payload.
    """
    last_exc: Exception | None = None
    for attempt in range(1, config.CALLBACK_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=config.CALLBACK_TIMEOUT)
            response.raise_for_status()
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Callback to %s failed (attempt %d/%d): %s",
                url, attempt, config.CALLBACK_MAX_RETRIES, exc,
            )
            if attempt < config.CALLBACK_MAX_RETRIES:
                time.sleep(config.CALLBACK_RETRY_BACKOFF * attempt)
    logger.error("Callback to %s failed after %d attempts: %s", url, config.CALLBACK_MAX_RETRIES, last_exc)


# --------------------------------------------------------------------------- #
# RAG query answering
# --------------------------------------------------------------------------- #
def run_rag_with_context(
    query: str,
    context_fragments: List[Dict[str, Any]],
    is_global_search: bool,
) -> str:
    """Build a context-grounded prompt and return the LLM's natural-language answer.

    Args:
        query: The user's query.
        context_fragments: Parent chunks retrieved from Milvus.
        is_global_search: True if the search spanned all documents.

    Returns:
        The model's answer, or a Polish error message on failure.
    """
    try:
        formatted_context = ""
        for i, fragment in enumerate(context_fragments):
            file_id = fragment.get("file_id", "N/A")
            contents = fragment.get("contents", "")
            if is_global_search:
                label = fragment.get("filename") or f"ID {file_id}"
                formatted_context += f"--- Fragment z dokumentu: {label} ---\n"
            else:
                formatted_context += f"--- Fragment {i + 1} ---\n"
            formatted_context += f"{contents}\n\n"

        if "qwen" in config.MODEL.lower():
            system_prompt = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                "Od teraz, będziesz komunikować się wyłącznie w języku polskim. \n"
            )
        else:
            system_prompt = ""

        system_prompt += (
            "Jesteś inteligentnym asystentem specjalizującym się w analizie dokumentów. "
            "Twoim zadaniem jest udzielanie precyzyjnych i pomocnych odpowiedzi na podstawie "
            "dostarczonych fragmentów.\n\n"
            "KONTEKST:\n"
            "Poniżej znajdują się fragmenty dokumentów, które posłużą jako podstawa Twojej odpowiedzi:\n"
        )
        system_prompt += formatted_context
        system_prompt += (
            "INSTRUKCJE:\n"
            "1. Odpowiadaj wyłącznie na podstawie informacji zawartych w powyższym "
            "kontekście. Nie korzystaj z własnej wiedzy ani z informacji spoza kontekstu.\n"
            "2. Jeżeli odpowiedź na pytanie nie znajduje się w kontekście, napisz wprost: "
            "\"Nie znalazłem tej informacji w dostarczonych dokumentach.\" "
            "Nie zgaduj i nie uzupełniaj brakujących danych.\n"
            "3. Odpowiadaj zwięźle, naturalnie i profesjonalnie.\n"
        )
        if is_global_search:
            system_prompt += (
                "4. Na końcu odpowiedzi wskaż nazwy dokumentów, z których pochodzą "
                "informacje, np. 'Informacje pochodzą z: umowa_najmu.pdf'.\n"
            )

        options = {"temperature": 0.2, "top_p": 0.5, "num_predict": 1024, "repeat_penalty": 1.1}
        return call_llm(prompt=query, system_message=system_prompt, options=options)

    except requests.exceptions.RequestException as exc:
        logger.error("RAG inference request failed: %s", exc)
        return "Przepraszam, wystąpił błąd podczas komunikacji z modelem językowym."
    except Exception as exc:
        logger.error("Unexpected error in run_rag_with_context: %s", exc)
        return "Przepraszam, wystąpił nieoczekiwany błąd."


# --------------------------------------------------------------------------- #
# Task dispatch
# --------------------------------------------------------------------------- #
def handle_task(
    document_id: int,
    task_type: str,
    task_id: str,
    params: Dict[str, Any],
    success_cb: str,
    error_cb: str,
    precomputed_summary: str,
) -> None:
    """Run a single task against the precomputed summary and fire its callback.

    Args:
        document_id: Document id.
        task_type: One of the supported task type strings.
        task_id: Unique id for this task.
        params: Per-task parameters keyed by task type.
        success_cb: Success callback URL.
        error_cb: Error callback URL.
        precomputed_summary: The document summary, computed once during ingestion.
    """
    try:
        if task_type == "SUMMARY":
            # summary() already returns clean, markdown-free prose.
            task_result = {"results": {"SUMMARY": precomputed_summary}}

        elif task_type == "CATEGORY_SUBCATEGORY":
            category_data = params.get("CATEGORY_SUBCATEGORY", {}).get("categories", [])
            formatted_categories = {
                cat["name"]: [s["name"] for s in cat.get("subcategories", [])]
                for cat in category_data
            }
            found_category, found_subcategory = category_subcategory(precomputed_summary, formatted_categories)
            results_dict: Dict[str, Any] = {"CATEGORY": found_category}
            if found_subcategory:
                results_dict["SUBCATEGORY"] = found_subcategory
            task_result = {"results": {"CATEGORY_SUBCATEGORY": results_dict}}

        elif task_type == "DEPARTMENT_ASSIGNMENT":
            departments_data = params.get("DEPARTMENT_ASSIGNMENT", {}).get("departments", [])
            task_result = {"results": {"DEPARTMENT_ASSIGNMENT": department_assignment(precomputed_summary, departments_data)}}

        elif task_type == "BASE_EXTRACTION":
            # Single structured call now returns the rendered key-value text.
            task_result = {"results": {"BASE_EXTRACTION": base_extraction(precomputed_summary)}}

        elif task_type == "CHECK_CONFIDENTIAL":
            task_result = {"results": {"CHECK_CONFIDENTIAL": check_confidential(precomputed_summary)}}

        elif task_type == "SUGGESTED_ACTION":
            task_result = {"results": {"SUGGESTED_ACTION": suggested_action(precomputed_summary)}}

        elif task_type == "OTHER":
            prompt = params.get("OTHER", {}).get("query", "")
            task_result = {"results": {"OTHER": other(prompt)}}

        else:
            logger.error("Unsupported task type: %s", task_type)
            raise ValueError(f"Unsupported task type: {task_type}")

        payload = {"documentId": document_id, "taskId": task_id, "taskType": task_type, **task_result}
        logger.info("Task %s (%s) for document %s completed", task_id, task_type, document_id)
        logger.debug("Task payload: %s", preview(json.dumps(payload, ensure_ascii=False)))
        send_callback(success_cb, payload)

    except Exception as exc:
        logger.error("Task %s (%s) for document %s failed: %s", task_id, task_type, document_id, exc)
        payload = {"documentId": document_id, "taskId": task_id, "taskType": task_type, "error": str(exc)}
        send_callback(error_cb, payload)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/health", methods=["GET"])
def health() -> Any:
    """Liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/task", methods=["POST"])
def create_task() -> Any:
    """Accept a document + task list, return task ids, and process it serially.

    Applies admission control (503 over the pending cap), then queues one
    ingestion job on the single-worker executor; that job ingests, summarizes,
    and runs every task inline before the next document begins.
    """
    global _pending_docs
    try:
        document_id = int(request.form["documentId"])
        content_type = request.form["documentContentType"]
        tasks = json.loads(request.form["tasks"])
        params = json.loads(request.form.get("params", "{}"))
        success_cb = request.form["successCallbackUrl"]
        error_cb = request.form["errorCallbackUrl"]

        if content_type not in CONTENT_TYPES:
            return jsonify({"error": f"Unsupported Media Type: {content_type}"}), 415

        # Admission control: refuse new work once the pending backlog is full, so
        # a burst applies backpressure to the caller instead of growing our queue
        # and temp-file footprint without bound.
        with _pending_lock:
            if _pending_docs >= config.MAX_PENDING_DOCS:
                logger.warning(
                    "Rejecting document %s: %d pending >= cap %d",
                    document_id, _pending_docs, config.MAX_PENDING_DOCS,
                )
                return (
                    jsonify({"error": "Server busy; too many pending documents",
                             "retryAfter": config.RETRY_AFTER_SECONDS}),
                    503,
                    {"Retry-After": str(config.RETRY_AFTER_SECONDS)},
                )
            _pending_docs += 1

        try:
            # Persist the upload to a temp file for processing.
            upload = request.files["documentFile"]
            filename = upload.filename or ""
            suffix = f'.{content_type.split("/")[-1]}'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            upload.save(temp_file.name)
            temp_file.close()
            temp_file_path = temp_file.name

            # SHA-256 of raw upload bytes (pre-conversion, stable) = content key.
            with open(temp_file_path, "rb") as fh:
                content_hash = hashlib.sha256(fh.read()).hexdigest()

            # Generate task ids up front and return them immediately.
            task_dicts = [{"taskType": task_type, "taskId": generate_task_id()} for task_type in tasks]

            def _run_tasks_inline(precomputed_summary: str) -> None:
                # Tasks run inline on the ingestion worker (each handle_task
                # catches its own errors and fires its own callback), so the
                # document is fully finished before the next one starts.
                for task_dict in task_dicts:
                    handle_task(
                        document_id, task_dict["taskType"], task_dict["taskId"],
                        params, success_cb, error_cb, precomputed_summary,
                    )

            def _ingest_and_run(open_doc: Callable[[], Any]) -> None:
                # Open once to ingest into Milvus, once more to summarize (the
                # first context manager closes the document), then run the tasks.
                with open_doc() as doc:
                    upload_to_milvus(doc, document_id, encoder_ingest, filename, content_hash)
                with open_doc() as doc:
                    precomputed_summary = summary(doc)
                _run_tasks_inline(precomputed_summary)

            def background_processing() -> None:
                global _pending_docs
                try:
                    if content_type == "application/pdf":
                        _ingest_and_run(lambda: fitz.open(temp_file_path))
                    elif content_type in _OFFICE_CONTENT_TYPES:
                        pdf_bytes = convert_to_pdf(temp_file_path, content_type)
                        _ingest_and_run(lambda: fitz.open(stream=pdf_bytes, filetype="pdf"))
                    else:
                        raise ValueError(f"Unsupported file type for processing: {content_type}")
                except Exception as exc:
                    logger.error("Background processing failed for document %s: %s", document_id, exc)
                    send_callback(error_cb, {"documentId": document_id, "taskType": "error", "error": str(exc)})
                finally:
                    logger.debug("Cleaning up temporary file: %s", temp_file_path)
                    try:
                        os.remove(temp_file_path)
                    except OSError as exc:
                        logger.warning("Failed to remove temp file %s: %s", temp_file_path, exc)
                    with _pending_lock:
                        _pending_docs -= 1

            ingest_executor.submit(background_processing)
            return jsonify(task_dicts), 200

        except Exception:
            # Setup/submission failed after admission — release the slot here,
            # since background_processing (which normally releases it) never ran.
            with _pending_lock:
                _pending_docs -= 1
            raise

    except KeyError as exc:
        return jsonify({"error": f"Missing required field: {exc}"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in 'tasks' or 'params' field."}), 400
    except Exception as exc:
        logger.error("create_task failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/query", methods=["POST"])
def sync_query() -> Any:
    """Synchronous RAG query over ingested documents."""
    try:
        data = request.get_json()
        document_id = data.get("documentId")
        document_id = int(document_id) if document_id is not None else None
        task_type = "OTHER"
        task_id = generate_task_id()
        query = data["query"]

        try:
            is_global = document_id is None
            used_full_document = False
            context_fragments: List[Dict[str, Any]] = []

            if document_id is not None:
                # Full-document fast path: feed the whole doc if it fits the budget.
                full_text = fetch_document_text(document_id)
                doc_budget_chars = config.RAG_CONTEXT_TOKEN_BUDGET * config.CHARS_PER_TOKEN
                if full_text and len(full_text) <= doc_budget_chars:
                    context_fragments = [{"file_id": document_id, "contents": full_text}]
                    used_full_document = True
                    logger.info(
                        "Query for document %s: full-document path (%d chars)",
                        document_id, len(full_text),
                    )

            if not used_full_document:
                context_fragments = search_vectors(query, encoder_rag, document_id, reranker=reranker)

            if not context_fragments:
                return jsonify({"error": "No context found"}), 404

            response = run_rag_with_context(query, context_fragments, is_global)

            payload = {
                "task_type": task_type,
                "task_id": task_id,
                "response": response,
                "sources": context_fragments,
            }
            return jsonify(payload), 200

        except Exception as exc:
            logger.error("sync_query processing failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = config.APP_PORT
    try:
        # Ensure the Milvus collection exists, sized to the live encoder.
        ensure_collection(encoder_ingest.get_sentence_embedding_dimension())
        logger.info("Starting backend server on host 0.0.0.0, port %s", port)
        serve(app, host="0.0.0.0", port=port)
    except Exception as exc:
        logger.critical("Server startup or execution failed: %s", exc)
        raise
    