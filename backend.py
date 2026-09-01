"""backend.py - Flask application for document-processing tasks and RAG queries.

Exposes:

* ``POST   /api/task``                  - accepts a document uuid plus a list of tasks,
  returns generated task ids immediately, then pulls the document from CSX
  storage and processes it in the background, delivering each task result via
  the success/error callback URLs.
* ``POST   /api/query``                 - synchronous RAG query over ingested documents.
* ``DELETE /ai/document/<document_id>`` - removes a document's data from Milvus.
* ``GET    /health``                    - liveness probe.

Retrieval from CSX storage (``fetch_document``), ingestion (``upload_to_milvus``),
search (``search_vectors``), conversion (``convert_to_pdf``) and the per-task LLM
operations live in their own modules; this file is the HTTP layer, task routing,
and background orchestration.
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

import io
import pdfplumber
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
    delete_document,
    document_exists,
    fetch_document,
    fetch_document_text,
    search_vectors,
    upload_to_milvus,
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
            contents = fragment.get("contents", "")
            if is_global_search:
                # Chunk rows are owned by content, not by a uuid, so filename is
                # the only per-fragment identifier available here.
                label = fragment.get("filename") or f"dokument {fragment.get('content_hash', '')[:8]}"
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
    file_uuid: str,
    task_type: str,
    task_id: str,
    params: Dict[str, Any],
    success_cb: str,
    error_cb: str,
    precomputed_summary: str,
) -> None:
    """Run a single task against the precomputed summary and fire its callback.

    Args:
        file_uuid: CSX document uuid; echoed back to the caller as documentId.
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

        payload = {"documentId": file_uuid, "taskId": task_id, "taskType": task_type, **task_result}
        logger.info("Task %s (%s) for document %s completed", task_id, task_type, file_uuid)
        logger.debug("Task payload: %s", preview(json.dumps(payload, ensure_ascii=False)))
        send_callback(success_cb, payload)

    except Exception as exc:
        logger.error("Task %s (%s) for document %s failed: %s", task_id, task_type, file_uuid, exc)
        payload = {"documentId": file_uuid, "taskId": task_id, "taskType": task_type, "error": str(exc)}
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
    """Accept a document uuid + task list, return task ids, and process it serially.

    The file is pulled from CSX storage by uuid. Object existence is validated
    synchronously so a bad uuid fails fast; the download itself runs on the
    ingestion worker so the caller is not held for it. Applies admission control
    (503 over the pending cap), then queues one ingestion job on the single-worker
    executor; that job ingests, summarizes, and runs every task inline before the
    next document begins.
    """
    global _pending_docs
    try:
        # Accept either JSON or form-encoded fields.
        data = request.get_json(silent=True) or request.form

        def _maybe_json(value: Any, default: Any) -> Any:
            if value is None:
                return default
            return json.loads(value) if isinstance(value, str) else value

        # Round-tripping through uuid.UUID validates the format and normalises
        # case; it is what makes interpolating this into a Milvus filter safe.
        file_uuid = str(uuid.UUID(str(data["fileUuid"])))
        tasks = _maybe_json(data["tasks"], [])
        params = _maybe_json(data.get("params"), {})
        success_cb = data["successCallbackUrl"]
        error_cb = data["errorCallbackUrl"]

        # Existence probe: metadata-only, and lets us 404 before taking a slot.
        try:
            if not document_exists(file_uuid):
                logger.warning("Unknown object %s", file_uuid)
                return jsonify({"error": f"Unknown file id: {file_uuid}"}), 404
        except Exception as exc:
            logger.error("CSX storage unavailable for %s: %s", file_uuid, exc)
            return jsonify({"error": "Object storage unavailable"}), 502

        # Admission control: refuse new work once the pending backlog is full, so
        # a burst applies backpressure to the caller instead of growing our queue
        # and temp-file footprint without bound.
        with _pending_lock:
            if _pending_docs >= config.MAX_PENDING_DOCS:
                logger.warning(
                    "Rejecting document %s: %d pending >= cap %d",
                    file_uuid, _pending_docs, config.MAX_PENDING_DOCS,
                )
                return (
                    jsonify({"error": "Server busy; too many pending documents",
                             "retryAfter": config.RETRY_AFTER_SECONDS}),
                    503,
                    {"Retry-After": str(config.RETRY_AFTER_SECONDS)},
                )
            _pending_docs += 1

        try:
            # Extension is irrelevant: routing is by Content-Type, and
            # convert_to_pdf substitutes a correct name for the converter.
            temp_file_path = os.path.join(tempfile.gettempdir(), f"{file_uuid}.{uuid.uuid4().hex[:8]}.bin")

            # Generate task ids up front and return them immediately.
            task_dicts = [{"taskType": task_type, "taskId": generate_task_id()} for task_type in tasks]

            def _run_tasks_inline(precomputed_summary: str) -> None:
                # Tasks run inline on the ingestion worker (each handle_task
                # catches its own errors and fires its own callback), so the
                # document is fully finished before the next one starts.
                for task_dict in task_dicts:
                    handle_task(
                        file_uuid, task_dict["taskType"], task_dict["taskId"],
                        params, success_cb, error_cb, precomputed_summary,
                    )

            def _ingest_and_run(open_doc: Callable[[], Any], content_hash: str, filename: str) -> None:
                # Open once to ingest into Milvus, once more to summarize (the
                # first context manager closes the document), then run the tasks.
                with open_doc() as doc:
                    upload_to_milvus(doc, file_uuid, encoder_ingest, filename, content_hash)
                with open_doc() as doc:
                    precomputed_summary = summary(doc)
                _run_tasks_inline(precomputed_summary)

            def background_processing() -> None:
                global _pending_docs
                try:
                    content_type, filename = fetch_document(file_uuid, temp_file_path)
                    content_type = content_type.split(";")[0].strip().lower()
                    if content_type not in CONTENT_TYPES:
                        raise ValueError(f"Unsupported Media Type: {content_type}")

                    # SHA-256 of the raw bytes (pre-conversion, stable) = content key.
                    digest = hashlib.sha256()
                    with open(temp_file_path, "rb") as fh:
                        for block in iter(lambda: fh.read(1 << 20), b""):
                            digest.update(block)
                    content_hash = digest.hexdigest()

                    if content_type == "application/pdf":
                        _ingest_and_run(lambda: pdfplumber.open(temp_file_path), content_hash, filename)
                    else:
                        pdf_bytes = convert_to_pdf(temp_file_path, content_type)
                        _ingest_and_run(
                            lambda: pdfplumber.open(io.BytesIO(pdf_bytes)),
                            content_hash, filename
                        )
                except Exception as exc:
                    logger.error("Background processing failed for document %s: %s", file_uuid, exc)
                    send_callback(error_cb, {"documentId": file_uuid, "taskType": "error", "error": str(exc)})
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
    except ValueError as exc:
        return jsonify({"error": f"Invalid field value: {exc}"}), 400
    except Exception as exc:
        logger.error("create_task failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/query", methods=["POST"])
def sync_query() -> Any:
    """Synchronous RAG query over ingested documents."""
    try:
        data = request.get_json()
        raw_uuid = data.get("fileUuid")
        file_uuid = str(uuid.UUID(str(raw_uuid))) if raw_uuid else None
        task_type = "OTHER"
        task_id = generate_task_id()
        query = data["query"]

        try:
            is_global = file_uuid is None
            used_full_document = False
            context_fragments: List[Dict[str, Any]] = []

            if file_uuid is not None:
                # Full-document fast path: feed the whole doc if it fits the budget.
                full_text = fetch_document_text(file_uuid)
                doc_budget_chars = config.RAG_CONTEXT_TOKEN_BUDGET * config.CHARS_PER_TOKEN
                if full_text and len(full_text) <= doc_budget_chars:
                    context_fragments = [{"file_uuid": file_uuid, "contents": full_text}]
                    used_full_document = True
                    logger.info(
                        "Query for document %s: full-document path (%d chars)",
                        file_uuid, len(full_text),
                    )

            if not used_full_document:
                context_fragments = search_vectors(query, encoder_rag, file_uuid, reranker=reranker)

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
    

@app.route("/ai/document/<document_id>", methods=["DELETE"])
def delete_document_route(document_id: str) -> Any:
    """Remove a document's data from Milvus.

    Idempotent: deleting a uuid that was never ingested is a success, not an
    error. Chunks are removed only when no other uuid still names that content.
    """
    try:
        file_uuid = str(uuid.UUID(document_id))
    except ValueError:
        return jsonify({"error": f"Invalid document id: {document_id}"}), 400

    try:
        result = delete_document(file_uuid)
    except Exception as exc:
        logger.error("Delete failed for document %s: %s", file_uuid, exc)
        return jsonify({"error": str(exc)}), 500

    if not result["deleted"]:
        return jsonify({
            "documentId": file_uuid,
            "deleted": False,
            "message": "No such document in the database",
        }), 200

    return jsonify({
        "documentId": file_uuid,
        "deleted": True,
        "kind": result["kind"],
    }), 200


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
