"""utils.py - LLM inference, Milvus hybrid search, and document-to-PDF conversion.

These are the integration points to external systems: the inference backend
(Ollama or IBM watsonx), the Milvus vector store, and the Gotenberg conversion
service. Heavy/optional dependencies (watsonx SDK) are imported lazily so a
deployment only needs the libraries for the provider it actually uses.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import requests
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

import config
from logging_utils import preview
from milvus import get_milvus_client, reset_client
from processing import processing_pipeline

if TYPE_CHECKING:  # type-only; these objects are passed in, never imported at runtime
    import fitz
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Serializes forward passes through the shared RAG query encoder, which several
# request threads may hit at once. Document-ingestion embedding uses a separate
# encoder and is single-threaded, so it is not guarded here.
_rag_encode_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# LLM inference
# --------------------------------------------------------------------------- #
# Sampling defaults; caller-supplied options are merged over these, so a partial
# options dict can never raise KeyError in the watsonx parameter mapping.
_DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.5,
    "top_p": 0.5,
    "num_predict": 1024,
    "repeat_penalty": 1.1,
}

# Persistent watsonx client, built once on first use to avoid re-authenticating
# on every call.
_watsonx_client: Optional[Any] = None


def _get_watsonx_client() -> Any:
    """Return the shared watsonx ModelInference, constructing it once.

    Returns:
        The cached ``ModelInference`` instance.
    """
    global _watsonx_client
    if _watsonx_client is None:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        logger.info("Initializing watsonx model '%s'", config.MODEL)
        _watsonx_client = ModelInference(
            model_id=config.MODEL,
            credentials=Credentials(
                api_key=config.API_KEY,
                url=config.INFERENCE_PROVIDER_BASE_URL,
            ),
            project_id=config.PROJECT_ID,
        )
    return _watsonx_client


def _call_ollama(
    prompt: str,
    system_message: str,
    options: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the Ollama /api/generate endpoint.

    Args:
        prompt: User prompt.
        system_message: System instruction.
        options: Merged sampling options (passed through to Ollama).
        schema: Optional JSON schema; when given, Ollama constrains the output
            to match it (grammar-constrained decoding, guaranteed-valid JSON).

    Returns:
        The generated text.
    """
    payload = {
        "model": config.MODEL,
        "prompt": prompt,
        "system": system_message,
        "options": options,
        "stream": False,
    }
    if schema is not None:
        payload["format"] = schema  # native structured output
    response = requests.post(
        f"{config.INFERENCE_PROVIDER_BASE_URL}/api/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=config.LLM_REQUEST_TIMEOUT,
        verify=config.INFERENCE_PROVIDER_SSL_VERIFY,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _call_watsonx(
    prompt: str,
    system_message: str,
    options: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Call IBM watsonx chat via the persistent client.

    Args:
        prompt: User prompt.
        system_message: System instruction.
        options: Merged sampling options.
        schema: Optional JSON schema for structured output.

    Returns:
        The generated text.

    Note:
        Structured output here is "JSON envelope + validators", not hard schema
        enforcement. JSON_OBJECT (set when a schema is given) guarantees valid,
        parseable JSON; the prompt still describes the schema (JSON_OBJECT
        requires this), and the structured_output resolvers + repair cascade
        enforce the actual schema. For decode-time enforcement, switch to
        function-calling (tools + forced tool_choice).
    """
    from ibm_watsonx_ai.foundation_models.schema import (
        TextChatParameters,
        TextChatResponseFormat,
        TextChatResponseFormatType,
    )

    if schema is not None:
        system_message = (
            f"{system_message}\n\nReturn ONLY a JSON object that conforms to this "
            f"JSON schema, with no other text:\n{json.dumps(schema, ensure_ascii=False)}"
        )

    client = _get_watsonx_client()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    param_kwargs: Dict[str, Any] = {
        "temperature": options["temperature"],
        "max_tokens": options["num_predict"],
        "top_p": options["top_p"],
        "repetition_penalty": options["repeat_penalty"],
    }
    if schema is not None:
        # JSON_OBJECT guarantees a valid JSON envelope (no fences/prose); it does
        # NOT enforce the schema — that stays the job of the structured_output
        # resolvers/repair cascade downstream.
        param_kwargs["response_format"] = TextChatResponseFormat(
            TextChatResponseFormatType.JSON_OBJECT
        )
    params = TextChatParameters(**param_kwargs)
    response = client.chat(messages=messages, params=params)
    return response["choices"][0]["message"]["content"].strip()


def call_llm(
    prompt: str,
    system_message: str = "",
    options: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a completion from the configured provider (Ollama or watsonx).

    Args:
        prompt: The user prompt / input text.
        system_message: System instruction for context.
        options: Sampling options; merged over module defaults.
        schema: Optional JSON schema for structured output. On Ollama this is
            enforced (guaranteed-valid JSON); on watsonx it is currently a prompt
            directive (interim) backed by the structured_output validators.

    Returns:
        The generated text, stripped.

    Raises:
        requests.exceptions.RequestException: On Ollama HTTP failure.
        Exception: On other inference failures (propagated, not swallowed).
    """
    merged_options = {**_DEFAULT_OPTIONS, **(options or {})}

    logger.debug(
        "LLM call (provider=%s, structured=%s)\n  system: %s\n  prompt: %s",
        config.PROVIDER, schema is not None, preview(system_message), preview(prompt),
    )
    try:
        if config.PROVIDER == "ollama":
            result = _call_ollama(prompt, system_message, merged_options, schema)
        else:
            result = _call_watsonx(prompt, system_message, merged_options, schema)
    except requests.exceptions.RequestException as exc:
        logger.error("LLM request failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error in call_llm: %s", exc)
        raise

    logger.debug("LLM response: %s", preview(result))
    return result


def strip_markdown(text: str) -> str:
    """Remove Markdown syntax that the frontend would render as literal symbols.

    Targets emphasis (``**``/``*``/``__``/``_``), inline/code-fence backticks, and
    leading heading/blockquote markers. Leaves list hyphens and ordinary text
    intact (the italic rule uses word boundaries so it won't touch in-word
    underscores). Deterministic — this replaces the old LLM formatter loop.

    Args:
        text: Text possibly containing Markdown.

    Returns:
        Plain text with Markdown syntax removed.
    """
    if not text:
        return ""
    # Code fences and inline code: keep the content, drop the backticks.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", text)
    text = text.replace("`", "")
    # Emphasis: **bold**, __bold__, *italic*, _italic_ -> inner text.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text, flags=re.DOTALL)
    # Leading heading (#) and blockquote (>) markers, line by line.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    # Any stray emphasis characters left over.
    text = text.replace("**", "")
    return text.strip()


# --------------------------------------------------------------------------- #
# Milvus hybrid search
# --------------------------------------------------------------------------- #
def _milvus_read(operation: Callable[[MilvusClient], Any]) -> Any:
    """Run a read-only Milvus operation, reconnecting once on failure.

    Reads are idempotent, so a single retry after rebuilding the client safely
    absorbs the transient connection drops that previously motivated rebuilding
    the client on every call.

    Args:
        operation: Callable receiving the client and returning a result.

    Returns:
        The operation's result.
    """
    try:
        return operation(get_milvus_client())
    except Exception as exc:
        logger.warning("Milvus read failed (%s); reconnecting and retrying once.", exc)
        reset_client()
        return operation(get_milvus_client())


def search_vectors(
    query_text: str,
    encoder: "SentenceTransformer",
    document_id: Optional[int] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Hybrid (dense + BM25) search over child chunks, returning parent context.

    Matches children with reciprocal-rank fusion, then fetches their parent
    chunks by globally-unique ``parent_id`` (so no cross-document contamination).

    Args:
        query_text: The natural-language query.
        encoder: Sentence-transformer used for the dense query vector.
        document_id: If given, restrict the search to one document.
        top_k: Number of fused child hits to keep.

    Returns:
        Parent chunk dicts (``parent_id``, ``contents``, ``file_id``, ``type``),
        or an empty list if nothing matched.
    """
    document_id = int(document_id) if document_id is not None else None
    # The RAG encoder is shared across concurrent query threads, so serialize the
    # forward pass. Held only for one short query embed (microseconds), never
    # behind document-ingestion embedding (that uses a separate, ingest-owned
    # encoder), so RAG latency stays low under ingestion load.
    with _rag_encode_lock:
        query_embedding = encoder.encode(query_text).tolist()

    child_filter = "hierarchy == 'child'"
    if document_id is not None:
        child_filter += f" and file_id == {document_id}"

    def _search(client: MilvusClient) -> List[Dict[str, Any]]:
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP"},
            limit=top_k * 2,  # over-fetch candidates for better fusion
            expr=child_filter,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],  # Milvus builds the BM25 sparse vector from text
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=top_k * 2,
            expr=child_filter,
        )
        child_hits = client.hybrid_search(
            collection_name=config.MILVUS_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["parent_id"],
            consistency_level="Strong",
        )

        # Collect the (globally unique) parent ids of the matched children.
        parent_ids = {
            hit.get("entity", {}).get("parent_id")
            for hits in child_hits
            for hit in hits
            if hit.get("entity", {}).get("parent_id") is not None
        }
        if not parent_ids:
            return []

        # Flat fetch: parent_id is globally unique, so no file_id scoping needed.
        parent_filter = f"parent_id in {list(parent_ids)} and hierarchy == 'parent'"
        return client.query(
            collection_name=config.MILVUS_COLLECTION,
            filter=parent_filter,
            output_fields=["parent_id", "contents", "file_id", "type"],
            consistency_level="Strong",
        )

    results = _milvus_read(_search)
    logger.debug(
        "search %s (doc=%s) -> %d parent fragment(s)",
        preview(query_text, 120), document_id, len(results),
    )
    return results


def upload_to_milvus(doc: "fitz.Document", file_id: int, model: "SentenceTransformer") -> None:
    """Process a document and insert its chunks into Milvus, if not already present.

    Args:
        doc: An open PyMuPDF document.
        file_id: Unique document id.
        model: Sentence-transformer encoder.
    """
    file_id = int(file_id)
    try:
        client = get_milvus_client()

        if not client.has_collection(config.MILVUS_COLLECTION):
            logger.error(
                "Collection '%s' does not exist; cannot upload file %s",
                config.MILVUS_COLLECTION, file_id,
            )
            return

        existing = client.query(
            collection_name=config.MILVUS_COLLECTION,
            filter=f"file_id == {file_id}",
            output_fields=["file_id"],
            limit=1,
        )
        if existing:
            logger.info("File %s already present in collection; skipping upload", file_id)
            return

        logger.info("Processing file %s for upload", file_id)
        rows = processing_pipeline(doc, file_id, model)
        if not rows:
            logger.warning("Processing produced no rows for file %s; nothing to upload", file_id)
            return

        result = client.insert(collection_name=config.MILVUS_COLLECTION, data=rows)
        logger.info("Inserted %s record(s) for file %s", result["insert_count"], file_id)

    except Exception:
        # Clear a possibly-stale connection so later operations reconnect, but do
        # NOT retry: insert is not idempotent and a blind retry could duplicate rows.
        reset_client()
        raise


# --------------------------------------------------------------------------- #
# Document -> PDF conversion (Gotenberg sidecar)
# --------------------------------------------------------------------------- #
# Map of accepted content types to the file extension LibreOffice needs to pick
# the right import filter. PDF is handled by passthrough, not conversion.
_CONTENT_TYPE_EXTENSION: Dict[str, str] = {
    "application/doc": ".doc",
    "application/msword": ".doc",
    "application/docx": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/txt": ".txt",
    "text/plain": ".txt",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


def convert_to_pdf(file_path: str, content_type: str) -> bytes:
    """Convert a document to PDF, returning the PDF bytes.

    PDFs are returned unchanged. Office and text formats are converted by the
    Gotenberg LibreOffice route, which preserves layout and tables and paginates
    correctly (unlike manual text-to-PDF rendering).

    Args:
        file_path: Path to the input file.
        content_type: MIME type of the input file.

    Returns:
        The PDF as bytes.

    Raises:
        ValueError: If the content type is not supported.
        RuntimeError: If the Gotenberg conversion fails.
    """
    if content_type == "application/pdf":
        with open(file_path, "rb") as handle:
            return handle.read()

    extension = _CONTENT_TYPE_EXTENSION.get(content_type)
    if extension is None:
        raise ValueError(f"Unsupported content type for conversion: {content_type}")

    # Give LibreOffice a filename with a recognizable extension regardless of how
    # the temp file on disk happens to be named.
    base = os.path.basename(file_path)
    upload_name = base if base.lower().endswith(extension) else f"document{extension}"

    url = f"{config.GOTENBERG_URL}/forms/libreoffice/convert"
    logger.debug("Converting '%s' (%s) to PDF via Gotenberg", upload_name, content_type)

    try:
        with open(file_path, "rb") as handle:
            response = requests.post(
                url,
                files={"files": (upload_name, handle, content_type)},
                timeout=config.GOTENBERG_TIMEOUT,
            )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gotenberg conversion failed for {upload_name}: {exc}") from exc

    logger.debug("Gotenberg returned %d bytes of PDF", len(response.content))
    return response.content