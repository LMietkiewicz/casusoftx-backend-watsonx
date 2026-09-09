"""utils.py - LLM inference, Postgres hybrid search, and document-to-PDF conversion.

These are the integration points to external systems: the inference backend
(Ollama or IBM watsonx), the Postgres/pgvector store, and the unoserver sidecar
"""
from __future__ import annotations

import json
import logging
import re
import socket
import threading

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import requests
import psycopg

import config
from logging_utils import preview
from pgstore import get_pool, reset_pool
from processing import processing_pipeline
from s3gateway_client import S3GatewayError, get_csx_storage_client

if TYPE_CHECKING:  # type-only; these objects are passed in, never imported at runtime
    from pdf_io import PdfBundle
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Serializes forward passes through the shared RAG query encoder and reranker, which 
# several request threads may hit at once. Document-ingestion embedding uses a separate
# encoder and is single-threaded, so it is not guarded here.
_rag_encode_lock = threading.Lock()

# Serializes the shared cross-encoder reranker across concurrent query threads.
_rerank_lock = threading.Lock()

# Each leg ranks CHILD chunks, then collapses to parents with MIN(rn)
_HYBRID_SEARCH_SQL = """
WITH dense AS (
    SELECT parent_id, MIN(rn) AS rank FROM (
        SELECT parent_id,
               ROW_NUMBER() OVER (ORDER BY dense_embedding <#> %(qvec)s) AS rn
        FROM {table}
        WHERE hierarchy = 'child' AND is_search
          AND (%(scope)s::text IS NULL OR content_hash = %(scope)s::text)
        ORDER BY dense_embedding <#> %(qvec)s
        LIMIT %(wide)s
    ) d GROUP BY parent_id
),
sparse AS (
    SELECT parent_id, MIN(rn) AS rank FROM (
        SELECT parent_id,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(contents_tsv,
                                       plainto_tsquery('simple', %(qtext)s)) DESC
               ) AS rn
        FROM {table}
        WHERE hierarchy = 'child' AND is_search
          AND (%(scope)s::text IS NULL OR content_hash = %(scope)s::text)
          AND contents_tsv @@ plainto_tsquery('simple', %(qtext)s)
        ORDER BY ts_rank_cd(contents_tsv,
                            plainto_tsquery('simple', %(qtext)s)) DESC
        LIMIT %(wide)s
    ) s GROUP BY parent_id
),
fused AS (
    SELECT COALESCE(d.parent_id, s.parent_id) AS parent_id,
           COALESCE(1.0 / (60 + d.rank), 0) + COALESCE(1.0 / (60 + s.rank), 0) AS score
    FROM dense d FULL OUTER JOIN sparse s USING (parent_id)
    ORDER BY score DESC
    LIMIT %(cand)s
)
SELECT c.parent_id, c.contents, c.content_hash, c.type, c.filename
FROM {table} c JOIN fused f USING (parent_id)
WHERE c.hierarchy = 'parent' AND c.is_search
ORDER BY f.score DESC
"""

# Column order is fixed here so processing_pipeline's row dicts insert directly.
_INSERT_SQL = """
INSERT INTO {table}
    (file_uuid, content_hash, parent_id, hierarchy, type, filename,
     is_search, contents, dense_embedding)
VALUES (%(file_uuid)s, %(content_hash)s, %(parent_id)s, %(hierarchy)s, %(type)s,
        %(filename)s, %(is_search)s, %(contents)s, %(dense_embedding)s)
"""


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
# Query-side helpers: dedup alias resolution & full-document reassembly
# --------------------------------------------------------------------------- #
def resolve_document_scope(file_uuid: str) -> Optional[str]:
    """The content_hash of whatever content this uuid names, or None if unknown.

    Every uuid is a pointer row carrying a content_hash; the chunks belong to the
    content. Returns the hash itself rather than a filter fragment — callers
    bind it as a query parameter, so it is never interpolated into SQL.
    """
    with get_pool().connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT content_hash FROM {config.POSTGRES_TABLE} "
            f"WHERE file_uuid = %s LIMIT 1",
            (file_uuid,),
        )
        row = cursor.fetchone()
    return row[0] if row else None


def fetch_document_text(file_uuid: str) -> str:
    """Reassemble a document's full text from its text parents, in reading order.

    Text parents tile the document and parent_id encodes order; table parents are
    excluded (their cell text is already inline). Follows the dedup alias.
    """
    content_hash = resolve_document_scope(file_uuid)
    if content_hash is None:
        return ""
    with get_pool().connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT contents FROM {config.POSTGRES_TABLE} "
            f"WHERE content_hash = %s AND is_search "
            f"  AND hierarchy = 'parent' AND type = 'text' "
            f"ORDER BY parent_id",
            (content_hash,),
        )
        rows = cursor.fetchall()
    return "\n".join(row[0] for row in rows)


# --------------------------------------------------------------------------- #
# Postgres hybrid search & storage
# --------------------------------------------------------------------------- #
def _pg_read(operation: Callable[[Any], Any]) -> Any:
    """Run a read-only Postgres operation, reconnecting once on failure.

    Reads are idempotent, so a single retry after rebuilding the pool safely
    absorbs transient connection drops.

    Args:
        operation: Callable receiving a connection and returning a result.

    Returns:
        The operation's result.
    """
    try:
        with get_pool().connection() as connection:
            return operation(connection)
    except Exception as exc:
        logger.warning("Postgres read failed (%s); reconnecting and retrying once.", exc)
        reset_pool()
        with get_pool().connection() as connection:
            return operation(connection)


def search_vectors(
    query_text: str,
    encoder: "SentenceTransformer",
    file_uuid: Optional[str] = None,
    top_k: int = 10,
    reranker: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Hybrid (dense + full-text) search over children, returning reranked parents.

    Single-document searches follow the dedup alias; global searches are pinned
    to canonical (is_search) chunks. With a reranker, a wide candidate pool is
    pulled and re-scored, keeping RERANK_TOP_K.
    """
    use_rerank = config.RERANK_ENABLED and reranker is not None
    candidate_k = config.RERANK_CANDIDATES if use_rerank else top_k

    # mmlw-retrieval: dense query needs the query prefix; the full-text leg
    # gets raw text.
    with _rag_encode_lock:
        query_embedding = encoder.encode(config.QUERY_PREFIX + query_text).tolist()

    if file_uuid is not None:
        content_hash = resolve_document_scope(file_uuid)
        if content_hash is None:
            logger.debug("search: document %s unknown; no results", file_uuid)
            return []
    else:
        content_hash = None

    def _search(connection: Any) -> List[Dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(
                _HYBRID_SEARCH_SQL.format(table=config.POSTGRES_TABLE),
                {
                    "qvec": str(query_embedding),
                    "qtext": query_text,
                    "scope": content_hash,
                    "wide": candidate_k * 2,
                    "cand": candidate_k,
                },
            )
            return [
                {"parent_id": r[0], "contents": r[1], "content_hash": r[2],
                 "type": r[3], "filename": r[4]}
                for r in cursor.fetchall()
            ]

    parents = _pg_read(_search)

    if use_rerank and parents:
        with _rerank_lock:
            scores = reranker.predict([(query_text, p["contents"]) for p in parents])
        ranked = sorted(zip(parents, scores), key=lambda ps: float(ps[1]), reverse=True)
        parents = [p for p, _ in ranked[: config.RERANK_TOP_K]]

    logger.debug(
        "search %s (doc=%s, rerank=%s) -> %d parent fragment(s)",
        preview(query_text, 120), file_uuid, use_rerank, len(parents),
    )
    return parents


def _as_params(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a processing_pipeline row for insertion.

    Only the vector needs adapting: pgvector accepts its text form, "[1.0,2.0]",
    which is exactly str() of a Python list. Every other key maps to a column
    of the same name.
    """
    params = dict(row)
    params["dense_embedding"] = str(row["dense_embedding"])
    return params


def upload_to_postgres(
    doc: "PdfBundle",
    file_uuid: str,
    model: "SentenceTransformer",
    filename: str = "",
    content_hash: str = "",
) -> None:
    """Ingest a document into Postgres and name it with a pointer row.

    1. uuid already known -> nothing to do.
    2. content already ingested -> skip chunking, write the pointer only.
    3. new content -> full ingest, then the pointer.
    """
    if not content_hash:
        raise ValueError("content_hash is required; it is the ownership key")

    table = config.POSTGRES_TABLE
    insert_sql = _INSERT_SQL.format(table=table)

    try:
        with get_pool().connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT 1 FROM {table} WHERE file_uuid = %s LIMIT 1",
                    (file_uuid,),
                )
                if cursor.fetchone():
                    logger.info("File %s already present; skipping upload", file_uuid)
                    return

                cursor.execute(
                    f"SELECT 1 FROM {table} WHERE content_hash = %s AND is_search LIMIT 1",
                    (content_hash,),
                )
                if cursor.fetchone():
                    logger.info(
                        "Content of file %s already ingested (hash match); "
                        "writing pointer only", file_uuid,
                    )
                else:
                    logger.info("Processing file %s for upload", file_uuid)
                    rows = processing_pipeline(doc, content_hash, model, filename=filename)
                    if not rows:
                        logger.warning(
                            "Processing produced no rows for file %s; nothing to upload",
                            file_uuid,
                        )
                        return
                    cursor.executemany(insert_sql, [_as_params(row) for row in rows])
                    logger.info("Inserted %d record(s) for file %s", len(rows), file_uuid)

                cursor.execute(
                    insert_sql,
                    _as_params(_pointer_row(
                        file_uuid, filename, content_hash,
                        model.get_sentence_embedding_dimension(),
                    )),
                )
            connection.commit()

    except psycopg.OperationalError:
        reset_pool()
        raise

def delete_document(file_uuid: str) -> Dict[str, Any]:
    """Remove a uuid from Postgres. Idempotent.

    Deletes the uuid's pointer row. The chunks go only when no other uuid still
    names that content, so deleting one copy never breaks its duplicates.

    Returns:
        ``{"deleted": bool, "kind": str}`` where kind is one of
        ``not_found`` | ``pointer_removed`` | ``content_removed``.
    """
    table = config.POSTGRES_TABLE
    try:
        with get_pool().connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT content_hash FROM {table} WHERE file_uuid = %s LIMIT 1",
                    (file_uuid,),
                )
                row = cursor.fetchone()
                if row is None:
                    logger.info(
                        "Delete requested for unknown document %s; nothing to do",
                        file_uuid,
                    )
                    return {"deleted": False, "kind": "not_found"}
                content_hash = row[0]

                cursor.execute(f"DELETE FROM {table} WHERE file_uuid = %s", (file_uuid,))

                # Other pointers naming this content? Checked AFTER the delete,
                # which is safe here (unlike in Milvus) because both statements
                # share one transaction and see each other immediately.
                cursor.execute(
                    f"SELECT 1 FROM {table} "
                    f"WHERE content_hash = %s AND NOT is_search LIMIT 1",
                    (content_hash,),
                )
                if cursor.fetchone():
                    connection.commit()
                    logger.info(
                        "Deleted pointer %s; content still named by other uuid(s)",
                        file_uuid,
                    )
                    return {"deleted": True, "kind": "pointer_removed"}

                cursor.execute(
                    f"DELETE FROM {table} WHERE content_hash = %s AND is_search",
                    (content_hash,),
                )
            connection.commit()

        logger.info("Deleted %s and its content (last reference)", file_uuid)
        return {"deleted": True, "kind": "content_removed"}

    except psycopg.OperationalError:
        reset_pool()
        raise


def _pointer_row(file_uuid: str, filename: str, content_hash: str, embedding_dim: int) -> Dict[str, Any]:
    """Non-searchable row naming a uuid and binding it to a content hash.

    is_search=False excludes it from every query; resolve_document_scope reads
    its content_hash to find the chunks.
    """
    return {
        "file_uuid": file_uuid,
        "content_hash": content_hash,
        "parent_id": file_uuid,  
        "hierarchy": "parent",
        "type": "text",
        "contents": "x",        
        "filename": filename,
        "is_search": False,
        "dense_embedding": [0.0] * embedding_dim,
    }


# --------------------------------------------------------------------------- #
# Document -> PDF conversion (unoserver sidecar)
# --------------------------------------------------------------------------- #
# Accepted content types for conversion. LibreOffice sniffs the format from
# file content, so the extension is not used to select an import filter; this
# map is the supported-types gate. PDF is passthrough, not conversion.
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
    """Convert a document to PDF via the unoserver sidecar, returning PDF bytes.

    PDFs are returned unchanged. Office and text formats go to a LibreOffice
    sidecar container in the same pod, which preserves layout and tables and
    paginates correctly (unlike manual text-to-PDF rendering).

    LibreOffice cannot live in this image: the base image is RHEL 10, which no
    longer ships LibreOffice packages. The sidecar runs Ubuntu, which does.

    Args:
        file_path: Path to the input file.
        content_type: MIME type of the input file.

    Returns:
        The PDF as bytes.

    Raises:
        ValueError: If the content type is not supported.
        RuntimeError: If the conversion fails, times out, or the sidecar is
            unreachable.
    """
    if content_type == "application/pdf":
        with open(file_path, "rb") as handle:
            return handle.read()

    if content_type not in _CONTENT_TYPE_EXTENSION:
        raise ValueError(f"Unsupported content type for conversion: {content_type}")

    from unoserver.client import UnoClient

    logger.debug("Converting '%s' (%s) to PDF via unoserver", file_path, content_type)

    with open(file_path, "rb") as handle:
        source_bytes = handle.read()

    # host_location only affects the inpath (file-path) code path, which we do
    # not use — we always send bytes. Set to 'remote' as documentation: the
    # sidecar shares our network namespace but NOT our filesystem.
    client = UnoClient(
        server=config.UNOSERVER_HOST,
        port=str(config.UNOSERVER_PORT),
        host_location="remote",
    )

    # UnoClient has no timeout of its own. Without one, a sidecar that accepts
    # the connection and then hangs blocks the single ingestion worker forever.
    # setdefaulttimeout is process-global, but conversion only ever runs on that
    # one worker thread, and the previous value is restored in finally.
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(config.UNOSERVER_TIMEOUT)
    try:
        pdf_bytes = client.convert(indata=source_bytes, convert_to="pdf")
    except Exception as exc:
        raise RuntimeError(f"unoserver conversion failed for {file_path}: {exc}") from exc
    finally:
        socket.setdefaulttimeout(previous_timeout)

    if not pdf_bytes:
        raise RuntimeError(f"unoserver returned no PDF for {file_path}")

    logger.debug("unoserver returned %d bytes of PDF", len(pdf_bytes))
    return pdf_bytes

# --------------------------------------------------------------------------- #
# CSX storage retrieval
# --------------------------------------------------------------------------- #
def document_exists(file_uuid: str) -> bool:
    """Check that an object is present in CSX storage.

    Args:
        file_uuid: The gateway's file id.

    Returns:
        True if present, False on 404.

    Raises:
        S3GatewayError: On any non-404 gateway failure.
        requests.exceptions.RequestException: If the gateway is unreachable.
    """
    try:
        get_csx_storage_client().get_metadata(file_uuid)
        return True
    except S3GatewayError as exc:
        if exc.status == 404:
            return False
        raise


def fetch_document(file_uuid: str, dest_path: str) -> tuple[str, str]:
    """Stream an object from CSX storage to a local path.

    Streamed rather than buffered, with a running size guard, so an oversized
    document is refused before it is fully written to disk.

    Args:
        file_uuid: The gateway's file id.
        dest_path: Local path to write to.

    Returns:
        (content_type, filename) as reported by the gateway.

    Raises:
        ValueError: If the object exceeds MAX_DOCUMENT_BYTES.
        S3GatewayError: On gateway failure.
    """
    headers, chunks = get_csx_storage_client().download_stream(file_uuid)

    # Content-Length is -1 when the gateway omits the header; the running
    # check below covers that case.
    if 0 <= config.MAX_DOCUMENT_BYTES < headers.content_length:
        chunks.close()
        raise ValueError(
            f"Document {file_uuid} is {headers.content_length} bytes, "
            f"over the {config.MAX_DOCUMENT_BYTES} limit"
        )

    written = 0
    with open(dest_path, "wb") as sink:
        for chunk in chunks:
            written += len(chunk)
            if written > config.MAX_DOCUMENT_BYTES:
                chunks.close()
                raise ValueError(f"Document {file_uuid} exceeded the size limit mid-stream")
            sink.write(chunk)

    logger.debug("Fetched %s (%d bytes, %s)", file_uuid, written, headers.content_type)
    return headers.content_type, headers.filename or file_uuid
