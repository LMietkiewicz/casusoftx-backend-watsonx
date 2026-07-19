"""milvus.py - Milvus connection lifecycle and collection schema management.

This module is the single owner of everything Milvus:

* the shared :class:`MilvusClient` instance (a lazy singleton with an explicit
  rebuild hook), imported by the rest of the app instead of each module opening
  its own connection;
* the collection schema and indexes, defined exactly once;
* :func:`ensure_collection` (idempotent, safe to call at startup) and
  :func:`reset_collection` (destructive, manual wipe).

Run as a script:

    python milvus.py            # idempotent: create the collection if missing
    python milvus.py --reset    # DESTROYS data: drop and recreate (confirmation)
    python milvus.py --reset --force   # same, without the confirmation prompt
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #
# Module-level singleton. Created lazily on first use and reused thereafter so
# we are not paying connection setup on every query/insert. Callers that detect
# a stale/broken connection should call reset_client() and retry their op.
_client: Optional[MilvusClient] = None

# Guards only the singleton bookkeeping below (the check/create/clear of the
# shared pointer), NOT the Milvus calls themselves — concurrent inserts/searches
# through one client are fine; the race is a reset swapping the client out from
# under an in-flight op. Held for microseconds.
_client_lock = threading.Lock()


def _connect() -> MilvusClient:
    """Open a new Milvus client connection.

    Returns:
        A freshly constructed :class:`MilvusClient`.
    """
    uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
    logger.info("Connecting to Milvus at %s", uri)
    return MilvusClient(uri=uri)


def get_milvus_client() -> MilvusClient:
    """Return the shared Milvus client, creating it on first call.

    The connection is *not* health-checked on every call (that would add a
    round-trip per operation). If an operation fails due to a dropped
    connection, call :func:`reset_client` and retry.

    Returns:
        The shared :class:`MilvusClient`.
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = _connect()
        return _client


def reset_client() -> None:
    """Discard the cached client so the next call reconnects.

    Use this after a connection-level failure before retrying an operation.
    """
    global _client
    with _client_lock:
        _client = None
    logger.info("Milvus client reset; will reconnect on next use")


# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #
def _build_schema(embedding_dim: int) -> "MilvusClient.create_schema":
    """Build the collection schema.

    Args:
        embedding_dim: Dimensionality of the dense vector field. Sourced from
            the live encoder at call time so the schema can never drift from
            the model — it is deliberately not configured anywhere.

    Fields:
        id:               auto-generated INT64 primary key.
        file_id:          INT64 identifier of the source document.
        parent_id:        INT64 globally-unique parent id, composed at ingest as
                          file_id * PARENT_ID_MULTIPLIER + local index, so
                          retrieval fetches parents with a flat `parent_id in [...]`.
        hierarchy:        'parent' or 'child'.
        type:             'text' or 'table'.
        filename:         the source file name (for display).
        content_hash:     the source file's content hash (for deduplication).
        is_search:        whether to include this row in search results.
        contents:         the chunk text; analyzer-enabled to feed BM25.
        dense_embedding:  semantic vector (zero placeholder for parent rows).
        sparse_embedding: BM25 vector produced from `contents` by the function.

    Returns:
        The configured schema object.
    """
    # The standard tokenizer + lowercase filter is language-agnostic. For
    # Polish legal text, a Polish-aware analyzer (stemming / stop words) would
    # improve BM25 recall; left standard here to match the validated setup.
    analyzer_params = {
        "tokenizer": "standard",
        "filter": ["lowercase"],
    }

    schema = MilvusClient.create_schema()

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(field_name="file_id", datatype=DataType.INT64)
    schema.add_field(field_name="parent_id", datatype=DataType.INT64)
    schema.add_field(field_name="hierarchy", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content_hash", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="is_search", datatype=DataType.BOOL)
    schema.add_field(
        field_name="contents",
        datatype=DataType.VARCHAR,
        enable_analyzer=True,
        analyzer_params=analyzer_params,
        max_length=65535,  # Milvus VARCHAR ceiling; table-JSON parents can be large.
    )
    schema.add_field(
        field_name="dense_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=embedding_dim,
    )
    schema.add_field(
        field_name="sparse_embedding",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # BM25 turns `contents` into the sparse vector automatically at insert time.
    bm25 = Function(
        name="BM25",
        function_type=FunctionType.BM25,
        input_field_names=["contents"],
        output_field_names=["sparse_embedding"],
    )
    schema.add_function(bm25)

    return schema


def _build_index_params() -> "MilvusClient.prepare_index_params":
    """Build index parameters for the dense and sparse vector fields.

    Returns:
        The configured index-parameters object.
    """
    index_params = MilvusClient.prepare_index_params()

    # FLAT = exact, exhaustive search. Correct and simple for modest volumes;
    # it scans every vector at query time. Switch to "HNSW" (with an efparams
    # block) once the collection grows large enough to need approximate search.
    index_params.add_index(
        field_name="dense_embedding",
        index_type="FLAT",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        },
    )

    # Scalar indexes for hot-path filter fields (else linear scans).
    index_params.add_index(field_name="file_id", index_type="INVERTED")
    index_params.add_index(field_name="content_hash", index_type="INVERTED")
    index_params.add_index(field_name="is_search", index_type="INVERTED")

    return index_params


def _create_collection(client: MilvusClient, embedding_dim: int) -> None:
    """Create the collection with schema and indexes. Assumes it does not exist.

    Args:
        client: The Milvus client to use.
        embedding_dim: Dense vector dimensionality (from the live encoder).
    """
    client.create_collection(
        collection_name=config.MILVUS_COLLECTION,
        schema=_build_schema(embedding_dim),
        index_params=_build_index_params(),
        consistency_level="Strong",
    )
    logger.info("Collection '%s' created (dim=%d)", config.MILVUS_COLLECTION, embedding_dim)


# --------------------------------------------------------------------------- #
# Public collection operations
# --------------------------------------------------------------------------- #
def ensure_collection(embedding_dim: int) -> bool:
    """Create the collection if it does not already exist. Idempotent.

    Safe to call on every application startup; never drops data.

    Args:
        embedding_dim: Dense vector dimensionality (from the live encoder).

    Returns:
        True if the collection was created, False if it already existed.
    """
    client = get_milvus_client()
    if client.has_collection(config.MILVUS_COLLECTION):
        logger.info("Collection '%s' already exists", config.MILVUS_COLLECTION)
        return False
    _create_collection(client, embedding_dim)
    return True


def reset_collection(embedding_dim: int) -> None:
    """Drop the collection (if present) and recreate it empty. DESTRUCTIVE.

    This permanently deletes all stored vectors and metadata. Intended for
    deliberate, manual use only.

    Args:
        embedding_dim: Dense vector dimensionality (from the live encoder).
    """
    client = get_milvus_client()
    if client.has_collection(config.MILVUS_COLLECTION):
        logger.warning("Dropping collection '%s' and all its data", config.MILVUS_COLLECTION)
        client.drop_collection(config.MILVUS_COLLECTION)
    _create_collection(client, embedding_dim)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def _main() -> None:
    """Command-line entrypoint: ensure by default, reset with --reset."""
    import argparse
    import sys

    from logging_utils import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Manage the Milvus collection.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DESTROYS data: drop and recreate the collection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the confirmation prompt for --reset.",
    )
    args = parser.parse_args()

    # The embedding dimension comes from the live encoder — the single source of
    # truth — so a standalone run loads it to size the schema. Imported locally
    # to keep the heavy ML dependency off the normal import path of this module.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading encoder '%s' to determine embedding dimension...", config.ENCODER_MODEL)
    embedding_dim = SentenceTransformer(config.ENCODER_MODEL).get_sentence_embedding_dimension()

    if args.reset:
        if not args.force:
            prompt = (
                f"This will DELETE all data in '{config.MILVUS_COLLECTION}'. "
                f"Type the collection name to confirm: "
            )
            if input(prompt).strip() != config.MILVUS_COLLECTION:
                logger.info("Aborted; collection unchanged.")
                sys.exit(1)
        reset_collection(embedding_dim)
        logger.info("Collection '%s' reset.", config.MILVUS_COLLECTION)
    else:
        created = ensure_collection(embedding_dim)
        logger.info("Created." if created else "Already exists; nothing to do.")


if __name__ == "__main__":
    _main()
    