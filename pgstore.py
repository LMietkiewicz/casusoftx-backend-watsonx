"""pgstore.py - Postgres connection lifecycle and schema management.

This module is the single owner of everything Postgres:

* the shared connection pool, imported by the rest of the app instead of each
  module opening its own connection;
* the table schema and indexes, defined exactly once;
* :func:`ensure_schema` (idempotent, safe to call at startup) and
  :func:`reset_schema` (destructive, manual wipe).

Run as a script:

    python pgstore.py            # idempotent: create the schema if missing
    python pgstore.py --reset    # DESTROYS data: drop and recreate (confirmation)
    python pgstore.py --reset --force   # same, without the confirmation prompt
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import psycopg
from psycopg_pool import ConnectionPool

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #
# Module-level singleton pool, created lazily on first use and reused, so we are
# not paying connection setup on every query/insert. A pool rather than a single
# connection because psycopg connections are NOT thread-safe for concurrent use,
# and /api/query is served by concurrent request threads.
_pool: Optional[ConnectionPool] = None

# Guards only the singleton bookkeeping below (the check/create/clear of the
# shared pointer), NOT the queries themselves — the pool is thread-safe. The
# race this closes is a reset swapping the pool out from under an in-flight op.
# Held for microseconds.
_pool_lock = threading.Lock()


def _dsn() -> str:
    """Build the libpq connection string from config."""
    return (
        f"host={config.POSTGRES_HOST} port={config.POSTGRES_PORT} "
        f"dbname={config.POSTGRES_DB} user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )


def _connect() -> ConnectionPool:
    """Open a new connection pool.

    Returns:
        A freshly constructed, opened :class:`ConnectionPool`.
    """
    logger.info(
        "Connecting to Postgres at %s:%s/%s",
        config.POSTGRES_HOST, config.POSTGRES_PORT, config.POSTGRES_DB,
    )
    pool = ConnectionPool(
        conninfo=_dsn(),
        min_size=1,
        max_size=config.POSTGRES_POOL_MAX,
        timeout=config.POSTGRES_POOL_TIMEOUT,
        open=False,
    )
    pool.open()
    return pool


def get_pool() -> ConnectionPool:
    """Return the shared connection pool, creating it on first call.

    The pool is *not* health-checked on every call. If an operation fails due to
    a dropped connection, call :func:`reset_pool` and retry.

    Returns:
        The shared :class:`ConnectionPool`.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _connect()
        return _pool


def reset_pool() -> None:
    """Discard the cached pool so the next call reconnects.

    Use this after a connection-level failure before retrying an operation.
    """
    global _pool
    with _pool_lock:
        stale, _pool = _pool, None
    if stale is not None:
        try:
            stale.close()
        except Exception as exc:          # closing a broken pool may itself fail
            logger.debug("Ignoring error while closing stale pool: %s", exc)
    logger.info("Postgres pool reset; will reconnect on next use")


# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #
# Text search configuration for the sparse (keyword) half of hybrid search.
# 'simple' = tokenize + lowercase, no stemming and no stop words. 
TEXT_SEARCH_CONFIG = "simple"


def _schema_sql(embedding_dim: int) -> str:
    """Build the CREATE TABLE statement.

    Args:
        embedding_dim: Dimensionality of the dense vector column. Sourced from
            the live encoder at call time so the schema can never drift from
            the model — it is deliberately not configured anywhere.

    Columns:
        id:               auto-generated BIGINT primary key.
        file_uuid:        CSX object uuid. Set on POINTER rows only; empty on
                          chunk rows, which are owned by content, not by a uuid.
        content_hash:     SHA-256 of the source bytes. The ownership key: pointer
                          rows name it, chunk rows carry it.
        parent_id:        globally-unique parent id, composed at ingest as
                          f"{content_hash}:{index:06d}", so retrieval fetches
                          parents with a flat `parent_id = ANY(...)` and reading
                          order is plain lexicographic sort.
        hierarchy:        'parent' or 'child' ('parent' on pointer rows, unused).
        type:             'text' or 'table'.
        filename:         the source file name (for display).
        is_search:        True on chunk rows, False on pointer rows. Every query
                          pins to True so pointer rows never surface.
        contents:         the chunk text.
        dense_embedding:  semantic vector (zero placeholder on pointer rows).
        contents_tsv:     GENERATED from contents; replaces sparse_embedding and
                          the Milvus BM25 function. Maintained by Postgres.

    Returns:
        The DDL as a single statement.
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {config.POSTGRES_TABLE} (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        file_uuid       VARCHAR(36)  NOT NULL DEFAULT '',
        content_hash    VARCHAR(64)  NOT NULL,
        parent_id       VARCHAR(72)  NOT NULL,
        hierarchy       VARCHAR(16)  NOT NULL,
        type            VARCHAR(16)  NOT NULL,
        filename        VARCHAR(512) NOT NULL DEFAULT '',
        is_search       BOOLEAN      NOT NULL,
        contents        TEXT         NOT NULL,
        dense_embedding vector({embedding_dim}) NOT NULL,
        contents_tsv    tsvector
                        GENERATED ALWAYS AS
                        (to_tsvector('{TEXT_SEARCH_CONFIG}', contents)) STORED
    )
    """


def _index_sql() -> list[str]:
    """Build the index statements.

    Replaces the Milvus index_params block. The dense column deliberately gets
    NO index: Milvus used FLAT (exact, exhaustive), and  pgvector's default
    behaviour without an index is the same exact scan. Add HNSW when volume
    demands approximate search — see the commented statement below.

    Returns:
        DDL statements, each idempotent.
    """
    table = config.POSTGRES_TABLE
    return [
        # Sparse half of hybrid search; replaces SPARSE_INVERTED_INDEX + BM25.
        f"CREATE INDEX IF NOT EXISTS {table}_tsv_idx "
        f"ON {table} USING gin (contents_tsv)",

        # Scalar filters on the hot path; replaces the INVERTED indexes.
        # Partial: only pointer rows carry a uuid, so the index stays small.
        f"CREATE INDEX IF NOT EXISTS {table}_file_uuid_idx "
        f"ON {table} (file_uuid) WHERE file_uuid <> ''",
        f"CREATE INDEX IF NOT EXISTS {table}_content_hash_idx "
        f"ON {table} (content_hash)",
        f"CREATE INDEX IF NOT EXISTS {table}_parent_id_idx "
        f"ON {table} (parent_id)",
        # Composite: every search filters on both of these together.
        f"CREATE INDEX IF NOT EXISTS {table}_search_idx "
        f"ON {table} (is_search, hierarchy)",

        # Dense ANN index — enable when the exact scan gets too slow.
        # vector_ip_ops matches the Milvus metric_type="IP" and the <#> operator.
        # NOTE: building this on a large table takes a while and needs
        # maintenance_work_mem raised.
        # f"CREATE INDEX IF NOT EXISTS {table}_dense_idx "
        # f"ON {table} USING hnsw (dense_embedding vector_ip_ops)",
    ]


def _create_schema(connection: psycopg.Connection, embedding_dim: int) -> None:
    """Create the extension, table and indexes. Idempotent at every step.

    Args:
        connection: An open connection.
        embedding_dim: Dense vector dimensionality (from the live encoder).
    """
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(_schema_sql(embedding_dim))
        for statement in _index_sql():
            cursor.execute(statement)
    connection.commit()
    logger.info(
        "Schema '%s' ensured (dim=%d)", config.POSTGRES_TABLE, embedding_dim
    )


def _table_exists(connection: psycopg.Connection) -> bool:
    """Return True if the chunks table is already present."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (config.POSTGRES_TABLE,))
        return cursor.fetchone()[0] is not None


def _stored_dimension(connection: psycopg.Connection) -> Optional[int]:
    """Return the dense_embedding dimension recorded in the live schema.

    Used to detect encoder/schema drift — the failure the Milvus version could
    not check for, because the dimension was fixed at collection-creation time
    and silently disagreed afterwards.

    Returns:
        The dimension, or None if the table or column is absent.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = to_regclass(%s) AND attname = 'dense_embedding'
            """,
            (config.POSTGRES_TABLE,),
        )
        row = cursor.fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return row[0]


# --------------------------------------------------------------------------- #
# Public schema operations
# --------------------------------------------------------------------------- #
def ensure_schema(embedding_dim: int) -> bool:
    """Create the table if it does not already exist. Idempotent.

    Safe to call on every application startup; never drops data. If the table
    exists with a different vector dimension, this raises rather than running
    against a schema the encoder cannot populate.

    Args:
        embedding_dim: Dense vector dimensionality (from the live encoder).

    Returns:
        True if the schema was created, False if it already existed.

    Raises:
        RuntimeError: If the existing table's vector dimension differs from the
            live encoder's.
    """
    with get_pool().connection() as connection:
        if _table_exists(connection):
            stored = _stored_dimension(connection)
            if stored is not None and stored != embedding_dim:
                raise RuntimeError(
                    f"Table '{config.POSTGRES_TABLE}' stores {stored}-dim vectors "
                    f"but the encoder produces {embedding_dim}. The encoder "
                    f"changed; every stored vector is invalid. Re-index with "
                    f"`python postgres.py --reset` and re-ingest."
                )
            logger.info("Table '%s' already exists", config.POSTGRES_TABLE)
            return False
        _create_schema(connection, embedding_dim)
        return True


def reset_schema(embedding_dim: int) -> None:
    """Drop the table (if present) and recreate it empty. DESTRUCTIVE.

    This permanently deletes all stored vectors and metadata. Intended for
    deliberate, manual use only.

    Args:
        embedding_dim: Dense vector dimensionality (from the live encoder).
    """
    with get_pool().connection() as connection:
        with connection.cursor() as cursor:
            logger.warning(
                "Dropping table '%s' and all its data", config.POSTGRES_TABLE
            )
            cursor.execute(f"DROP TABLE IF EXISTS {config.POSTGRES_TABLE}")
        connection.commit()
        _create_schema(connection, embedding_dim)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def _main() -> None:
    """Command-line entrypoint: ensure by default, reset with --reset."""
    import argparse
    import sys

    from logging_utils import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Manage the Postgres schema.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DESTROYS data: drop and recreate the table.",
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
                f"This will DELETE all data in '{config.POSTGRES_TABLE}'. "
                f"Type the table name to confirm: "
            )
            if input(prompt).strip() != config.POSTGRES_TABLE:
                logger.info("Aborted; table unchanged.")
                sys.exit(1)
        reset_schema(embedding_dim)
        logger.info("Table '%s' reset.", config.POSTGRES_TABLE)
    else:
        created = ensure_schema(embedding_dim)
        logger.info("Created." if created else "Already exists; nothing to do.")


if __name__ == "__main__":
    _main()