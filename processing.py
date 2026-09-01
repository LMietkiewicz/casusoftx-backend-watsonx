"""processing.py - PDF text/table extraction, parent-child chunking, and the
ingest pipeline that turns a document into Milvus-ready rows.

Chunking strategy (parent-child / small-to-big retrieval):

* **Parent chunks** are large, overlapping context windows. They are stored but
  NOT semantically embedded (they carry a placeholder dense vector); their job
  is to be returned as context once a child matches.
* **Child chunks** are small, sentence-aware units that ARE embedded (dense) and
  fed to BM25 (sparse). Each child points back to its parent via a globally
  unique ``parent_id``.

Sizing lives in two independent regimes (see also config.py): the LLM
summarization budget is unrelated to the *encoder* sequence limit that bounds
child-chunk size here.

This module imports neither pdfplumber nor sentence-transformers at runtime — the
``pdfplumber.PDF`` and ``SentenceTransformer`` objects are passed in and used via
duck typing; they appear only as type-checking-time annotations.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import config 
from logging_utils import preview

if TYPE_CHECKING:  # imported only for type hints; never required at runtime
    import pdfplumber
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Chunking parameters
# --------------------------------------------------------------------------- #
# Separators tried in priority order by the recursive splitter: paragraph break,
# line break, sentence end. Promote these to config.py if you want them tunable
# per deployment without a code change.
TEXT_SEPARATORS: List[str] = ["\n\n", "\n", ". "]
PARENT_CHUNK_SIZE: int = 1500
PARENT_CHUNK_OVERLAP: int = 0 #DO NOT CHANGE — needs adjusting for full-text search, which is not implemented yet.
CHILD_CHUNK_SIZE: int = 400

# Zero-padding width for the local parent index inside parent_id. Padded so a
# plain lexicographic sort of parent_ids yields reading order. 6 digits keeps the
# composed id at 71 chars, inside the VARCHAR(72) field.
_PARENT_INDEX_WIDTH: int = 6
_PARENT_INDEX_LIMIT: int = 10 ** _PARENT_INDEX_WIDTH

# Polish abbreviations whose trailing period must NOT be treated as a sentence
# boundary. Extend as needed; for high-accuracy Polish segmentation a dedicated
# library (e.g. spaCy `pl_core_news_*`) is the proper upgrade path.
_ABBREVIATIONS = {
    "art", "ust", "pkt", "nr", "tj", "np", "itp", "itd", "ww", "ds", "tzw",
    "tzn", "godz", "ul", "al", "zob", "por", "cyt", "wg", "dot", "m.in",
    "mln", "tys", "proc", "ok", "lit", "rozdz", "par", "str", "poz", "tab",
}


# --------------------------------------------------------------------------- #
# Part 1: Extraction
# --------------------------------------------------------------------------- #
def extract_text_and_tables(doc: "pdfplumber.PDF") -> Tuple[str, List[Dict[str, Any]]]:
    """Extract concatenated text and structured tables from a PDF.

    Note: ``get_text`` already includes table cell text, so table content also
    appears in the text stream; tables are additionally captured structurally
    for dedicated row-level chunking.

    Args:
        doc: An open pdfplumber document.

    Returns:
        A tuple of (all page text joined by blank lines, list of table dicts
        each shaped ``{"page_number": int, "data": List[List]}``).
    """
    text_parts: List[str] = []
    tables: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc):
        text_parts.append(page.get_text("text"))
        for table in page.find_tables():
            data = table.extract()
            if data:
                tables.append({"page_number": page_num + 1, "data": data})

    return "\n\n".join(text_parts).strip(), tables


# --------------------------------------------------------------------------- #
# Part 2: Recursive parent chunker
# --------------------------------------------------------------------------- #
def _hard_split(text: str, size: int, overlap: int = 0) -> List[str]:
    """Split text into fixed-size windows (last-resort, when no separator fits).

    Args:
        text: Text to split.
        size: Maximum window size in characters.
        overlap: Characters shared between consecutive windows.

    Returns:
        Non-empty text windows.
    """
    if size <= 0:
        return [text] if text.strip() else []
    step = max(1, size - overlap)
    windows = [text[i:i + size] for i in range(0, len(text), step)]
    return [w for w in windows if w.strip()]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Prepend a tail of each chunk to the next, creating contextual overlap.

    Args:
        chunks: Ordered, non-overlapping chunks.
        overlap: Characters of the previous chunk to carry into the next.

    Returns:
        Chunks with overlap applied (first chunk unchanged).
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for prev, current in zip(chunks, chunks[1:]):
        tail = prev[-overlap:].lstrip()
        result.append(f"{tail} {current}".strip())
    return result


def _split_text_recursive(
    text: str,
    separators: List[str],
    chunk_size: int,
) -> List[str]:
    """Recursively split text into <=chunk_size pieces. No overlap applied here."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator, remaining = "", []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator, remaining = sep, separators[i + 1:]
            break

    if separator == "":
        # Hard split with NO overlap; overlap is applied once, later.
        return _hard_split(text, chunk_size, overlap=0)

    merged: List[str] = []
    current = ""
    for piece in text.split(separator):
        if len(piece) > chunk_size:
            if current.strip():
                merged.append(current.strip())
                current = ""
            merged.extend(_split_text_recursive(piece, remaining, chunk_size))
            continue
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                merged.append(current.strip())
            current = piece

    if current.strip():
        merged.append(current.strip())

    return merged


def create_text_parent_chunks(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split text into large parent chunks, then apply overlap exactly once."""
    chunks = _split_text_recursive(text, separators, chunk_size)
    return _apply_overlap(chunks, chunk_overlap)


# --------------------------------------------------------------------------- #
# Part 3: Sentence-aware child chunker
# --------------------------------------------------------------------------- #
def _ends_with_abbreviation(text: str) -> bool:
    """Heuristic: does ``text`` end on an abbreviation / initial / enumeration?

    Used to suppress false sentence breaks (e.g. after 'art.' or 'J.' or '1.').

    Args:
        text: A candidate sentence accumulated so far.

    Returns:
        True if the trailing token looks like a non-terminal abbreviation.
    """
    if not text.endswith("."):
        return False
    last_token = text.rsplit(None, 1)[-1].rstrip(".").lower()
    if len(last_token) <= 1 or last_token.isdigit():  # initials, enumerations
        return True
    # Keep letters, digits and internal dots (covers 'm.in').
    token = re.sub(r"[^a-ząćęłńóśźż0-9.]", "", last_token)
    return token in _ABBREVIATIONS


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, protecting common Polish abbreviations.

    Splits on sentence-ending punctuation followed by whitespace and on
    paragraph breaks, then merges fragments that were split after an
    abbreviation, initial, or enumeration marker.

    Args:
        text: Text to segment.

    Returns:
        Sentence strings (whitespace-trimmed, non-empty).
    """
    fragments = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    sentences: List[str] = []
    buffer = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        buffer = f"{buffer} {fragment}".strip() if buffer else fragment
        if not _ends_with_abbreviation(buffer):
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def _group_sentences(sentences: List[str], max_size: int) -> List[str]:
    """Group sentences into chunks of at most ``max_size`` chars, unbroken.

    A sentence longer than ``max_size`` is hard-split (so it cannot exceed the
    encoder's sequence limit and get silently truncated at embedding time).

    Args:
        sentences: Ordered sentences.
        max_size: Maximum child chunk size in characters.

    Returns:
        Child chunk strings.
    """
    children: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_size:
            if current.strip():
                children.append(current.strip())
                current = ""
            children.extend(_hard_split(sentence, max_size))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current.strip():
                children.append(current.strip())
            current = sentence
    if current.strip():
        children.append(current.strip())
    return children


def create_text_child_chunks(
    parent_chunks: List[str],
    child_chunk_size: int,
) -> List[Dict[str, Any]]:
    """Split parent chunks into sentence-aware child chunks.

    Args:
        parent_chunks: Parent chunks from the recursive splitter.
        child_chunk_size: Target maximum child size in characters.

    Returns:
        Child-parent pairs shaped
        ``{"child_contents", "parent_contents", "parent_id", "type"}`` where
        ``parent_id`` is the LOCAL parent index (composed to a global id later).
    """
    pairs: List[Dict[str, Any]] = []
    for local_parent_id, parent in enumerate(parent_chunks):
        if len(parent) <= child_chunk_size:
            children = [parent]
        else:
            children = _group_sentences(_split_into_sentences(parent), child_chunk_size)

        for child in children:
            child = child.strip()
            if child:
                pairs.append(
                    {
                        "child_contents": child,
                        "parent_contents": parent,
                        "parent_id": local_parent_id,
                        "type": "text",
                    }
                )
    return pairs


# --------------------------------------------------------------------------- #
# Part 4: Table chunker
# --------------------------------------------------------------------------- #
def create_table_chunks(
    tables: List[Dict[str, Any]],
    parent_id_start_index: int = 0,
) -> List[Dict[str, Any]]:
    """Create parent-child pairs from structured tables.

    Parent: the whole table as a compact JSON string. Child: one descriptive
    Polish sentence per row, so each row is independently retrievable.

    Args:
        tables: Tables from :func:`extract_text_and_tables`.
        parent_id_start_index: Local parent index offset (continues after the
            text parents so the two id spaces don't overlap before composition).

    Returns:
        Child-parent pairs with ``type == "table"`` and LOCAL ``parent_id``.
    """
    pairs: List[Dict[str, Any]] = []
    for i, table_obj in enumerate(tables):
        local_parent_id = parent_id_start_index + i
        table_data = table_obj.get("data")

        # Must be a non-empty list of rows.
        if not isinstance(table_data, list) or not table_data:
            continue

        # Skip degenerate "tables" (single tiny header, no body).
        header_cells = [
            str(cell).strip() for cell in table_data[0] if cell is not None and str(cell).strip()
        ]
        if len(header_cells) < 2 and len(table_data) < 2:
            continue

        header = [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in table_data[0]
        ]
        data_rows = table_data[1:] if len(table_data) > 1 else []

        # Parent: list-of-dicts JSON for the whole table.
        rows_as_dicts: List[Dict[str, str]] = []
        for row in data_rows:
            row_dict: Dict[str, str] = {}
            for j, cell in enumerate(row):
                col_name = header[j] if j < len(header) else f"column_{j + 1}"
                value = str(cell).replace("\n", " ").strip() if cell is not None else ""
                if col_name:
                    row_dict[col_name] = value
            if row_dict:
                rows_as_dicts.append(row_dict)

        parent_json = json.dumps(rows_as_dicts, ensure_ascii=False, separators=(",", ":"))

        # Children: one sentence per row.
        for row in data_rows:
            clauses: List[str] = []
            for j, cell in enumerate(row):
                col_name = header[j] if j < len(header) else ""
                value = str(cell).replace("\n", " ").strip() if cell is not None else ""
                if col_name and value:
                    clauses.append(f"wartość dla '{col_name}' to '{value}'")
            if clauses:
                child_sentence = (
                    "W tej tabeli wiersz zawiera następujące dane: "
                    + "; ".join(clauses)
                    + "."
                )
                pairs.append(
                    {
                        "child_contents": child_sentence,
                        "parent_contents": parent_json,
                        "parent_id": local_parent_id,
                        "type": "table",
                    }
                )

    return pairs


# --------------------------------------------------------------------------- #
# Part 5: Global id composition
# --------------------------------------------------------------------------- #
def _compose_parent_id(content_hash: str, local_index: int) -> str:
    """Compose a globally unique parent id from a content hash and local index.

    Chunks are owned by content, not by a uuid, so the id is keyed on the hash —
    a second document with identical bytes reuses these exact rows.

    Args:
        content_hash: SHA-256 of the source document's bytes.
        local_index: Per-document parent index (0-based).

    Returns:
        ``f"{content_hash}:{local_index:06d}"``.

    Raises:
        ValueError: If ``local_index`` reaches the padding limit, past which ids
            sort out of order ("…:1000000" precedes "…:999999").
    """
    if local_index >= _PARENT_INDEX_LIMIT:
        raise ValueError(
            f"Content {content_hash[:12]} produced >= {_PARENT_INDEX_LIMIT} parent "
            f"chunks; widen _PARENT_INDEX_WIDTH to preserve reading order."
        )
    return f"{content_hash}:{local_index:0{_PARENT_INDEX_WIDTH}d}"


# --------------------------------------------------------------------------- #
# Part 6: Full ingest pipeline
# --------------------------------------------------------------------------- #
def processing_pipeline(
    doc: "pdfplumber.PDF",
    content_hash: str,
    model: "SentenceTransformer",
    filename: str = "",
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Process a PDF into Milvus-ready parent and child rows.

    Args:
        doc: An open pdfplumber document.
        content_hash: SHA-256 of the source bytes; owns these rows and seeds
            parent-id composition.
        model: Sentence-transformer encoder for dense child embeddings.
        filename: The name of the file.
        show_progress: Whether the encoder shows a progress bar (off in prod).

    Returns:
        Rows ready for ``MilvusClient.insert`` — parent rows (placeholder dense
        vector) and child rows (real dense embedding).
    """
    full_text, tables = extract_text_and_tables(doc)
    logger.debug(
        "Document %s: extracted %d chars of text, %d table(s)",
        content_hash, len(full_text), len(tables),
    )

    text_parents = create_text_parent_chunks(
        full_text, TEXT_SEPARATORS, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP
    )
    text_pairs = create_text_child_chunks(text_parents, CHILD_CHUNK_SIZE)
    table_pairs = create_table_chunks(tables, parent_id_start_index=len(text_parents))

    logger.debug(
        "Document %s: %d parent chunk(s) -> %d text child(ren), %d table child(ren)",
        content_hash, len(text_parents), len(text_pairs), len(table_pairs),
    )
    # Full per-chunk dump only when DEBUG is actually enabled, so the preview
    # strings are never built otherwise. This is the "see the processing
    # results" view for validating chunking on real documents.
    if logger.isEnabledFor(logging.DEBUG):
        for i, parent in enumerate(text_parents):
            logger.debug("  parent[%d] (%d chars): %s", i, len(parent), preview(parent))
        for pair in text_pairs:
            logger.debug(
                "  text-child -> local parent %d: %s",
                pair["parent_id"], preview(pair["child_contents"]),
            )
        for pair in table_pairs:
            logger.debug(
                "  table-child -> local parent %d: %s",
                pair["parent_id"], preview(pair["child_contents"]),
            )

    # Make the encoder's sequence limit an explicit, checked constraint.
    max_seq = getattr(model, "max_seq_length", None)
    if max_seq:
        safe_chars = int(max_seq * config.CHARS_PER_TOKEN)
        if CHILD_CHUNK_SIZE > safe_chars:
            logger.warning(
                "CHILD_CHUNK_SIZE=%d exceeds encoder safe budget ~%d chars "
                "(%d tokens); children may be truncated when embedded.",
                CHILD_CHUNK_SIZE, safe_chars, max_seq,
            )

    embedding_dim = model.get_sentence_embedding_dimension()
    parent_placeholder = [0.0] * embedding_dim

    rows: List[Dict[str, Any]] = []
    seen_parents: set[str] = set()
    child_pairs = text_pairs + table_pairs

    # Parent rows (deduplicated by global id).
    for pair in child_pairs:
        global_id = _compose_parent_id(content_hash, pair["parent_id"])
        if global_id in seen_parents:
            continue
        seen_parents.add(global_id)
        rows.append(
            {
                "file_uuid": "",          
                "content_hash": content_hash,
                "parent_id": global_id,
                "hierarchy": "parent",
                "type": pair["type"],
                "contents": pair["parent_contents"],
                "filename": filename,
                "is_search": True,
                "dense_embedding": parent_placeholder,
            }
        )

    # Batch-embed all child contents in one call, then build child rows.
    contents_to_embed = [pair["child_contents"] for pair in child_pairs]
    embeddings = (
        model.encode(contents_to_embed, show_progress_bar=show_progress)
        if contents_to_embed
        else []
    )

    for pair, embedding in zip(child_pairs, embeddings):
        rows.append(
            {
                "file_uuid": "",
                "content_hash": content_hash,
                "parent_id": _compose_parent_id(content_hash, pair["parent_id"]),
                "hierarchy": "child",
                "type": pair["type"],
                "contents": pair["child_contents"],
                "filename": filename,
                "is_search": True,
                "dense_embedding": embedding.tolist(),
            }
        )

    logger.info(
        "Document %s: %d parent rows, %d child rows",
        content_hash, len(seen_parents), len(contents_to_embed),
    )
    return rows
