"""logging_utils.py - centralized logging configuration and safe content previews.

Application entrypoints (the Flask app, the milvus CLI) call :func:`setup_logging`
exactly once. Library modules must NOT configure logging themselves — they only
``logging.getLogger(__name__)`` and emit records, leaving the routing/level
decision to the entrypoint.
"""
from __future__ import annotations

import logging
from typing import Optional

import config

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure the root logger once, from ``config.LOG_LEVEL``.

    Idempotent via ``force=True`` (replaces any handlers a stray ``basicConfig``
    may have installed). Emits a loud warning when DEBUG content logging is on.
    """
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)

    if config.DEBUG:
        logging.getLogger(__name__).warning(
            "DEBUG logging is ON — logs will contain confidential document content "
            "and LLM inputs/outputs. Do not run this way in production."
        )


def preview(text: object, limit: Optional[int] = None) -> str:
    """Return a length-annotated, truncated preview of ``text`` for debug logs.

    Truncation is log hygiene, not redaction: DEBUG is gated precisely because
    the visible content can be sensitive. Pass a larger ``limit`` (or a big
    number) for content you want in full, such as bounded LLM responses.

    Args:
        text: Any value; coerced to ``str``.
        limit: Max characters before truncation. Defaults to
            ``config.DEBUG_PREVIEW_CHARS``.

    Returns:
        The text, or a truncated form annotated with how much was cut.
    """
    s = str(text)
    cap = limit if limit is not None else config.DEBUG_PREVIEW_CHARS
    if len(s) <= cap:
        return s
    return f"{s[:cap]}… [truncated {len(s) - cap} of {len(s)} chars]" 