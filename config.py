"""config.py - central configuration management for the document-processing backend.

Loads settings from the environment (via a local ``.env`` file when present),
applies typed coercion with sane defaults, and validates the resulting
configuration at import time so that misconfiguration fails fast at startup
rather than surfacing as an obscure runtime error deep in a request handler.

All values are exposed as module-level constants and may be imported directly,
e.g. ``import config; config.MILVUS_HOST``.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from cipher import resolve

# Load variables from a local .env file if one exists. Real environment
# variables always take precedence over .env entries.
load_dotenv()

# Passphrase for decrypting ENC(...) config values. Unset is fine as long as no
# ENC(...) values are present; resolve() raises a clear error if one appears
# without it. Read from the raw env, never itself encrypted.
_CONFIG_PASSPHRASE: Optional[str] = os.getenv("CONFIG_PASSPHRASE")


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Typed environment helpers
# --------------------------------------------------------------------------- #
def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a string environment variable, stripping surrounding whitespace.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The stripped value, or ``default`` when unset.
    """
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The parsed integer.

    Raises:
        ConfigError: If the variable is set but not a valid integer.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Truthy values (case-insensitive): ``1, true, yes, on``.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The parsed boolean.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The parsed float.

    Raises:
        ConfigError: If the variable is set but not a valid float.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc
    
    
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a possibly-encrypted secret, stripping whitespace, then decrypting
    an ENC(...) envelope if present. Plain values pass through untouched."""
    return resolve(_get_str(name, default), _CONFIG_PASSPHRASE)


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
_VALID_PROVIDERS = {"ollama", "watsonx"}

# Explicit provider selection replaces the legacy implicit `API_KEY == ""`
# toggle. Backward-compatibility: if PROVIDER is unset but an API key is
# present, assume watsonx so existing .env files keep working. An explicit
# PROVIDER value always wins.
_explicit_provider = _get_str("PROVIDER")
_legacy_api_key = _get_secret("API_KEY", "")
PROVIDER: str = (
    _explicit_provider.lower()
    if _explicit_provider
    else ("watsonx" if _legacy_api_key else "ollama")
)

# --------------------------------------------------------------------------- #
# Inference provider & model configuration
# --------------------------------------------------------------------------- #
API_KEY: str = _legacy_api_key
PROJECT_ID: str = _get_secret("PROJECT_ID", "")

# --- Model selection (gated) ---
# Default models are pinned in code per provider and are deliberately NOT read
# from the normal env surface, so they cannot be changed by accident during
# deployment. Override is possible but must be doubly explicit (see below).
DEFAULT_MODEL_OLLAMA: str = "llama3.3:70b"   
DEFAULT_MODEL_WATSONX: str = "meta-llama/llama-3-3-70b-instruct"  


def _resolve_model() -> str:
    """Resolve the active model, honoring only a deliberate, gated override.

    An override requires BOTH ``MODEL_OVERRIDE`` (the model string) and
    ``ALLOW_MODEL_OVERRIDE=true``. Supplying one without the other is a hard
    error, so a model change can never happen by accident.

    Returns:
        The resolved model string for the active provider.

    Raises:
        ConfigError: If the override is half-specified.
    """
    override = _get_str("MODEL_OVERRIDE")
    gate = _get_bool("ALLOW_MODEL_OVERRIDE", False)

    if override and gate:
        return override
    if override and not gate:
        raise ConfigError(
            "MODEL_OVERRIDE is set but ALLOW_MODEL_OVERRIDE is not true; "
            "an override must be deliberate — set ALLOW_MODEL_OVERRIDE=true to confirm."
        )
    if gate and not override:
        raise ConfigError("ALLOW_MODEL_OVERRIDE=true but no MODEL_OVERRIDE provided.")

    return DEFAULT_MODEL_WATSONX if PROVIDER == "watsonx" else DEFAULT_MODEL_OLLAMA


MODEL: str = _resolve_model()

# --- Encoder selection (gated) ---
# The embedding dimension is NOT configured — it is read from this encoder at
# runtime so schema and model can never drift. Changing the encoder changes the
# embedding dimension and invalidates ALL stored vectors (requires a full
# reset_collection + re-ingest), hence the gate + warning.
DEFAULT_ENCODER_MODEL: str = "sdadas/mmlw-retrieval-roberta-large-v2"


def _resolve_encoder() -> str:
    """Resolve the encoder model, honoring only a deliberate, gated override."""
    override = _get_str("ENCODER_OVERRIDE")
    gate = _get_bool("ALLOW_ENCODER_OVERRIDE", False)
    if override and gate:
        return override
    if override and not gate:
        raise ConfigError(
            "ENCODER_OVERRIDE is set but ALLOW_ENCODER_OVERRIDE is not true. "
            "Changing the encoder invalidates ALL stored embeddings and requires "
            "a full re-index — set ALLOW_ENCODER_OVERRIDE=true to confirm."
        )
    if gate and not override:
        raise ConfigError("ALLOW_ENCODER_OVERRIDE=true but no ENCODER_OVERRIDE provided.")
    return DEFAULT_ENCODER_MODEL


ENCODER_MODEL: str = _resolve_encoder()

# Dense-query prefix for mmlw-retrieval (queries only; never BM25/passages).
QUERY_PREFIX: str = _get_str("QUERY_PREFIX", "zapytanie: ")

# Base URL of the inference backend (Ollama host or watsonx endpoint).
INFERENCE_PROVIDER_BASE_URL: Optional[str] = _get_str("INFERENCE_PROVIDER_BASE_URL")

# TLS verification for the inference backend. Defaults to True; only disable
# for trusted internal hosts with self-signed certificates.
INFERENCE_PROVIDER_SSL_VERIFY: bool = _get_bool("INFERENCE_PROVIDER_SSL_VERIFY", True)

# Per-request timeout (seconds) for LLM calls.
LLM_REQUEST_TIMEOUT: int = _get_int("LLM_REQUEST_TIMEOUT", 120)

# Output parsing configuration for structured output extraction.    
STRUCTURED_MAX_REPAIRS: int = _get_int("STRUCTURED_MAX_REPAIRS", 1)
COERCION_THRESHOLD: float = _get_float("COERCION_THRESHOLD", 0.80)
COERCION_TIE_EPSILON: float = _get_float("COERCION_TIE_EPSILON", 0.02)

# --------------------------------------------------------------------------- #
# Milvus configuration
# --------------------------------------------------------------------------- #
MILVUS_HOST: str = _get_str("MILVUS_HOST", "localhost")
MILVUS_PORT: int = _get_int("MILVUS_PORT", 19530)
MILVUS_COLLECTION: str = _get_str("MILVUS_COLLECTION", "file_embeddings")

# --------------------------------------------------------------------------- #
# CasuSoft X storage (s3-gateway)
# --------------------------------------------------------------------------- #
CSX_BASE_URL = os.getenv("CSX_BASE_URL", "http://localhost:8989")
CSX_USERNAME = os.getenv("CSX_USERNAME", "")
CSX_PASSWORD = os.getenv("CSX_PASSWORD", "")
CSX_TENANT = os.getenv("CSX_TENANT") or None   # reader resolves by id; None is correct for us
CSX_TIMEOUT = float(os.getenv("CSX_TIMEOUT", "30"))

# Hard cap on a fetched document. MAX_CONTENT_LENGTH no longer protects us now
# that we pull bytes instead of receiving them.
MAX_DOCUMENT_BYTES = int(os.getenv("MAX_DOCUMENT_BYTES", str(200 * 1024 * 1024)))

# --------------------------------------------------------------------------- #
# Document conversion (Gotenberg sidecar)
# --------------------------------------------------------------------------- #
# Base URL of the Gotenberg service used to convert office formats to PDF.
# In a docker-compose setup this is typically the service name, e.g.
# "http://gotenberg:3000".
GOTENBERG_URL: str = _get_str("GOTENBERG_URL", "http://gotenberg:3000")
GOTENBERG_TIMEOUT: int = _get_int("GOTENBERG_TIMEOUT", 60)

# --------------------------------------------------------------------------- #
# Summarization budgets
# --------------------------------------------------------------------------- #
# Token budget that decides single-pass vs. map-reduce summarization and caps
# each map chunk. These sit deliberately BELOW model context limits: a stuffed
# window degrades summary quality (lost-in-the-middle), so this is a quality
# knob, not a capacity ceiling. Tuned independently per provider.
SUMMARY_TOKEN_BUDGET_OLLAMA: int = _get_int("SUMMARY_TOKEN_BUDGET_OLLAMA", 3000)
SUMMARY_TOKEN_BUDGET_WATSONX: int = _get_int("SUMMARY_TOKEN_BUDGET_WATSONX", 20000)

# Active budget resolved against the selected provider.
SUMMARY_TOKEN_BUDGET: int = (
    SUMMARY_TOKEN_BUDGET_WATSONX if PROVIDER == "watsonx" else SUMMARY_TOKEN_BUDGET_OLLAMA
)

# Heuristic characters-per-token ratio used to translate the token budget into
# a character budget for the char-based recursive chunker. ~3.5 is a reasonable
# approximation for Polish text; lower it to be more conservative.
CHARS_PER_TOKEN: float = float(_get_str("CHARS_PER_TOKEN", "3.5"))

# --------------------------------------------------------------------------- #
# RAG retrieval & context budgets
# --------------------------------------------------------------------------- #
# Full-document fast path: a single-document query whose estimated token count
# fits this budget is fed whole (no chunked retrieval). 80% of the active
# model's context window; the 20% is headroom for prompt + answer.
# >>> FILL IN the real *served* context windows (watsonx may cap below native).
_QWEN_CONTEXT_WINDOW: int = _get_int("QWEN_CONTEXT_WINDOW", 30000)    
_LLAMA_CONTEXT_WINDOW: int = _get_int("LLAMA_CONTEXT_WINDOW", 131000)   
_UNKNOWN_CONTEXT_BUDGET: int = _get_int("UNKNOWN_CONTEXT_BUDGET", 10000)


def _resolve_context_budget(model: str) -> int:
    m = (model or "").lower()
    if "qwen" in m:
        return int(_QWEN_CONTEXT_WINDOW * 0.8)
    if "llama" in m:
        return int(_LLAMA_CONTEXT_WINDOW * 0.8)
    return _UNKNOWN_CONTEXT_BUDGET


RAG_CONTEXT_TOKEN_BUDGET: int = _resolve_context_budget(MODEL)

# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
RERANK_ENABLED: bool = _get_bool("RERANK_ENABLED", True)
RERANKER_MODEL: str = _get_str("RERANKER_MODEL", "sdadas/polish-reranker-roberta-v3")
RERANK_CANDIDATES: int = _get_int("RERANK_CANDIDATES", 50)
RERANK_TOP_K: int = _get_int("RERANK_TOP_K", 10)

# --------------------------------------------------------------------------- #
# HTTP server & concurrency
# --------------------------------------------------------------------------- #
APP_PORT: int = _get_int("PORT", 5000)

# Maximum accepted upload size in bytes (default 50 MiB). Enforced by Flask to
# reject oversized payloads before they are buffered to disk.
MAX_CONTENT_LENGTH: int = _get_int("MAX_CONTENT_LENGTH", 50 * 1024 * 1024)

# Upper bound on max pending documents.
# Retry-After header value (seconds) when rejecting requests due to max pending docs.
MAX_PENDING_DOCS: int = _get_int("MAX_PENDING_DOCS", 500)
RETRY_AFTER_SECONDS: int = _get_int("RETRY_AFTER_SECONDS", 10)

# --------------------------------------------------------------------------- #
# Result callbacks
# --------------------------------------------------------------------------- #
CALLBACK_TIMEOUT: int = _get_int("CALLBACK_TIMEOUT", 10)
CALLBACK_MAX_RETRIES: int = _get_int("CALLBACK_MAX_RETRIES", 3)
# Base backoff (seconds) between callback retries; grows linearly per attempt.
CALLBACK_RETRY_BACKOFF: float = float(_get_str("CALLBACK_RETRY_BACKOFF", "1.0"))

# --------------------------------------------------------------------------- #
# Debug mode
# --------------------------------------------------------------------------- #
# --- Debug mode (gated) ---
# Verbose content logging writes confidential document content and LLM I/O to
# logs, so it requires an explicit acknowledgment — and must never run in prod.
DEBUG: bool = _get_bool("DEBUG", False)
_ack_sensitive = _get_bool("ACKNOWLEDGE_SENSITIVE_LOGGING", False)
if DEBUG and not _ack_sensitive:
    raise ConfigError(
        "DEBUG=true requires ACKNOWLEDGE_SENSITIVE_LOGGING=true — debug logs include "
        "confidential document content and LLM I/O. Set it to confirm, and never "
        "enable DEBUG in production."
    )
LOG_LEVEL: str = _get_str("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
DEBUG_PREVIEW_CHARS: int = _get_int("DEBUG_PREVIEW_CHARS", 2000)

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate() -> None:
    """Validate the loaded configuration.

    Checks provider validity and the presence of provider-specific credentials
    and shared required values. Called automatically at import time.

    Raises:
        ConfigError: If any required setting is missing or invalid.
    """
    if PROVIDER not in _VALID_PROVIDERS:
        raise ConfigError(
            f"PROVIDER must be one of {sorted(_VALID_PROVIDERS)}, got {PROVIDER!r}"
        )

    if not INFERENCE_PROVIDER_BASE_URL:
        raise ConfigError("INFERENCE_PROVIDER_BASE_URL is required")

    if not MODEL:
        raise ConfigError(
            f"No model configured for PROVIDER={PROVIDER}; set the pinned "
            f"DEFAULT_MODEL_{PROVIDER.upper()} constant in config.py."
        )
    
    if not ENCODER_MODEL:
        raise ConfigError("No encoder configured; set DEFAULT_ENCODER_MODEL in config.py.")

    if PROVIDER == "watsonx":
        if not API_KEY:
            raise ConfigError("API_KEY is required when PROVIDER=watsonx")
        if not PROJECT_ID:
            raise ConfigError("PROJECT_ID is required when PROVIDER=watsonx")

    if CHARS_PER_TOKEN <= 0:
        raise ConfigError(f"CHARS_PER_TOKEN must be positive, got {CHARS_PER_TOKEN}")


# Fail fast at startup. If you prefer lazy validation (e.g. for test imports),
# remove this call and invoke config.validate() explicitly from the app entrypoint.
validate()
