"""structured_output.py - schemas, validation, and rendering for structured tasks.

One source of truth per task: Pydantic models for fixed-shape tasks, JSON-schema
builders for the dynamic, API-driven ones (category/department). The same schema
feeds Ollama's ``format`` parameter and watsonx's structured-output mechanism;
the parse/validate helpers are the safety net for the soft (watsonx) path; the
renderers turn validated results into the clean, markdown-free Polish text the
frontend displays.

Design rule: schema field names and enum values are **English**. Polish appears
only in the rendered, user-facing output (via the label maps below).
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from logging_utils import preview

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fixed task schemas
# --------------------------------------------------------------------------- #
class ConfidentialType(str, Enum):
    """Categories of sensitive information (the *kinds* present, never the data)."""

    FIRST_NAMES = "first_names"
    SURNAMES = "surnames"
    ID_NUMBERS = "id_numbers"                # PESEL, NIP, ID card, etc.
    CONTACT_DATA = "contact_data"            # email, phone, address
    FINANCIAL_DATA = "financial_data"        # accounts, amounts, cards
    MEDICAL_DATA = "medical_data"
    CONFIDENTIAL_DECISIONS = "confidential_decisions"
    TRADE_SECRETS = "trade_secrets"
    CASE_NUMBERS = "case_numbers"            # case file signatures, ref numbers


class ConfidentialReport(BaseModel):
    """Whether sensitive data is present, and which categories — not the values."""

    contains_sensitive: bool
    types: List[ConfidentialType] = []


class ExtractionItem(BaseModel):
    """One extracted field. Field/value content is model-produced (Polish); the
    schema keys are English."""

    field: str
    value: str


class ExtractionResult(BaseModel):
    """Open-ended key-value extraction — no fixed field list."""

    data: List[ExtractionItem] = []


# Polish display labels (rendering only; the schema/enum stay English).
_CONFIDENTIAL_LABELS: Dict[ConfidentialType, str] = {
    ConfidentialType.FIRST_NAMES: "Imiona",
    ConfidentialType.SURNAMES: "Nazwiska",
    ConfidentialType.ID_NUMBERS: "Numery identyfikacyjne",
    ConfidentialType.CONTACT_DATA: "Dane kontaktowe",
    ConfidentialType.FINANCIAL_DATA: "Dane finansowe",
    ConfidentialType.MEDICAL_DATA: "Dane medyczne",
    ConfidentialType.CONFIDENTIAL_DECISIONS: "Decyzje poufne",
    ConfidentialType.TRADE_SECRETS: "Tajemnice handlowe",
    ConfidentialType.CASE_NUMBERS: "Sygnatury akt",
}


def confidential_schema() -> Dict[str, Any]:
    """JSON schema for the confidentiality report (for Ollama `format` / watsonx)."""
    return ConfidentialReport.model_json_schema()


def extraction_schema() -> Dict[str, Any]:
    """JSON schema for key-value extraction."""
    return ExtractionResult.model_json_schema()


# --------------------------------------------------------------------------- #
# Dynamic schema builders (forced choice from API-provided classes; no null)
# --------------------------------------------------------------------------- #
def build_category_schema(categories: Dict[str, List[str]]) -> Dict[str, Any]:
    """Build a JSON schema forcing a valid category (+ subcategory when one exists).

    One branch per category ties the subcategory enum to the chosen category: a
    category *with* subcategories requires one; a category *without* omits the
    field. There is no null option — the model must pick a provided class.

    Args:
        categories: Mapping of category name -> list of subcategory names.

    Returns:
        A JSON schema (``anyOf`` of per-category branches).

    Raises:
        ValueError: If ``categories`` is empty.
    """
    if not categories:
        raise ValueError("categories must not be empty")

    branches: List[Dict[str, Any]] = []
    for category, subs in categories.items():
        if subs:
            branches.append({
                "type": "object",
                "properties": {
                    "category": {"const": category},
                    "subcategory": {"enum": list(subs)},
                },
                "required": ["category", "subcategory"],
                "additionalProperties": False,
            })
        else:
            branches.append({
                "type": "object",
                "properties": {"category": {"const": category}},
                "required": ["category"],
                "additionalProperties": False,
            })
    return {"anyOf": branches}


def build_department_schema(departments: List[str]) -> Dict[str, Any]:
    """Build a JSON schema forcing a valid department (required, no null).

    Args:
        departments: List of department names.

    Returns:
        A JSON schema with a required, enum-constrained ``department`` field.

    Raises:
        ValueError: If ``departments`` is empty.
    """
    if not departments:
        raise ValueError("departments must not be empty")
    return {
        "type": "object",
        "properties": {"department": {"enum": list(departments)}},
        "required": ["department"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Parsing & validation (safety net for the soft provider path)
# --------------------------------------------------------------------------- #
def parse_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object from model output.

    Tolerates Markdown fences and surrounding prose. Returns ``{}`` (logged) on
    failure rather than raising.

    Args:
        raw: Raw model output.

    Returns:
        The parsed object, or ``{}``.
    """
    text = re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse JSON object from output: %s", preview(raw))
    return {}


def validate_model(raw: str, model: Type[BaseModel]) -> Optional[BaseModel]:
    """Parse and validate model output into a Pydantic instance.

    On Ollama (grammar-constrained) this always succeeds; on watsonx it is the
    safety net that catches the occasional malformed response.

    Args:
        raw: Raw model output.
        model: Target Pydantic model.

    Returns:
        The validated instance, or ``None`` if it could not be recovered.
    """
    data = parse_json_object(raw)
    if not data:
        return None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        logger.warning("Output failed %s validation: %s", model.__name__, exc)
        return None


# --------------------------------------------------------------------------- #
# Closed-vocabulary coercion (near-miss recovery for the soft provider path)
# --------------------------------------------------------------------------- #
# Terminal fallback when a closed-vocabulary field cannot be resolved: route the
# document to a human instead of emitting a confident wrong answer or a silent
# null. Used as the sentinel by the task-level repair cascade.
MANUAL_REVIEW = "Do ręcznej weryfikacji"

# The allowed confidential-type values, for coercion candidates.
CONFIDENTIAL_TYPE_VALUES: List[str] = [t.value for t in ConfidentialType]

# Coercion tuning. A value must reach this normalized similarity to be accepted,
# and if two or more candidates tie within the epsilon the match is treated as
# ambiguous (rejected), never guessed. The threshold is deliberately permissive
# (0.80 admits a one-character inflection on a 5-char word, e.g. "Umowa"->"Umowy")
# because the tie-guard, not the threshold, is the real protection against wrong
# matches. Tasks may override both from config; tune against real class names.
_DEFAULT_COERCION_THRESHOLD = 0.80
_DEFAULT_TIE_EPSILON = 0.02

_TRUE_TOKENS = {"true", "tak", "yes", "1"}
_FALSE_TOKENS = {"false", "nie", "no", "0"}


def _normalize(s: str) -> str:
    """Casefold and collapse whitespace (diacritics preserved)."""
    return " ".join(s.casefold().split())


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (iterative DP, O(len(a)*len(b)) time)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]: ``1 - levenshtein / max(len)`` after
    normalization. Identical (post-normalization) strings score 1.0."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return 1.0
    longest = max(len(na), len(nb))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein(na, nb) / longest


def coerce_to_vocabulary(
    value: Any,
    candidates: List[str],
    *,
    threshold: float = _DEFAULT_COERCION_THRESHOLD,
    tie_epsilon: float = _DEFAULT_TIE_EPSILON,
) -> Optional[str]:
    """Snap a model-produced value to the nearest allowed candidate, or reject it.

    Exact (post-normalization) matches score 1.0 and win outright. Otherwise the
    best normalized similarity must reach ``threshold``; and if two or more
    candidates fall within ``tie_epsilon`` of the best score, the match is
    ambiguous and rejected — we escalate rather than guess.

    Args:
        value: The model's value (only ``str`` is considered).
        candidates: The closed vocabulary to match against.
        threshold: Minimum normalized similarity to accept.
        tie_epsilon: Candidates within this of the best score count as ties.

    Returns:
        The single best candidate, or ``None`` if below threshold or ambiguous.
    """
    if not isinstance(value, str) or not candidates:
        return None
    scored = [(c, _similarity(value, c)) for c in candidates]
    best = max(score for _, score in scored)
    if best < threshold:
        return None
    winners = [c for c, score in scored if score >= best - tie_epsilon]
    if len(winners) != 1:
        logger.debug("Coercion of %r ambiguous/unmatched (winners=%s)", value, winners)
        return None
    return winners[0]


def _coerce_bool(value: Any) -> Optional[bool]:
    """Interpret a bool or common true/false token (PL/EN); else ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return None


def resolve_category(
    raw: str,
    categories: Dict[str, List[str]],
    *,
    threshold: float = _DEFAULT_COERCION_THRESHOLD,
    tie_epsilon: float = _DEFAULT_TIE_EPSILON,
) -> Optional[Tuple[str, Optional[str]]]:
    """Parse and coerce a category pick (category before subcategory).

    Returns ``(category, subcategory)`` (subcategory present only when the chosen
    category has subcategories), or ``None`` if the category — or a required
    subcategory — cannot be unambiguously resolved.
    """
    data = parse_json_object(raw)
    if not data:
        return None
    category = coerce_to_vocabulary(
        data.get("category"), list(categories), threshold=threshold, tie_epsilon=tie_epsilon
    )
    if category is None:
        return None
    subs = categories.get(category) or []
    if subs:
        subcategory = coerce_to_vocabulary(
            data.get("subcategory"), list(subs), threshold=threshold, tie_epsilon=tie_epsilon
        )
        if subcategory is None:
            return None
        return category, subcategory
    return category, None


def resolve_department(
    raw: str,
    departments: List[str],
    *,
    threshold: float = _DEFAULT_COERCION_THRESHOLD,
    tie_epsilon: float = _DEFAULT_TIE_EPSILON,
) -> Optional[str]:
    """Parse and coerce a department pick, or ``None`` if unresolved."""
    data = parse_json_object(raw)
    if not data:
        return None
    return coerce_to_vocabulary(
        data.get("department"), list(departments), threshold=threshold, tie_epsilon=tie_epsilon
    )


def resolve_confidential(
    raw: str,
    *,
    threshold: float = _DEFAULT_COERCION_THRESHOLD,
    tie_epsilon: float = _DEFAULT_TIE_EPSILON,
) -> Optional[ConfidentialReport]:
    """Parse and coerce a confidentiality report.

    The boolean accepts common PL/EN tokens; each ``types`` entry is coerced to
    the allowed enum. Any unresolvable field yields ``None`` (escalate).
    """
    data = parse_json_object(raw)
    if not data:
        return None
    contains_sensitive = _coerce_bool(data.get("contains_sensitive"))
    if contains_sensitive is None:
        return None
    coerced_types: List[str] = []
    for item in data.get("types", []) or []:
        match = coerce_to_vocabulary(
            item, CONFIDENTIAL_TYPE_VALUES, threshold=threshold, tie_epsilon=tie_epsilon
        )
        if match is None:
            return None
        coerced_types.append(match)
    try:
        return ConfidentialReport(contains_sensitive=contains_sensitive, types=coerced_types)
    except ValidationError:
        return None


# --------------------------------------------------------------------------- #
# Renderers (validated object -> clean, markdown-free Polish text)
# --------------------------------------------------------------------------- #
def render_confidential(report: ConfidentialReport) -> str:
    """Render the confidentiality report as plain Polish text.

    Args:
        report: A validated report.

    Returns:
        Markdown-free text, e.g. "Dane wrażliwe: TAK\\nWykryte typy:\\n- imiona".
    """
    head = "Dane wrażliwe: " + ("TAK" if report.contains_sensitive else "NIE")
    if not report.contains_sensitive or not report.types:
        return head

    lines = [head, "Wykryte typy:"]
    seen: set[ConfidentialType] = set()
    for t in report.types:
        if t in seen:
            continue
        seen.add(t)
        lines.append(f"- {_CONFIDENTIAL_LABELS.get(t, t.value)}")
    return "\n".join(lines)


def render_extraction(result: ExtractionResult) -> str:
    """Render extracted key-value pairs as "Field: Value" lines.

    Args:
        result: A validated extraction result.

    Returns:
        Newline-separated "field: value" lines (field/value are model-produced).
    """
    return "\n".join(f"{item.field}: {item.value}" for item in result.data if item.field)