"""tasks.py - the LLM-driven document tasks: summarization, classification,
extraction, confidentiality, and recommendations.

Design after the structured-output rework:

* Classification/extraction/confidentiality use **structured output** — a JSON
  schema is passed to ``call_llm`` (enforced on Ollama, prompt-directed on
  watsonx), then validated and rendered via ``structured_output``. This removes
  the old JSON-parsing fragility and the separate "formatter" LLM passes.
* Prose tasks (summary, suggested actions) return a single paragraph cleaned by
  the deterministic ``strip_markdown`` instead of an LLM formatting loop.
* Prompts are slim — they target a capable model (Llama). The Qwen preamble is
  the one model-specific workaround kept, since Qwen is overfit to it.

Tasks do not swallow errors; failures propagate to ``handle_task`` in backend.py.
``upload_to_milvus`` now lives in utils.py (it is Milvus infrastructure).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import config
from structured_output import (
    ExtractionResult,
    MANUAL_REVIEW,
    build_category_schema,
    build_department_schema,
    confidential_schema,
    extraction_schema,
    render_confidential,
    render_extraction,
    resolve_category,
    resolve_confidential,
    resolve_department,
    validate_model,
)
from processing import TEXT_SEPARATORS, create_text_parent_chunks
from utils import call_llm, strip_markdown

if TYPE_CHECKING:  # type-only; the doc is passed in, never imported at runtime
    from pdf_io import PdfBundle

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared prompt scaffolding
# --------------------------------------------------------------------------- #
# Qwen is overfit to an English priming preamble before switching to Polish;
# every other model (Llama primary) gets a concise Polish role sentence.
_QWEN_INTRO = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
    "From now on, you will receive instructions in Polish only. "
    "Your answers HAVE TO be in Polish only as well."
)

_ROLE_SUMMARY = "Jesteś asystentem AI streszczającym dokumenty."
_ROLE_CATEGORY = "Jesteś asystentem AI kategoryzującym dokumenty."
_ROLE_DEPARTMENT = "Jesteś asystentem AI przypisującym dokumenty do departamentów."
_ROLE_EXTRACTION = "Jesteś asystentem AI wyodrębniającym dane z dokumentów."
_ROLE_CONFIDENTIAL = "Jesteś asystentem AI wykrywającym dane wrażliwe w dokumentach."
_ROLE_SUGGESTED = "Jesteś asystentem AI proponującym działania na podstawie dokumentów."

# Sampling profiles. Structured tasks use temperature 0 for determinism.
_BALANCED_OPTIONS: Dict[str, Any] = {"temperature": 0.5, "top_p": 0.5, "num_predict": 1000, "repeat_penalty": 1.1}
_PRECISE_OPTIONS: Dict[str, Any] = {"temperature": 0.2, "top_p": 0.5, "num_predict": 1000, "repeat_penalty": 1.1}
_STRUCTURED_OPTIONS: Dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "num_predict": 1024, "repeat_penalty": 1.0}

_MAX_REDUCE_PASSES = 5  # safety bound on hierarchical summary reduction


def _build_system(role: str, body: str) -> str:
    """Assemble a system prompt: model-appropriate intro + task body."""
    intro = _QWEN_INTRO if "qwen" in config.MODEL.lower() else role
    return f"{intro}\n\n{body}"


def _run_llm_task(
    role: str,
    body: str,
    user_input: str,
    options: Optional[Dict[str, Any]] = None,
    *,
    schema: Optional[Dict[str, Any]] = None,
    task_name: str,
) -> str:
    """Build the system prompt and run one LLM call (optionally structured)."""
    logger.debug("Running task %s (structured=%s)", task_name, schema is not None)
    return call_llm(
        prompt=user_input,
        system_message=_build_system(role, body),
        options=options,
        schema=schema,
    )


# --------------------------------------------------------------------------- #
# Structured-output resolution cascade
# --------------------------------------------------------------------------- #
# schema check -> coercion (inside the resolver) -> LLM repair (capped) -> manual.
# The repair re-asks with the rejected answer and an instruction to pick only
# from the allowed options the system prompt already lists.
_REPAIR_DIRECTIVE = (
    "Twoja poprzednia odpowiedź była niepoprawna: nie pasowała do wymaganego "
    "formatu lub zawierała wartość spoza dozwolonej listy. Wybierz wartości "
    "WYŁĄCZNIE z dozwolonych opcji podanych powyżej i zwróć poprawny obiekt."
)


def _repair_call(
    role: str,
    body: str,
    original_input: str,
    bad_raw: str,
    schema: Dict[str, Any],
    task_name: str,
) -> str:
    """Re-request a structured answer, showing the model its rejected output."""
    logger.debug("Repair call for %s", task_name)
    system_message = _build_system(role, body) + "\n\n" + _REPAIR_DIRECTIVE
    prompt = f"{original_input}\n\n[Niepoprawna odpowiedź do poprawienia]:\n{bad_raw}"
    return call_llm(prompt=prompt, system_message=system_message, options=_STRUCTURED_OPTIONS, schema=schema)


def _resolve_with_repair(
    initial_raw: str,
    resolve_once: "Callable[[str], Any]",
    repair: "Callable[[str], str]",
    *,
    max_attempts: int,
    sentinel: Any,
    task_name: str,
) -> Any:
    """Run the resolve -> repair -> manual cascade.

    ``resolve_once`` returns the resolved value (parse + coerce/validate) or
    ``None``. On ``None`` and while attempts remain, ``repair`` produces a fresh
    raw answer and we retry. After ``max_attempts`` repairs the document is
    routed to manual review via ``sentinel``.

    Args:
        initial_raw: The first model output.
        resolve_once: Pure resolver; value or ``None``.
        repair: Re-request callable; takes the bad raw, returns new raw.
        max_attempts: Maximum repair attempts (0 disables repair).
        sentinel: Returned when everything fails (the manual-review value).
        task_name: For logging.

    Returns:
        The resolved value, or ``sentinel``.
    """
    raw = initial_raw
    for attempt in range(max_attempts + 1):
        result = resolve_once(raw)
        if result is not None:
            if attempt:
                logger.info("%s: resolved after %d repair attempt(s)", task_name, attempt)
            return result
        if attempt < max_attempts:
            logger.info("%s: unresolved, repair attempt %d/%d", task_name, attempt + 1, max_attempts)
            raw = repair(raw)
    logger.warning("%s: unresolved after %d attempt(s); routing to manual review", task_name, max_attempts + 1)
    return sentinel


# --------------------------------------------------------------------------- #
# Summarization (token-budget map-reduce; prose output, markdown-stripped)
# --------------------------------------------------------------------------- #
_SUMMARY_MAP_BODY = (
    "Streść poniższy fragment dokumentu zwięźle i formalnie, po polsku. "
    "Zachowaj kluczowe fakty, daty, nazwy stron i istotne postanowienia. "
    "Zwróć wyłącznie samo streszczenie, bez komentarzy."
)
_SUMMARY_FINAL_BODY = (
    "Na podstawie poniższego tekstu napisz jedno zwięzłe, formalne streszczenie "
    "całego dokumentu w jednym akapicie, po polsku. Ujmij najważniejsze punkty, "
    "daty i strony. Zwróć wyłącznie samo streszczenie, bez komentarzy."
)


def _estimate_tokens(text: str) -> int:
    """Approximate token count from character length (config.CHARS_PER_TOKEN)."""
    return int(len(text) / config.CHARS_PER_TOKEN)


def _summarize_text(text: str, *, final: bool) -> str:
    """Summarize one block of text (a map chunk, or the final reduction)."""
    body = _SUMMARY_FINAL_BODY if final else _SUMMARY_MAP_BODY
    return _run_llm_task(
        _ROLE_SUMMARY, body, text, _BALANCED_OPTIONS,
        task_name="SUMMARY_FINAL" if final else "SUMMARY_MAP",
    )


def summary(doc: "PdfBundle") -> str:
    """Summarize a document (single-pass or map-reduce by token budget).

    Args:
        doc: An open PdfBundle object.

    Returns:
        A single-paragraph Polish summary with Markdown stripped, or "" if the
        document has no extractable text.
    """
    full_text = "\n\n".join(doc.page_text(i).strip() for i in range(len(doc))).strip()
    if not full_text:
        logger.warning("Document has no extractable text; returning empty summary")
        return ""

    if _estimate_tokens(full_text) <= config.SUMMARY_TOKEN_BUDGET:
        logger.debug("Summary: single-pass (~%d tokens)", _estimate_tokens(full_text))
        return strip_markdown(_summarize_text(full_text, final=True))

    char_budget = int(config.SUMMARY_TOKEN_BUDGET * config.CHARS_PER_TOKEN)
    overlap = min(200, char_budget // 10)

    chunks = create_text_parent_chunks(full_text, TEXT_SEPARATORS, char_budget, overlap)
    logger.debug("Summary: map-reduce over %d chunk(s)", len(chunks))
    combined = "\n\n".join(s for s in (_summarize_text(c, final=False) for c in chunks) if s.strip())

    for _ in range(_MAX_REDUCE_PASSES):
        if _estimate_tokens(combined) <= config.SUMMARY_TOKEN_BUDGET:
            break
        sub_chunks = create_text_parent_chunks(combined, TEXT_SEPARATORS, char_budget, overlap)
        logger.debug("Summary: reduce pass over %d chunk(s)", len(sub_chunks))
        combined = "\n\n".join(s for s in (_summarize_text(c, final=False) for c in sub_chunks) if s.strip())
    else:
        logger.warning("Summary did not converge under budget; final pass on truncated text")
        combined = combined[:char_budget]

    return strip_markdown(_summarize_text(combined, final=True))


# --------------------------------------------------------------------------- #
# Classification (structured, forced choice)
# --------------------------------------------------------------------------- #
_CATEGORY_BODY = (
    "Przypisz dokument do jednej z dostępnych kategorii. Jeśli wybrana kategoria "
    "ma podkategorie, wybierz również jedną z nich.\n\n"
    "Dostępne kategorie:\n__CATEGORIES__"
)
_DEPARTMENT_BODY = (
    "Przypisz dokument do jednego z dostępnych departamentów.\n\n"
    "Dostępne departamenty:\n__DEPARTMENTS__"
)


def _format_category_listing(categories: Dict[str, List[str]]) -> str:
    """Render the category/subcategory options for the prompt."""
    lines = []
    for category, subs in categories.items():
        lines.append(f"- {category}: {', '.join(subs)}" if subs else f"- {category}")
    return "\n".join(lines)


def _department_names(departments: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """Extract department names from a list of names or {name, description} dicts."""
    return [d["name"] if isinstance(d, dict) else d for d in departments]


def _format_department_listing(departments: List[Union[str, Dict[str, Any]]]) -> str:
    """Render the department options (with descriptions when provided)."""
    lines = []
    for dept in departments:
        if isinstance(dept, dict):
            desc = dept.get("description")
            lines.append(f"- {dept['name']}: {desc}" if desc else f"- {dept['name']}")
        else:
            lines.append(f"- {dept}")
    return "\n".join(lines)


def category_subcategory(
    summary_text: str,
    categories: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """Assign the document to one category (+ subcategory when the category has any).

    Runs the resolve -> coerce -> repair -> manual cascade. On total failure
    returns ``(MANUAL_REVIEW, None)`` so the document is flagged for a human.

    Args:
        summary_text: The document summary.
        categories: Mapping of category name -> subcategory names.

    Returns:
        ``(category, subcategory)``; subcategory is present only when the chosen
        category has subcategories. ``(MANUAL_REVIEW, None)`` on unresolved.
    """
    body = _CATEGORY_BODY.replace("__CATEGORIES__", _format_category_listing(categories))
    schema = build_category_schema(categories)
    initial = _run_llm_task(
        _ROLE_CATEGORY, body, summary_text, _STRUCTURED_OPTIONS,
        schema=schema, task_name="CATEGORY_SUBCATEGORY",
    )
    return _resolve_with_repair(
        initial,
        resolve_once=lambda raw: resolve_category(
            raw, categories,
            threshold=config.COERCION_THRESHOLD, tie_epsilon=config.COERCION_TIE_EPSILON,
        ),
        repair=lambda bad: _repair_call(_ROLE_CATEGORY, body, summary_text, bad, schema, "CATEGORY_SUBCATEGORY"),
        max_attempts=config.STRUCTURED_MAX_REPAIRS,
        sentinel=(MANUAL_REVIEW, None),
        task_name="CATEGORY_SUBCATEGORY",
    )


def department_assignment(
    summary_text: str,
    departments: List[Union[str, Dict[str, Any]]],
) -> Optional[str]:
    """Assign the document to one of the provided departments.

    Runs the resolve -> coerce -> repair -> manual cascade.

    Args:
        summary_text: The document summary.
        departments: Department names, or {name, description} dicts.

    Returns:
        The chosen department name, or ``MANUAL_REVIEW`` on unresolved.
    """
    names = _department_names(departments)
    body = _DEPARTMENT_BODY.replace("__DEPARTMENTS__", _format_department_listing(departments))
    schema = build_department_schema(names)
    initial = _run_llm_task(
        _ROLE_DEPARTMENT, body, summary_text, _STRUCTURED_OPTIONS,
        schema=schema, task_name="DEPARTMENT_ASSIGNMENT",
    )
    return _resolve_with_repair(
        initial,
        resolve_once=lambda raw: resolve_department(
            raw, names,
            threshold=config.COERCION_THRESHOLD, tie_epsilon=config.COERCION_TIE_EPSILON,
        ),
        repair=lambda bad: _repair_call(_ROLE_DEPARTMENT, body, summary_text, bad, schema, "DEPARTMENT_ASSIGNMENT"),
        max_attempts=config.STRUCTURED_MAX_REPAIRS,
        sentinel=MANUAL_REVIEW,
        task_name="DEPARTMENT_ASSIGNMENT",
    )


# --------------------------------------------------------------------------- #
# Extraction (structured; single call replaces extract + format)
# --------------------------------------------------------------------------- #
_EXTRACTION_BODY = (
    "Wyodrębnij z dokumentu najważniejsze dane jako pary pole–wartość: daty, "
    "nazwy stron, kwoty, terminy, numery referencyjne i kluczowe postanowienia. "
    "Dla każdej istotnej informacji podaj nazwę pola i jego wartość."
)


def base_extraction(text: str) -> str:
    """Extract key facts as field/value pairs and render them as text.

    A single structured call replaces the old extract-then-format chain. Open
    field/value content has no closed vocabulary, so the cascade here is
    parse+validate -> repair -> manual (no coercion step). An empty result is a
    valid answer ("nothing found"), rendered as "".

    Args:
        text: The document text/summary.

    Returns:
        "Field: Value" lines, "" if nothing was extracted, or ``MANUAL_REVIEW``
        if the output could not be parsed/validated.
    """
    schema = extraction_schema()
    initial = _run_llm_task(
        _ROLE_EXTRACTION, _EXTRACTION_BODY, text, _STRUCTURED_OPTIONS,
        schema=schema, task_name="BASE_EXTRACTION",
    )
    result = _resolve_with_repair(
        initial,
        resolve_once=lambda raw: validate_model(raw, ExtractionResult),
        repair=lambda bad: _repair_call(_ROLE_EXTRACTION, _EXTRACTION_BODY, text, bad, schema, "BASE_EXTRACTION"),
        max_attempts=config.STRUCTURED_MAX_REPAIRS,
        sentinel=MANUAL_REVIEW,
        task_name="BASE_EXTRACTION",
    )
    return result if isinstance(result, str) else render_extraction(result)


# --------------------------------------------------------------------------- #
# Confidentiality (structured; reports type categories, never the data)
# --------------------------------------------------------------------------- #
_CONFIDENTIAL_BODY = (
    "Oceń, czy dokument zawiera dane wrażliwe, i wskaż wyłącznie ich TYPY — "
    "nigdy samych danych. Wybierz spośród dozwolonych typów:\n"
    "- first_names (imiona)\n"
    "- surnames (nazwiska)\n"
    "- id_numbers (PESEL, NIP, dowód)\n"
    "- contact_data (e-mail, telefon, adres)\n"
    "- financial_data (rachunki, kwoty, karty)\n"
    "- medical_data (dane medyczne)\n"
    "- confidential_decisions (decyzje poufne)\n"
    "- trade_secrets (tajemnice handlowe)\n"
    "- case_numbers (sygnatury akt)"
)


def check_confidential(text: str) -> str:
    """Determine whether the document contains sensitive data and which types.

    Runs the resolve -> coerce -> repair -> manual cascade. Crucially, an
    unresolved result routes to ``MANUAL_REVIEW`` — it never silently defaults to
    "NIE", which would be a fail-open on a safety-relevant field.

    Args:
        text: The document text/summary.

    Returns:
        Rendered Polish report (e.g. "Dane wrażliwe: TAK\\nWykryte typy:\\n- imiona"),
        or ``MANUAL_REVIEW`` if the assessment could not be resolved.
    """
    schema = confidential_schema()
    initial = _run_llm_task(
        _ROLE_CONFIDENTIAL, _CONFIDENTIAL_BODY, text, _STRUCTURED_OPTIONS,
        schema=schema, task_name="CHECK_CONFIDENTIAL",
    )
    result = _resolve_with_repair(
        initial,
        resolve_once=lambda raw: resolve_confidential(
            raw, threshold=config.COERCION_THRESHOLD, tie_epsilon=config.COERCION_TIE_EPSILON,
        ),
        repair=lambda bad: _repair_call(_ROLE_CONFIDENTIAL, _CONFIDENTIAL_BODY, text, bad, schema, "CHECK_CONFIDENTIAL"),
        max_attempts=config.STRUCTURED_MAX_REPAIRS,
        sentinel=MANUAL_REVIEW,
        task_name="CHECK_CONFIDENTIAL",
    )
    return result if isinstance(result, str) else render_confidential(result)


# --------------------------------------------------------------------------- #
# Recommendations & free-form (prose, markdown-stripped)
# --------------------------------------------------------------------------- #
_SUGGESTED_BODY = (
    "Zaproponuj konkretne, praktyczne działania, które należy podjąć w związku z "
    "treścią dokumentu. Odpowiedz jednym formalnym akapitem, po polsku. Jeśli "
    "dokument nie wymaga działań, napisz to wprost."
)


def suggested_action(text: str) -> str:
    """Propose concrete actions for the document (single paragraph).

    Args:
        text: The document text/summary.

    Returns:
        A markdown-free Polish paragraph.
    """
    raw = _run_llm_task(_ROLE_SUGGESTED, _SUGGESTED_BODY, text, _PRECISE_OPTIONS, task_name="SUGGESTED_ACTION")
    return strip_markdown(raw)


def other(prompt: str) -> str:
    """Send a free-form prompt to the model (no system role).

    Args:
        prompt: The prompt to send.

    Returns:
        The model's response, markdown-stripped.
    """
    logger.debug("Running task OTHER")
    return strip_markdown(call_llm(prompt=prompt, options=_PRECISE_OPTIONS))
