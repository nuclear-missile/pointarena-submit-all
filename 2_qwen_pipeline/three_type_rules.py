"""Pure helper rules for the local three-type cleaning pipeline."""

from __future__ import annotations

import re
from typing import Any


ALLOWED_CATEGORIES = ["Affordance", "Object Reference", "Reasoning"]
REMOVED_CATEGORIES = ["Counting", "Spatial Relation"]
_WHITESPACE_RE = re.compile(r"\s+")
_AFFORDANCE_CUE_PATTERNS = (
    r"\bwhere you would\b",
    r"\bwhere (?:a person|someone|somebody|people) would\b",
    r"\bused to\b",
    r"\bused as\b",
    r"\bpeople use\b",
    r"\bperson uses\b",
    r"\buser uses\b",
    r"\bto (?:hold|grip|stabilize|drink|eat|sit|stand|open|close|press|push|pull|turn|type|write|read|cut|wash|carry|wear|hang|rest|lean)\b",
    r"\bfor (?:holding|gripping|stabilizing|drinking|eating|sitting|standing|opening|closing|pressing|pushing|pulling|turning|typing|writing|reading|cutting|washing|carrying|wearing|hanging|resting|leaning)\b",
    r"\b(?:item|object|tool|device|thing|part) that (?:tells time|joins paper together|can be lit|is used|can be used|is for|cuts|opens|closes|holds|stirs|mixes|calculates|measures|writes|reads)\b",
    r"\bdevices? with buttons\b",
    r"\bspace bar\b",
)
_NON_HUMAN_FUNCTION_PATTERNS = (
    r"\bholding up\b",
    r"\bsupporting\b",
    r"\bcarrying people\b",
    r"\bresting upon\b",
    r"\bresting on\b",
)
_EXISTING_POINT_PATTERNS = (
    r"\bexisting point\b",
    r"\bcurrent point\b",
    r"\bexisting location\b",
    r"\bcurrent location\b",
    r"\bcursor\b",
)
_SEMANTIC_REASONING_PATTERNS = (
    r"\bbook about\b",
    r"\bmarriage marker\b",
)
_REASONING_CANDIDATE_PATTERNS = _SEMANTIC_REASONING_PATTERNS + (
    r"\bdirection\b",
    r"\bmoving\b",
    r"\bmost likely\b",
    r"\bwhere people\b",
    r"\bmight\b",
    r"\bcontains?\b",
    r"\bat which\b",
    r"\bin the direction of\b",
    r"\briding\b",
    r"\bbackground\b",
    r"\bhorizon\b",
    r"\bempty space\b",
    r"\bspace between\b",
    r"\bspace to\b",
    r"\bholding up\b",
    r"\bsupporting\b",
    r"\bcarrying\b",
)
_MULTI_INSTANCE_PATTERNS = (
    r"\ball\b",
    r"\bseveral\b",
    r"\bmultiple\b",
    r"\beach\b",
    r"\bevery\b",
)


def category_is_kept(category: str | None) -> bool:
    return str(category or "").strip() in ALLOWED_CATEGORIES


def build_allowed_categories_text() -> str:
    return "\n".join(f"- {label}" for label in ALLOWED_CATEGORIES)


def normalize_local_endpoint(endpoint: str) -> str:
    return str(endpoint).rstrip("/")


def counting_value_gt_one(value: Any) -> bool:
    try:
        return int(value) > 1
    except (TypeError, ValueError):
        return False


def should_drop_for_counting(record: dict[str, Any]) -> bool:
    return counting_value_gt_one(record.get("counting"))


def should_drop_for_multipoint(points_abs: Any, counting_disabled: bool = True) -> bool:
    if not counting_disabled:
        return False
    return isinstance(points_abs, list) and len(points_abs) > 1


def _normalize_text(text: str | None) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("\n", " ")
    value = _WHITESPACE_RE.sub(" ", value)
    return value


def has_explicit_affordance_cue(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _AFFORDANCE_CUE_PATTERNS)


def _has_non_human_function_cue(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _NON_HUMAN_FUNCTION_PATTERNS)


def _has_existing_point_cue(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _EXISTING_POINT_PATTERNS)


def _has_semantic_reasoning_cue(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _SEMANTIC_REASONING_PATTERNS)


def refine_three_type_category(
    original_query: str | None,
    core_target: str | None,
    predicted_category: str | None,
) -> str:
    normalized_category = str(predicted_category or "").strip() or "None"

    joined = " ".join(part for part in (_normalize_text(original_query), _normalize_text(core_target)) if part).strip()
    if _has_existing_point_cue(joined):
        return "Object Reference"
    if _has_semantic_reasoning_cue(joined) and normalized_category == "Object Reference":
        return "Reasoning"
    if normalized_category in {"Object Reference", "None"} and has_explicit_affordance_cue(joined):
        return "Affordance"
    if normalized_category != "Affordance":
        return normalized_category
    if has_explicit_affordance_cue(joined):
        return "Affordance"
    if _has_non_human_function_cue(joined):
        return "Reasoning"
    return "Object Reference"


def rewrite_preserves_category_cues(
    category: str | None,
    original_query: str | None,
    rewritten_query: str | None,
) -> bool:
    normalized_category = str(category or "").strip()
    if normalized_category != "Object Reference":
        return True
    if not _has_existing_point_cue(original_query):
        return True
    return _has_existing_point_cue(rewritten_query)


def _has_multi_instance_language(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _MULTI_INSTANCE_PATTERNS)


def has_reasoning_candidate_cue(text: str | None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _REASONING_CANDIDATE_PATTERNS)


def matches_target_candidate_prefilter(query: str | None, target_category: str | None) -> bool:
    normalized_target = str(target_category or "").strip()
    if normalized_target == "Affordance":
        return has_explicit_affordance_cue(query) and not _has_multi_instance_language(query)
    if normalized_target == "Reasoning":
        return has_reasoning_candidate_cue(query) and not _has_existing_point_cue(query)
    return True
