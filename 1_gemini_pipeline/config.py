"""Configuration for the Gemini 4-stage data processing pipeline.

Centralizes all configurable settings: API endpoints, model names,
category definitions, and output directory paths.
"""

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL: str = "https://api.vectorengine.ai/v1/chat/completions"
"""Default API endpoint compatible with OpenAI-style chat completions."""

DEFAULT_MODEL: str = "gemini-3-flash-preview-thinking"
"""Default model identifier used across all pipeline stages."""

API_KEY_ENV_VAR: str = "VECTORENGINE_API_KEY"
"""Environment variable name from which to read the API key."""

# ---------------------------------------------------------------------------
# Category Definitions
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "Reasoning",
    "Spatial Relation",
    "Affordance",
    "Counting",
    "Object Reference",
]
"""The five semantic categories used in classification and rewrite stages."""

CATEGORY_ALIASES: Dict[str, str] = {
    "reasoning": "Reasoning",
    "spatial relation": "Spatial Relation",
    "spatial": "Spatial Relation",
    "affordance": "Affordance",
    "counting": "Counting",
    "object reference": "Object Reference",
    "objectref": "Object Reference",
    "reference": "Object Reference",
    "none": "None",
}
"""Normalisation map for LLM output to canonical category names."""

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT: str = "mix4_first100_clean_multipoint_counting_same_session"
"""Default output root directory for processing results."""

DEFAULT_REQUESTS_LOG: str = "requests_log.jsonl"
"""Filename for the global request log under the output root."""

DEFAULT_SUMMARY_FILE: str = "summary.json"
"""Filename for the processing summary under the output root."""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Read the API key from the environment variable."""
    key = os.getenv(API_KEY_ENV_VAR)
    if not key:
        raise ValueError(
            f"API key not found. Set the {API_KEY_ENV_VAR} environment variable "
            f"or pass it explicitly via --api-key."
        )
    return key


def normalize_category(category: str | None) -> str:
    """Normalise a free-text category string to a canonical label."""
    if category is None:
        return "None"

    key = str(category).strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = __import__("re").sub(r"\s+", " ", key)

    return CATEGORY_ALIASES.get(key, "None")
