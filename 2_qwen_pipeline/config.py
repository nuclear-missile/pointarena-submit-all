#!/usr/bin/env python3
"""Configuration for the Qwen3-8B local data pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Model & Server Configuration
# ---------------------------------------------------------------------------

DEFAULT_VLLM_URL = "http://127.0.0.1:8018/v1/chat/completions"

DEFAULT_ENDPOINT = os.getenv("LOCAL_OPENAI_ENDPOINT", "http://127.0.0.1:8018")

DEFAULT_MODEL_NAME = os.getenv("LOCAL_OPENAI_MODEL", "Qwen3-8B")

# Path to the Qwen3-8B model directory relative to project root
DEFAULT_MODEL_DIR = os.getenv(
    "MODEL_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-8B"),
)

# vLLM server settings
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8018
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_MAX_MODEL_LEN = 8192

# ---------------------------------------------------------------------------
# Category Definitions
# ---------------------------------------------------------------------------

# Three kept categories for the 3-type pipeline
ALLOWED_CATEGORIES: List[str] = ["Affordance", "Object Reference", "Reasoning"]

# Categories that are explicitly excluded from the 3-type pipeline
REMOVED_CATEGORIES: List[str] = ["Counting", "Spatial Relation"]

# Category mappings used for parsing classification output
CATEGORY_ALIASES: Dict[str, str] = {
    "counting": "Counting",
    "spatial relation": "Spatial Relation",
    "spatial": "Spatial Relation",
}


def is_valid_three_type_category(category: Optional[str]) -> bool:
    """Check if a category is one of the three kept types."""
    return str(category or "").strip() in ALLOWED_CATEGORIES


def category_label_display() -> str:
    """Return a newline-separated list of allowed categories for prompt injection."""
    return "\n".join(f"- {label}" for label in ALLOWED_CATEGORIES)

# ---------------------------------------------------------------------------
# Output Directory Configuration
# ---------------------------------------------------------------------------

# Default output root directory name (under pointarena_extract)
DEFAULT_OUTPUT_ROOT = "three_types_qwen3_8b_local"

# Default filenames under output root
REQUESTS_LOG_FILE = "requests_log.jsonl"
SUMMARY_FILE = "summary.json"
SELECTION_STATS_FILE = "selection_stats.json"
TARGET_PROGRESS_FILE = "target_progress.json"

# ---------------------------------------------------------------------------
# Pipeline Stage Names
# ---------------------------------------------------------------------------

STAGE_CLEANING = "cleaning"
STAGE_MULTIPOINT = "multipoint"
STAGE_CLASSIFICATION = "classification"
STAGE_REWRITE = "rewrite"

ALL_STAGES = [STAGE_CLEANING, STAGE_MULTIPOINT, STAGE_CLASSIFICATION, STAGE_REWRITE]

# ---------------------------------------------------------------------------
# Runtime Defaults
# ---------------------------------------------------------------------------

DEFAULT_WORKERS = 32
DEFAULT_TIMEOUT = 60
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 6
DEFAULT_PARSE_RETRIES = 2
DEFAULT_SAMPLE_RETRIES = 4
DEFAULT_REWRITE_RETRIES = 3
