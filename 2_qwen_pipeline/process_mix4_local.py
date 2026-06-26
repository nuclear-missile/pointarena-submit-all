#!/usr/bin/env python3
"""Process cleaned mix4 samples with same-session cleaning, multipoint detection, classification, and rewrite."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
POINTARENA_EXTRACT_DIR = PROJECT_ROOT / "pointarena_extract"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))   # PointArena root for clean_3types_local
from clean_3types_local.three_type_rules import (  # noqa: E402
    ALLOWED_CATEGORIES,
    build_allowed_categories_text,
    category_is_kept,
    normalize_local_endpoint,
    refine_three_type_category,
    rewrite_preserves_category_cues,
    should_drop_for_counting,
    should_drop_for_multipoint,
)


BY_SOURCE_DIR = PROJECT_ROOT / "ans8" / "data" / "cleaned" / "by_source"
DEFAULT_VALID_CATEGORIES = list(ALLOWED_CATEGORIES)
DEFAULT_ENDPOINT = normalize_local_endpoint(os.getenv("LOCAL_OPENAI_ENDPOINT", "http://127.0.0.1:8018"))
DEFAULT_MODEL = os.getenv("LOCAL_OPENAI_MODEL", "Qwen3-8B")
thread_local = threading.local()


def chat_api_url(endpoint: str) -> str:
    return f"{normalize_local_endpoint(endpoint)}/v1/chat/completions"


CLASSIFICATION_PROMPT_TEMPLATE = """# Role and Objective
You are a highly precise Data Classifier for PointArena-style object-pointing instructions.

Your job is to classify the task into exactly ONE of THREE kept categories:
- Affordance
- Object Reference
- Reasoning

If the task is fundamentally Counting-like or Spatial-Relation-like, output category "None" instead.

# CRITICAL FORMATTING RULES
You must output ONLY a valid, flat JSON object.
- NO Markdown formatting (DO NOT use ```json or ```).
- NO conversational filler.
- Output exactly two keys: "thought" and "category".
- Keep "thought" under 20 words.

Example output:
{
  "thought": "Targets a precise text or pointer-relative reference.",
  "category": "Object Reference"
}

# Valid Category Values For This Run
- Affordance
- Object Reference
- Reasoning
- None

# Strict Rejection To None
- Output "None" for tasks about multiple instances, all items, plural collection scanning, exact counts, or several valid spots.
- Output "None" for tasks whose primary bottleneck is geometric relation such as left/right/above/below/between/nearest/farthest/closest.
- Do NOT force a Spatial Relation or Counting task into one of the three kept categories.

# Semantic Category Definitions & Decision Boundaries
Do not rely on keywords. Evaluate the PRIMARY mental effort required to solve the task.

## 1. Object Reference
- Semantic Feature: A precision crosshair task. The target is picked out by text, UI, cursor/existing-point relation, or very fine-grained pinpointing.
- Key Characteristics:
  - Reading text labels, fields, icons, or interface elements.
  - Existing point / current point / cursor-relative movement.
  - Fine-grained exact parts like the tip, base, crystal, or nearest side window.
  - If the instruction says to move the current point, move slightly up/down/left/right, or find the next/nearest target relative to the existing point, treat it as Object Reference, not Spatial Relation.

## 2. Affordance
- Semantic Feature: The target is defined by explicit human interaction or use, not just by being a usable object.
- Key Characteristics:
  - "tool used to...", "object people use to...", "where you would...", "part you would hold/open/press".
  - Keep Affordance only when the description explicitly frames the target by interaction, usage, or action.
  - A bare object or part name is NOT Affordance just because the object can be used. Examples like "handle", "tool", "amplifier", or "fire equipment" are usually Object Reference unless the wording adds an interaction cue.
  - If the target is defined by support/state/scene physics like "holding up the car" or "supporting the cars", it is more likely Reasoning.

## 3. Reasoning
- Semantic Feature: The target is isolated through world knowledge, state, ownership, order, direction, or scene-level deduction.
- Key Characteristics:
  - Direction of motion, tallest, second from the right, shoe of the cyclist.
  - Part-whole ownership, physical state, function in scene context, or implicit deduction.
  - If you must infer the direction another object is moving, or choose an object using that motion cue, it is Reasoning.
  - If the phrase sounds symbolic, social, or semantic on a physical object/body part rather than literal UI/text reading, prefer Reasoning.
  - If the instruction requires more than direct text reference or simple affordance naming, it often belongs here.

# Output Guardrails
- For "nearest/closest/left/right/above/below/between" style tasks, prefer "None" unless the real bottleneck is cursor-relative Object Reference.
- For "all", "several", "multiple", exact-number, or grouped targets, output "None".
- If the instruction explicitly mentions an existing point, current point, current location, or moving the point until a target is reached, do NOT output "None"; this usually belongs to Object Reference.
- Do not use Affordance for scene-level containers, passive support roles, or abstract likely-to-contain/use cases; those are Reasoning.

# Original instruction:
[INSERT_ORIGINAL_TASK_HERE]

# Cleaned core target:
[INSERT_CORE_TARGET_HERE]"""
NO_THINKING_EXTRA_BODY: Dict[str, Any] = {
    "google": {
        "thinking_config": {
            "include_thoughts": False,
            "thinkingBudget": 0,
        }
    },
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}
CLEANING_PROMPT_TEMPLATE = """# Role
You are a strict Data Gatekeeper for a Visual Grounding/Navigation dataset (Point Arena). Your ONLY job is to filter out invalid tasks and extract the pure visual target from valid ones.

# CRITICAL RULES for Keeping (keep: true) vs Rejecting (keep: false)
1. MUST BE A POINTING TASK (GROUNDING): The answer to the task MUST be a specific location, object, or space in the image (a coordinate).
2. REJECT PURE QA (keep: false): If the instruction asks a question that should be answered with TEXT (e.g., "What is the color...?", "How many...?", "Describe the...?", "Is there a...?"), REJECT IT.
3. ACCEPT EMPTY SPACE: Tasks asking to locate "free space", "vacant area", or "empty space" between/around objects are HIGHLY VALID. Keep them.
4. ACCEPT LOGICAL POINTING: Tasks requiring reasoning to find a location (e.g., "the direction the car is moving", "the tallest building") are VALID. Keep them.

# EXTRACTION RULE (If keep is true)
Extract the "core_target".
- Mentally DELETE ALL formatting garbage: "Your answer should be formatted as a list of tuples", "normalized pixel locations", "Identify several spots", "Locate a few points", "Pinpoint".
- Extract ONLY the noun phrase describing the object or space.
- Example 1: "Pinpoint several spots within the vacant space situated to the left of the object." -> core_target: "the vacant space to the left of the object"
- Example 2: "Point to all 4 tined fork in the image." -> core_target: "all 4 tined fork"

# OUTPUT FORMAT
Output ONLY a flat JSON object. NO Markdown formatting (DO NOT use ```json).
Keys:
- "keep": boolean
- "core_target": string (the cleaned target phrase, empty string if keep is false)
- "reason": string (briefly explain why kept or rejected, especially if rejecting QA)

Input Instruction:
{raw_instruction}"""
MULTIPOINT_PROMPT_TEMPLATE = """# Role
You are a strict answer-shape judge for the Point Arena visual grounding dataset.

# Task
Given the original instruction and the cleaned visual target, determine whether a correct answer should contain MULTIPLE valid points/coordinates rather than one single point.

# DECISION RULES
1. multi_point = true:
- the task asks for all instances of something
- the task asks for several/few/multiple points, spots, or locations
- the target is a free/vacant/empty space where many coordinates are valid
- the target is a plural/group target that should return more than one point
2. multi_point = false:
- the task asks for one best answer, one unique target, one option among choices, or one specific object/location/space

# OUTPUT FORMAT
Output ONLY a flat JSON object. NO Markdown formatting (DO NOT use ```json).
Keys:
- "multi_point": boolean
- "reason": string

Original instruction:
{original_instruction}

Cleaned core_target:
{core_target}"""
REWRITE_PROMPTS: Dict[str, str] = {
    "Reasoning": """You have already classified the original instruction as Reasoning.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Reasoning category.
- The rewritten task must be a single natural English sentence.
- It must start with \"Point to\".
- Keep it short and direct, like PointArena validation queries.
- Keep the cognitive bottleneck on state, causality, world knowledge, physical property, ownership, or scene-level reasoning.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
    "Affordance": """You have already classified the original instruction as Affordance.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Affordance category.
- The rewritten task must be a single natural English sentence.
- It must start with \"Point to\".
- Keep it short and template-like, matching PointArena affordance style.
- Keep the focus on what a person can do with the object, or the object's human-use function.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
    "Object Reference": """You have already classified the original instruction as Object Reference.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Object Reference category.
- The rewritten task must be a single natural English sentence.
- It must start with \"Point to\".
- Keep it short, exact, and PointArena-like.
- Preserve any cue about an existing point, cursor, icon, text/UI element, or very fine-grained pinpointing.
- If the original mentions an existing point, current point, existing location, current location, or cursor, the rewrite MUST keep that cue explicitly.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
}
REWRITE_JSON_FALLBACK_RULES: Dict[str, str] = {
    "Reasoning": "Keep the cognitive bottleneck on state, causality, world knowledge, physical property, ownership, or scene-level reasoning.",
    "Affordance": "Keep the focus on what a person can do with the object, or the object's human-use function.",
    "Object Reference": "Preserve any cue about an existing point, current point, existing location, current location, cursor, icon, text/UI element, or very fine-grained pinpointing. Never drop existing/current point cues if the original contains them.",
}
REWRITE_STYLE_EXAMPLES: Dict[str, List[str]] = {
    "Affordance": [
        "Point to the item that tells time.",
        "Point to the tool used for cutting trees.",
        "Point to the tool that joins paper together.",
    ],
    "Object Reference": [
        "Point to the logo above the existing point.",
        "Point to the right button of the existing point.",
        "Point to the crystal.",
    ],
    "Reasoning": [
        "Point to the moving truck.",
        "Point to where people sit.",
        "Point to the direction in which the car is moving.",
        "Point to the car in the direction of the bike's movement.",
    ],
}


@dataclass
class SampleJob:
    source: str
    source_rank: int
    global_rank: int
    source_file: str
    record: Dict[str, Any]


class RequestLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process mix4 samples with local Qwen three-type cleaning, classification, and rewrite.")
    parser.add_argument("--api-key", default=os.getenv("LOCAL_OPENAI_API_KEY", ""), help="Optional local API key.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Local OpenAI-compatible endpoint base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model name.")
    parser.add_argument("--limit-per-source", type=int, default=100, help="How many records to take from each source file.")
    parser.add_argument("--exclude-sources", default="", help="Comma-separated source names to skip.")
    parser.add_argument("--workers", type=int, default=24, help="Parallel workers across samples.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request read timeout in seconds.")
    parser.add_argument("--connect-timeout", type=int, default=15, help="Per-request connection timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=6, help="Network retry count per request.")
    parser.add_argument("--parse-retries", type=int, default=2, help="How many same-session JSON repair turns to allow per JSON stage.")
    parser.add_argument("--sample-retries", type=int, default=4, help="How many full-sample retries to allow for invalid or truncated outputs.")
    parser.add_argument("--rewrite-retries", type=int, default=3, help="How many same-session rewrite repair turns to allow.")
    parser.add_argument("--output-root", default="three_types_qwen3_8b_local", help="Output root directory.")
    parser.add_argument("--requests-log", default="requests_log.jsonl", help="Global request log filename under output root.")
    parser.add_argument("--summary-file", default="summary.json", help="Summary filename under output root.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if a per-sample output file already exists.")
    parser.add_argument("--counting-enabled", action="store_true", help="Enable Counting category (allows multi-point data).")
    return parser.parse_args()


def get_http_session() -> requests.Session:
    session = getattr(thread_local, "http_session", None)
    if session is None:
        session = requests.Session()
        thread_local.http_session = session
    return session


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def extract_retry_after_seconds(response: requests.Response) -> Optional[float]:
    retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if retry_after:
        retry_after = retry_after.strip()
        if retry_after.isdigit():
            return float(retry_after)

    text = response.text or ""
    patterns = [
        r"wait[:：]?\s*(\d+)\s*seconds",
        r"请等待[:：]?\s*(\d+)\s*seconds",
        r"(\d+)\s*seconds before trying again",
    ]
    waits: List[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                waits.append(float(match.group(1)))
            except (TypeError, ValueError):
                continue
    return max(waits) if waits else None


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe.strip("._") or "sample"


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_code_fence(text)
    candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_category(category: Optional[str]) -> str:
    if category is None:
        return "None"

    key = str(category).strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    alias = {
        "affordance": "Affordance",
        "object reference": "Object Reference",
        "objectref": "Object Reference",
        "reference": "Object Reference",
        "reasoning": "Reasoning",
        "counting": "Counting",
        "spatial relation": "Spatial Relation",
        "spatial": "Spatial Relation",
        "none": "None",
    }
    return alias.get(key, "None")


def classification_counting_disabled(allowed_categories: List[str], counting_enabled: bool = False) -> bool:
    if counting_enabled:
        return False
    return True


def get_allowed_categories_for_args(args: argparse.Namespace) -> List[str]:
    cats = list(DEFAULT_VALID_CATEGORIES)
    if getattr(args, 'counting_enabled', False) and "Counting" not in cats:
        cats.append("Counting")
    return cats


def get_allowed_categories_for_job(job: SampleJob, args: argparse.Namespace) -> List[str]:
    cats = list(DEFAULT_VALID_CATEGORIES)
    if getattr(args, 'counting_enabled', False) and "Counting" not in cats:
        cats.append("Counting")
    return cats


def normalize_allowed_categories(value: Any) -> List[str]:
    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            category = normalize_category(item)
            if category in DEFAULT_VALID_CATEGORIES and category not in normalized:
                normalized.append(category)
        if normalized:
            return normalized
    return list(DEFAULT_VALID_CATEGORIES)


def parse_boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "keep"}:
            return True
        if lowered in {"false", "no", "0", "drop"}:
            return False
    return None


def normalize_short_text(value: Any) -> str:
    if value is None:
        return ""
    text = strip_code_fence(str(value))
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_rewritten_text(raw_text: str) -> str:
    cleaned = strip_code_fence(raw_text)
    parsed = extract_json_object(cleaned)
    if parsed:
        for key in ("rewritten_query", "rewritten_instruction", "rewritten", "rewrite", "query", "rewritten_task", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                break

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]

    cleaned = cleaned.strip().strip('"').strip("'")
    point_match = re.search(r"Point to[\s\S]*", cleaned)
    if point_match:
        cleaned = point_match.group(0).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_valid_rewritten_query(text: Optional[str]) -> bool:
    return isinstance(text, str) and bool(text.strip()) and text.strip().startswith("Point to")


def build_cleaning_prompt(raw_instruction: str) -> str:
    return CLEANING_PROMPT_TEMPLATE.format(raw_instruction=raw_instruction)


def build_cleaning_retry_prompt(raw_instruction: str, previous_output: str) -> str:
    return f"""Your previous cleaning response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- \"keep\": boolean
- \"core_target\": string
- \"reason\": string

Rules:
- If the instruction is unusable, set keep to false and core_target to an empty string.
- If kept, core_target must be the pure visual target phrase.
- reason must briefly explain the keep/reject decision.
- Do not use markdown.

Input Instruction:
{raw_instruction}

Previous invalid output:
{previous_output}"""


def build_multipoint_prompt(original_instruction: str, core_target: str) -> str:
    return MULTIPOINT_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction,
        core_target=core_target,
    )


def build_multipoint_retry_prompt(original_instruction: str, core_target: str, previous_output: str) -> str:
    return f"""Your previous multi-point decision response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- \"multi_point\": boolean
- \"reason\": string

Original instruction:
{original_instruction}

Cleaned core_target:
{core_target}

Previous invalid output:
{previous_output}"""


def build_classification_prompt(original_task: str, core_target: str, allowed_categories: List[str]) -> str:
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.replace("[INSERT_ORIGINAL_TASK_HERE]", original_task)
    return prompt.replace("[INSERT_CORE_TARGET_HERE]", core_target)


def build_classification_retry_prompt(original_task: str, core_target: str, previous_output: str, allowed_categories: List[str]) -> str:
    category_lines = [f"- {category}" for category in allowed_categories] + ["- None"]
    category_text = "\n".join(category_lines)
    return f"""Your previous classification response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- \"thought\": string
- \"category\": string

Valid categories:
{category_text}

If the task is Spatial Relation-like or Counting-like, output "None".

Use the original instruction as the primary signal, and the cleaned core target only as a disambiguation aid.

Original instruction:
{original_task}

Cleaned core target:
{core_target}

Previous invalid output:
{previous_output}"""


def build_rewrite_prompt(category: str, rewrite_input_text: str) -> str:
    template = REWRITE_PROMPTS[category]
    return template.replace("[INSERT_ORIGINAL_TASK_HERE]", rewrite_input_text)


def build_rewrite_retry_prompt(category: str, rewrite_input_text: str, previous_output: str) -> str:
    return f"""Your previous rewrite response was invalid or truncated. Try again in the same session.

Requirements:
- Preserve the same target semantics and keep the task in the {category} category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
- Output only the rewritten question text.

Original instruction:
{rewrite_input_text}

Previous invalid output:
{previous_output}"""


def build_rewrite_json_fallback_prompt(category: str, rewrite_input_text: str, previous_output: Optional[str] = None) -> str:
    category_rule = REWRITE_JSON_FALLBACK_RULES[category]
    style_examples = "\n".join(f"- {item}" for item in REWRITE_STYLE_EXAMPLES[category])
    previous_output_block = ""
    if previous_output:
        previous_output_block = f"\nPrevious unusable output:\n{previous_output}\n"
    return f"""Your previous rewrite attempt did not produce a usable answer. Stay in the same session and answer again.

Return ONLY a flat JSON object with exactly one key:
- "rewritten_query": string

Requirements:
- Preserve the same target semantics and keep the task in the {category} category.
- The rewritten_query value must be one natural English sentence.
- It must start with "Point to".
- Match PointArena style: short, direct, plain, and minimally rewritten.
- If the original instruction already looks like a valid PointArena query, keep it unchanged or change as little as possible.
- Do not change the final target entity or head noun. If the original targets a car, truck, person, book, or traffic light, the rewrite must still target that same kind of thing.
- Do not replace the final target with an intermediate clue, cause, direction cue, or explanation.
- Do not add explanation words such as "located", "typically", "based on", "indicating", "operating", "current state", or any extra justification unless absolutely necessary.
- {category_rule}
- Do not add explanations or extra keys.{previous_output_block}

Style examples for {category}:
{style_examples}

Original instruction:
{rewrite_input_text}"""

def parse_cleaning_output(raw_text: str, parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    keep = parse_boolish(parsed.get("keep"))
    if keep is None:
        return None
    core_target = normalize_short_text(parsed.get("core_target"))
    reason = normalize_short_text(parsed.get("reason"))
    if not reason:
        return None
    if keep and not core_target:
        return None
    if not keep:
        core_target = ""
    return {
        "keep": keep,
        "core_target": core_target,
        "reason": reason,
    }


def parse_multipoint_output(raw_text: str, parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    multi_point = parse_boolish(parsed.get("multi_point"))
    reason = normalize_short_text(parsed.get("reason"))
    if multi_point is None or not reason:
        return None
    return {
        "multi_point": multi_point,
        "reason": reason,
    }


def parse_classification_output(
    raw_text: str,
    parsed: Optional[Dict[str, Any]],
    allowed_categories: List[str],
    original_task: str = "",
    core_target: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    thought = str(parsed.get("thought") or parsed.get("analysis") or "").strip()
    raw_category = normalize_category(parsed.get("category"))
    category = refine_three_type_category(original_task, core_target, raw_category)
    if not thought:
        return None
    normalized_allowed_categories = normalize_allowed_categories(allowed_categories)
    return {
        "thought": thought,
        "raw_category": raw_category,
        "category": category,
        "category_refined": category != raw_category,
        "allowed_categories": normalized_allowed_categories,
        "is_valid_category": category in normalized_allowed_categories,
    }


def parse_rewrite_output(raw_text: str) -> Optional[Dict[str, Any]]:
    rewritten = normalize_rewritten_text(raw_text)
    if not is_valid_rewritten_query(rewritten):
        return None
    return {
        "rewritten_query": rewritten,
    }


class ChatSession:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        sample_id: str,
        source: str,
        request_logger: RequestLogger,
        timeout: int,
        connect_timeout: int,
        max_retries: int,
        initial_messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.sample_id = sample_id
        self.source = source
        self.request_logger = request_logger
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.messages: List[Dict[str, str]] = list(initial_messages or [])

    def send_message(
        self,
        user_content: str,
        stage: str,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages + [{"role": "user", "content": user_content}],
            "temperature": 0,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body is not None:
            payload["extra_body"] = extra_body

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        attempts: List[Dict[str, Any]] = []
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            started = time.time()
            log_record: Dict[str, Any] = {
                "timestamp": utc_now(),
                "sample_id": self.sample_id,
                "source": self.source,
                "stage": stage,
                "attempt": attempt,
                "model": self.model,
                "request": payload,
            }
            attempt_record: Dict[str, Any] = {
                "stage": stage,
                "attempt": attempt,
                "timestamp": log_record["timestamp"],
                "timeout": {
                    "connect_timeout_sec": self.connect_timeout,
                    "read_timeout_sec": self.timeout,
                },
            }
            try:
                response = get_http_session().post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout, self.timeout),
                )
                elapsed = round(time.time() - started, 3)
                attempt_record["elapsed_sec"] = elapsed
                attempt_record["http_status"] = response.status_code
                attempt_record["response_text"] = response.text
                log_record["elapsed_sec"] = elapsed
                log_record["http_status"] = response.status_code
                log_record["response_text"] = response.text
                self.request_logger.log(log_record)
                attempts.append(attempt_record)

                if response.status_code == 429:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    attempt_record["error"] = last_error
                    server_retry_after = extract_retry_after_seconds(response)
                    retry_wait = 15.0 * (1.7 ** (attempt - 1)) + random.uniform(0.0, 5.0)
                    if server_retry_after is not None:
                        attempt_record["server_retry_after_sec"] = round(server_retry_after, 3)
                        retry_wait = max(retry_wait, server_retry_after + random.uniform(1.0, 3.0))
                    retry_wait = min(300.0, retry_wait)
                    attempt_record["retry_wait_sec"] = round(retry_wait, 3)
                    time.sleep(retry_wait)
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    attempt_record["error"] = last_error
                    retry_wait = min(8.0, (1.6 ** (attempt - 1)) + random.random())
                    attempt_record["retry_wait_sec"] = round(retry_wait, 3)
                    time.sleep(retry_wait)
                    continue

                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                    attempt_record["error"] = last_error
                    break

                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                assistant_content = str(message.get("content", ""))
                reasoning_content = str(message.get("reasoning_content", ""))
                finish_reason = choice.get("finish_reason")

                self.messages.append({"role": "user", "content": user_content})
                self.messages.append({"role": "assistant", "content": assistant_content})
                return {
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "reasoning_content": reasoning_content,
                    "finish_reason": finish_reason,
                    "usage": data.get("usage"),
                    "attempts": attempts,
                    "response_json": data,
                }
            except requests.exceptions.Timeout as exc:
                elapsed = round(time.time() - started, 3)
                last_error = f"timeout:{exc.__class__.__name__}: {exc}"
                attempt_record["elapsed_sec"] = elapsed
                attempt_record["error_type"] = "timeout"
                attempt_record["error"] = last_error
                log_record["elapsed_sec"] = elapsed
                log_record["error_type"] = "timeout"
                log_record["error"] = last_error
                self.request_logger.log(log_record)
                attempts.append(attempt_record)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, (1.6 ** (attempt - 1)) + random.random()))
            except requests.exceptions.RequestException as exc:
                elapsed = round(time.time() - started, 3)
                last_error = f"request_exception:{exc.__class__.__name__}: {exc}"
                attempt_record["elapsed_sec"] = elapsed
                attempt_record["error_type"] = "request_exception"
                attempt_record["error"] = last_error
                log_record["elapsed_sec"] = elapsed
                log_record["error_type"] = "request_exception"
                log_record["error"] = last_error
                self.request_logger.log(log_record)
                attempts.append(attempt_record)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, (1.6 ** (attempt - 1)) + random.random()))
            except Exception as exc:  # noqa: BLE001
                elapsed = round(time.time() - started, 3)
                last_error = str(exc)
                attempt_record["elapsed_sec"] = elapsed
                attempt_record["error_type"] = "unexpected_exception"
                attempt_record["error"] = last_error
                log_record["elapsed_sec"] = elapsed
                log_record["error_type"] = "unexpected_exception"
                log_record["error"] = last_error
                self.request_logger.log(log_record)
                attempts.append(attempt_record)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, (1.6 ** (attempt - 1)) + random.random()))

        raise RuntimeError(f"{stage} failed after {self.max_retries} retries: {last_error}")


def run_json_stage(
    session: ChatSession,
    initial_prompt: str,
    stage: str,
    parse_retries: int,
    parse_output,
    retry_prompt_builder,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    prompt = initial_prompt
    attempts: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    last_turn: Optional[Dict[str, Any]] = None
    last_parsed: Optional[Dict[str, Any]] = None
    normalized: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    for turn_idx in range(1, parse_retries + 2):
        stage_name = stage if turn_idx == 1 else f"{stage}_repair_{turn_idx - 1}"
        try:
            turn = session.send_message(
                prompt,
                stage=stage_name,
                response_format={"type": "json_object"},
                extra_body=NO_THINKING_EXTRA_BODY,
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "stage": stage_name,
                    "attempt": "session_failed",
                    "timestamp": utc_now(),
                    "error": str(exc),
                }
            )
            last_error = str(exc)
            break

        last_turn = turn
        last_parsed = extract_json_object(turn["assistant_content"])
        normalized = parse_output(turn["assistant_content"], last_parsed)
        attempts.extend(turn["attempts"])
        turns.append(
            {
                "stage": stage_name,
                "prompt": prompt,
                "raw_output": turn["assistant_content"],
                "reasoning_content": turn["reasoning_content"],
                "finish_reason": turn["finish_reason"],
                "parsed": last_parsed,
                "normalized": normalized,
                "attempts": turn["attempts"],
            }
        )

        if turn["finish_reason"] != "length" and normalized is not None:
            return last_turn, last_parsed, normalized, attempts, turns, None

        if turn_idx <= parse_retries:
            prompt = retry_prompt_builder(turn["assistant_content"])

    return last_turn, last_parsed, normalized, attempts, turns, last_error


def parse_csv_sources(raw_value: str) -> set[str]:
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def select_jobs(limit_per_source: int, exclude_sources: Optional[set[str]] = None) -> List[SampleJob]:
    if not BY_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing by-source directory: {BY_SOURCE_DIR}")

    jobs: List[SampleJob] = []
    global_rank = 0
    exclude_sources = exclude_sources or set()
    for source_file in sorted(BY_SOURCE_DIR.glob("*.jsonl")):
        source = source_file.stem
        if source in exclude_sources:
            continue
        with source_file.open("r", encoding="utf-8") as f:
            for source_rank, line in enumerate(f, start=1):
                if source_rank > limit_per_source:
                    break
                record = json.loads(line)
                global_rank += 1
                jobs.append(
                    SampleJob(
                        source=source,
                        source_rank=source_rank,
                        global_rank=global_rank,
                        source_file=str(source_file),
                        record=record,
                    )
                )
    return jobs


def output_path_for_job(output_root: Path, job: SampleJob) -> Path:
    dataset_dir = output_root / job.source
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sanitize_filename(job.record.get("id", f"{job.source}_{job.source_rank}"))
    return dataset_dir / f"{job.source_rank:04d}__{safe_id}.json"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def validate_sample_payload(sample: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not isinstance(sample, dict):
        return ["payload_not_dict"]

    processing = sample.get("processing")
    record = sample.get("record")
    cleaning = sample.get("cleaning")
    multipoint = sample.get("multipoint")
    classification = sample.get("classification")
    rewrite = sample.get("rewrite")
    session_messages = sample.get("session_messages")

    if not isinstance(processing, dict):
        reasons.append("missing_processing")
        return reasons
    if not isinstance(record, dict):
        reasons.append("missing_record")
    if not isinstance(cleaning, dict):
        reasons.append("missing_cleaning")
        return reasons
    if not isinstance(multipoint, dict):
        reasons.append("missing_multipoint")
        return reasons
    if not isinstance(classification, dict):
        reasons.append("missing_classification")
        return reasons
    if not isinstance(rewrite, dict):
        reasons.append("missing_rewrite")
        return reasons
    if not isinstance(session_messages, list):
        reasons.append("missing_session_messages")

    status = processing.get("status")
    filter_stage = processing.get("filter_stage")
    if status not in {"completed", "filtered_out"}:
        reasons.append(f"non_terminal_status:{status}")
    if not processing.get("completed_at"):
        reasons.append("missing_completed_at")

    cleaning_raw = cleaning.get("raw_output")
    cleaning_parsed = cleaning.get("parsed")
    cleaning_keep = cleaning.get("keep")
    cleaning_core_target = cleaning.get("core_target")
    cleaning_reason = cleaning.get("reason")
    cleaning_finish = cleaning.get("finish_reason")
    if not isinstance(cleaning_raw, str) or not cleaning_raw.strip():
        reasons.append("cleaning_missing_raw_output")
    if cleaning_finish == "length":
        reasons.append("cleaning_finish_length")
    if not isinstance(cleaning_parsed, dict):
        reasons.append("cleaning_missing_parsed")
    if not isinstance(cleaning_keep, bool):
        reasons.append("cleaning_missing_keep")
    if not isinstance(cleaning_reason, str) or not cleaning_reason.strip():
        reasons.append("cleaning_missing_reason")
    if cleaning_keep is True and (not isinstance(cleaning_core_target, str) or not cleaning_core_target.strip()):
        reasons.append("cleaning_missing_core_target")
    if cleaning_keep is False and status == "completed":
        reasons.append("cleaning_keep_false_but_completed")
    if cleaning_keep is False and status == "filtered_out" and filter_stage != "cleaning":
        reasons.append("filtered_out_from_wrong_stage_for_cleaning")

    if cleaning_keep is True:
        multipoint_raw = multipoint.get("raw_output")
        multipoint_parsed = multipoint.get("parsed")
        multipoint_value = multipoint.get("multi_point")
        multipoint_reason = multipoint.get("reason")
        multipoint_finish = multipoint.get("finish_reason")
        if not isinstance(multipoint_raw, str) or not multipoint_raw.strip():
            reasons.append("multipoint_missing_raw_output")
        if multipoint_finish == "length":
            reasons.append("multipoint_finish_length")
        if not isinstance(multipoint_parsed, dict):
            reasons.append("multipoint_missing_parsed")
        if not isinstance(multipoint_value, bool):
            reasons.append("multipoint_missing_value")
        if not isinstance(multipoint_reason, str) or not multipoint_reason.strip():
            reasons.append("multipoint_missing_reason")

        classification_category = classification.get("category")
        classification_finish = classification.get("finish_reason")
        classification_raw = classification.get("raw_output")
        classification_parsed = classification.get("parsed")
        thought = classification.get("thought")
        allowed_categories = normalize_allowed_categories(classification.get("allowed_categories"))
        classification_required = multipoint_value is False or processing.get("counting_enabled") is True

        if not classification_required:
            if status != "filtered_out" or filter_stage != "multipoint":
                reasons.append("multipoint_true_should_filter_out")
            if classification_category not in (None, ""):
                reasons.append("classification_should_be_empty_when_multipoint_filters")
            if thought not in (None, ""):
                reasons.append("classification_thought_should_be_empty_when_multipoint_filters")
            if classification_raw not in (None, ""):
                reasons.append("classification_raw_should_be_empty_when_multipoint_filters")
            if classification_parsed not in (None, {}):
                reasons.append("classification_parsed_should_be_empty_when_multipoint_filters")
        else:
            if not isinstance(classification_raw, str) or not classification_raw.strip():
                reasons.append("classification_missing_raw_output")
            if classification_finish == "length":
                reasons.append("classification_finish_length")
            if not isinstance(classification_parsed, dict):
                reasons.append("classification_missing_parsed")
            if not isinstance(thought, str) or not thought.strip():
                reasons.append("classification_missing_thought")
            if status == "completed" and classification_category not in allowed_categories:
                reasons.append(f"classification_invalid_category:{classification_category}")
            if status == "filtered_out" and filter_stage == "classification" and classification_category in allowed_categories:
                reasons.append("filtered_out_but_category_is_valid")
    else:
        if any(multipoint.get(key) not in (None, [], {}) for key in ("raw_output", "parsed", "multi_point", "reason", "finish_reason")):
            reasons.append("multipoint_should_be_empty_when_cleaning_rejects")

    if status == "completed":
        rewrite_raw = rewrite.get("raw_output")
        rewrite_text = rewrite.get("rewritten_query")
        rewrite_finish = rewrite.get("finish_reason")
        if not isinstance(rewrite_raw, str) or not rewrite_raw.strip():
            reasons.append("rewrite_missing_raw_output")
        if rewrite_finish == "length":
            reasons.append("rewrite_finish_length")
        if not isinstance(rewrite_text, str) or not rewrite_text.strip():
            reasons.append("rewrite_missing_query")
        elif not rewrite_text.strip().startswith("Point to"):
            reasons.append("rewrite_not_pointarena_style")

    if isinstance(session_messages, list):
        if status == "completed" and len(session_messages) < 6:
            reasons.append("session_messages_too_short_completed")
        if status == "filtered_out" and filter_stage == "cleaning" and len(session_messages) < 2:
            reasons.append("session_messages_too_short_cleaning_filter")
        if status == "filtered_out" and filter_stage == "multipoint" and len(session_messages) < 4:
            reasons.append("session_messages_too_short_multipoint_filter")
        if status == "filtered_out" and filter_stage == "classification" and len(session_messages) < 6:
            reasons.append("session_messages_too_short_classification_filter")

    return reasons


def load_existing_sample(path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, [f"file_read_or_json_error:{exc}"]
    return sample, validate_sample_payload(sample)


def get_rule_filter_reason(job: SampleJob, original_task: str, counting_enabled: bool = False) -> Optional[Tuple[str, str]]:
    points_abs = job.record.get("points_abs")
    if not isinstance(points_abs, list) or len(points_abs) == 0:
        return (
            "missing_points_abs",
            "Directly filtered because the original record has no usable points_abs supervision.",
        )
    if should_drop_for_counting(job.record):
        return (
            "counting_gt_one",
            "Directly filtered because the record-level counting field is greater than 1.",
        )
    if should_drop_for_multipoint(points_abs, counting_disabled=not counting_enabled):
        return (
            "points_abs_multi_point",
            "Directly filtered because the sample has multiple valid supervision points and is outside the local three-type single-point target scope.",
        )
    if job.source == "refspatial" and " (A) " in original_task:
        return (
            "refspatial_multiple_choice",
            "Directly filtered refspatial multiple-choice query containing ' (A) '.",
        )
    return None


def apply_rule_filter(file_payload: Dict[str, Any], rule_name: str, reason: str) -> Dict[str, Any]:
    synthetic = {"keep": False, "core_target": "", "reason": reason}
    raw_output = json.dumps(synthetic, ensure_ascii=False)
    prompt = file_payload["cleaning"]["prompt"]
    file_payload["cleaning"]["raw_output"] = raw_output
    file_payload["cleaning"]["reasoning_content"] = ""
    file_payload["cleaning"]["parsed"] = synthetic
    file_payload["cleaning"]["keep"] = False
    file_payload["cleaning"]["core_target"] = ""
    file_payload["cleaning"]["reason"] = reason
    file_payload["cleaning"]["finish_reason"] = "rule_filtered"
    file_payload["cleaning"]["attempts"] = [
        {
            "stage": "cleaning",
            "attempt": 0,
            "timestamp": utc_now(),
            "rule_filtered": rule_name,
            "error": None,
        }
    ]
    file_payload["cleaning"]["turns"] = [
        {
            "stage": "cleaning",
            "prompt": prompt,
            "raw_output": raw_output,
            "reasoning_content": "",
            "finish_reason": "rule_filtered",
            "parsed": synthetic,
            "normalized": synthetic,
            "attempts": file_payload["cleaning"]["attempts"],
        }
    ]
    file_payload["session_messages"] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": raw_output},
    ]
    file_payload["processing"]["status"] = "filtered_out"
    file_payload["processing"]["filter_stage"] = "cleaning"
    file_payload["processing"]["rule_filtered"] = rule_name
    file_payload["processing"]["completed_at"] = utc_now()
    return file_payload


def run_single_attempt(job: SampleJob, args: argparse.Namespace, output_path: Path, request_logger: RequestLogger) -> Dict[str, Any]:
    original_task = str(job.record.get("query", "")).strip()
    started_at = utc_now()
    sample_id = str(job.record.get("id", f"{job.source}:{job.source_rank}"))
    classification_allowed_categories = get_allowed_categories_for_job(job, args)
    request_api_url = chat_api_url(args.endpoint)

    file_payload: Dict[str, Any] = {
        "processing": {
            "status": "started",
            "workflow": "cleaning->multipoint->three_type_classification->rewrite_same_session_local",
            "started_at": started_at,
            "completed_at": None,
            "source": job.source,
            "source_rank": job.source_rank,
            "global_rank": job.global_rank,
            "source_file": job.source_file,
            "output_path": str(output_path),
            "model": args.model,
            "api_url": request_api_url,
        },
        "record": job.record,
        "cleaning": {
            "prompt": build_cleaning_prompt(original_task),
            "raw_output": None,
            "reasoning_content": None,
            "parsed": None,
            "keep": None,
            "core_target": None,
            "reason": None,
            "finish_reason": None,
            "attempts": [],
            "turns": [],
        },
        "multipoint": {
            "prompt": None,
            "raw_output": None,
            "reasoning_content": None,
            "parsed": None,
            "multi_point": None,
            "reason": None,
            "finish_reason": None,
            "attempts": [],
            "turns": [],
        },
        "classification": {
            "input_task": None,
            "prompt": None,
            "raw_output": None,
            "reasoning_content": None,
            "parsed": None,
            "thought": None,
            "category": None,
            "allowed_categories": classification_allowed_categories,
            "is_valid_category": False,
            "finish_reason": None,
            "attempts": [],
            "turns": [],
        },
        "rewrite": {
            "input_task": None,
            "prompt": None,
            "raw_output": None,
            "reasoning_content": None,
            "parsed": None,
            "rewritten_query": None,
            "finish_reason": None,
            "attempts": [],
            "turns": [],
        },
        "session_messages": [],
    }

    rule_filter = get_rule_filter_reason(job, original_task, counting_enabled=bool(getattr(args, 'counting_enabled', False)))
    if rule_filter is not None:
        rule_name, rule_reason = rule_filter
        return apply_rule_filter(file_payload, rule_name, rule_reason)

    session = ChatSession(
        api_url=request_api_url,
        api_key=args.api_key,
        model=args.model,
        sample_id=sample_id,
        source=job.source,
        request_logger=request_logger,
        timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        max_retries=args.max_retries,
    )

    cleaning_turn, cleaning_parsed, cleaning_data, cleaning_attempts, cleaning_turns, cleaning_error = run_json_stage(
        session=session,
        initial_prompt=file_payload["cleaning"]["prompt"],
        stage="cleaning",
        parse_retries=args.parse_retries,
        parse_output=parse_cleaning_output,
        retry_prompt_builder=lambda previous_output: build_cleaning_retry_prompt(original_task, previous_output),
    )
    file_payload["cleaning"]["raw_output"] = cleaning_turn["assistant_content"] if cleaning_turn else None
    file_payload["cleaning"]["reasoning_content"] = cleaning_turn["reasoning_content"] if cleaning_turn else None
    file_payload["cleaning"]["parsed"] = cleaning_parsed
    file_payload["cleaning"]["finish_reason"] = cleaning_turn["finish_reason"] if cleaning_turn else None
    file_payload["cleaning"]["attempts"] = cleaning_attempts
    file_payload["cleaning"]["turns"] = cleaning_turns
    file_payload["session_messages"] = list(session.messages)
    if cleaning_data is not None:
        file_payload["cleaning"]["keep"] = cleaning_data["keep"]
        file_payload["cleaning"]["core_target"] = cleaning_data["core_target"]
        file_payload["cleaning"]["reason"] = cleaning_data["reason"]

    if cleaning_turn is None:
        file_payload["processing"]["status"] = "cleaning_request_failed"
        file_payload["processing"]["error"] = cleaning_error
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if cleaning_data is None:
        file_payload["processing"]["status"] = "cleaning_parse_failed"
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if not cleaning_data["keep"]:
        file_payload["processing"]["status"] = "filtered_out"
        file_payload["processing"]["filter_stage"] = "cleaning"
        file_payload["processing"]["completed_at"] = utc_now()
        file_payload["session_messages"] = list(session.messages)
        return file_payload

    core_target = cleaning_data["core_target"]
    multipoint_prompt = build_multipoint_prompt(original_task, core_target)
    file_payload["multipoint"]["prompt"] = multipoint_prompt

    multipoint_turn, multipoint_parsed, multipoint_data, multipoint_attempts, multipoint_turns, multipoint_error = run_json_stage(
        session=session,
        initial_prompt=multipoint_prompt,
        stage="multipoint",
        parse_retries=args.parse_retries,
        parse_output=parse_multipoint_output,
        retry_prompt_builder=lambda previous_output: build_multipoint_retry_prompt(original_task, core_target, previous_output),
    )
    file_payload["multipoint"]["raw_output"] = multipoint_turn["assistant_content"] if multipoint_turn else None
    file_payload["multipoint"]["reasoning_content"] = multipoint_turn["reasoning_content"] if multipoint_turn else None
    file_payload["multipoint"]["parsed"] = multipoint_parsed
    file_payload["multipoint"]["finish_reason"] = multipoint_turn["finish_reason"] if multipoint_turn else None
    file_payload["multipoint"]["attempts"] = multipoint_attempts
    file_payload["multipoint"]["turns"] = multipoint_turns
    file_payload["session_messages"] = list(session.messages)
    if multipoint_data is not None:
        file_payload["multipoint"]["multi_point"] = multipoint_data["multi_point"]
        file_payload["multipoint"]["reason"] = multipoint_data["reason"]

    if multipoint_turn is None:
        file_payload["processing"]["status"] = "multipoint_request_failed"
        file_payload["processing"]["error"] = multipoint_error
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if multipoint_data is None:
        file_payload["processing"]["status"] = "multipoint_parse_failed"
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if multipoint_data["multi_point"] and not getattr(args, 'counting_enabled', False):
        file_payload["processing"]["status"] = "filtered_out"
        file_payload["processing"]["filter_stage"] = "multipoint"
        file_payload["processing"]["completed_at"] = utc_now()
        file_payload["processing"]["rule_filtered"] = "multipoint_true"
        file_payload["session_messages"] = list(session.messages)
        return file_payload
    if multipoint_data["multi_point"] and getattr(args, 'counting_enabled', False):
        file_payload["processing"]["counting_enabled"] = True
        # Force-classify all multi-point data as Counting when counting is enabled.
        # The Qwen3-8B model is only trained for 3-type classification;
        # multi-point data is definitionally Counting.
        category = "Counting"
        file_payload["classification"]["category"] = "Counting"
        file_payload["classification"]["is_valid_category"] = True
        file_payload["classification"]["input_task"] = original_task
        file_payload["classification"]["input_core_target"] = core_target
        file_payload["processing"]["classification_route"] = "forced_counting_from_multipoint"
        # Skip LLM classification — proceed directly to rewrite
        # (classification_prompt and run_json_stage are skipped below)

    if file_payload["classification"].get("category") is None:
        # Normal path: run LLM classification
        category = "None"
        classification_prompt = build_classification_prompt(original_task, core_target, classification_allowed_categories)
        file_payload["classification"]["input_task"] = original_task
        file_payload["classification"]["input_core_target"] = core_target
        file_payload["classification"]["prompt"] = classification_prompt
        file_payload["processing"]["classification_route"] = "classify_three_types_only"

        classification_turn, classification_parsed, classification_data, classification_attempts, classification_turns, classification_error = run_json_stage(
        session=session,
        initial_prompt=classification_prompt,
        stage="classification",
        parse_retries=args.parse_retries,
        parse_output=lambda raw_text, parsed: parse_classification_output(
            raw_text,
            parsed,
            classification_allowed_categories,
            original_task=original_task,
            core_target=core_target,
        ),
        retry_prompt_builder=lambda previous_output: build_classification_retry_prompt(
            original_task,
            core_target,
            previous_output,
            classification_allowed_categories,
        ),
    )
    file_payload["classification"]["raw_output"] = classification_turn["assistant_content"] if classification_turn else None
    file_payload["classification"]["reasoning_content"] = classification_turn["reasoning_content"] if classification_turn else None
    file_payload["classification"]["parsed"] = classification_parsed
    file_payload["classification"]["finish_reason"] = classification_turn["finish_reason"] if classification_turn else None
    file_payload["classification"]["attempts"] = classification_attempts
    file_payload["classification"]["turns"] = classification_turns
    file_payload["session_messages"] = list(session.messages)
    if classification_data is not None:
        file_payload["classification"]["thought"] = classification_data["thought"]
        file_payload["classification"]["raw_category"] = classification_data["raw_category"]
        file_payload["classification"]["category"] = classification_data["category"]
        file_payload["classification"]["category_refined"] = classification_data["category_refined"]
        file_payload["classification"]["allowed_categories"] = classification_data["allowed_categories"]
        file_payload["classification"]["is_valid_category"] = classification_data["is_valid_category"]
        category = classification_data["category"]

    if classification_turn is None:
        file_payload["processing"]["status"] = "classification_request_failed"
        file_payload["processing"]["error"] = classification_error
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if classification_data is None:
        file_payload["processing"]["status"] = "classification_parse_failed"
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if category not in classification_allowed_categories:
        file_payload["processing"]["status"] = "filtered_out"
        file_payload["processing"]["filter_stage"] = "classification"
        file_payload["processing"]["completed_at"] = utc_now()
        file_payload["session_messages"] = list(session.messages)
        return file_payload

    rewrite_input_text = core_target
    file_payload["rewrite"]["input_task"] = rewrite_input_text
    rewrite_prompt = build_rewrite_prompt(category, rewrite_input_text)
    file_payload["rewrite"]["prompt"] = rewrite_prompt

    rewrite_attempts: List[Dict[str, Any]] = []
    rewrite_turns: List[Dict[str, Any]] = []
    best_turn: Optional[Dict[str, Any]] = None
    best_parsed: Optional[Dict[str, Any]] = None
    best_query: Optional[str] = None
    rewrite_error: Optional[str] = None
    current_prompt = build_rewrite_json_fallback_prompt(category, rewrite_input_text)
    current_response_format: Optional[Dict[str, Any]] = {"type": "json_object"}

    for rewrite_try in range(1, args.rewrite_retries + 1):
        stage_name = "rewrite" if rewrite_try == 1 else f"rewrite_repair_{rewrite_try - 1}"
        try:
            rewrite_turn = session.send_message(
                current_prompt,
                stage=stage_name,
                response_format=current_response_format,
                extra_body=NO_THINKING_EXTRA_BODY,
            )
        except Exception as exc:  # noqa: BLE001
            rewrite_attempts.append(
                {
                    "stage": stage_name,
                    "attempt": "session_failed",
                    "timestamp": utc_now(),
                    "error": str(exc),
                }
            )
            rewrite_error = str(exc)
            if rewrite_try < args.rewrite_retries:
                current_prompt = build_rewrite_json_fallback_prompt(category, rewrite_input_text)
                current_response_format = {"type": "json_object"}
                continue
            break

        rewrite_parsed = extract_json_object(rewrite_turn["assistant_content"])
        rewrite_data = parse_rewrite_output(rewrite_turn["assistant_content"])
        rewrite_invalid_reason: Optional[str] = None
        if rewrite_data is not None and not rewrite_preserves_category_cues(
            category,
            rewrite_input_text,
            rewrite_data["rewritten_query"],
        ):
            rewrite_invalid_reason = "Dropped existing/current point cue required for Object Reference."
            rewrite_data = None
        rewrite_attempts.extend(rewrite_turn["attempts"])
        rewrite_turns.append(
            {
                "stage": stage_name,
                "prompt": current_prompt,
                "raw_output": rewrite_turn["assistant_content"],
                "reasoning_content": rewrite_turn["reasoning_content"],
                "finish_reason": rewrite_turn["finish_reason"],
                "parsed": rewrite_parsed,
                "normalized": rewrite_data,
                "invalid_reason": rewrite_invalid_reason,
                "attempts": rewrite_turn["attempts"],
            }
        )
        best_turn = rewrite_turn
        best_parsed = rewrite_parsed
        best_query = rewrite_data["rewritten_query"] if rewrite_data is not None else normalize_rewritten_text(rewrite_turn["assistant_content"])
        file_payload["session_messages"] = list(session.messages)

        if rewrite_turn["finish_reason"] != "length" and rewrite_data is not None:
            break

        if rewrite_try < args.rewrite_retries:
            previous_output = rewrite_turn["assistant_content"]
            if rewrite_invalid_reason:
                previous_output = f"{previous_output}\n\nWhy invalid: {rewrite_invalid_reason}"
            current_prompt = build_rewrite_json_fallback_prompt(category, rewrite_input_text, previous_output)
            current_response_format = {"type": "json_object"}


    file_payload["rewrite"]["raw_output"] = best_turn["assistant_content"] if best_turn else None
    file_payload["rewrite"]["reasoning_content"] = best_turn["reasoning_content"] if best_turn else None
    file_payload["rewrite"]["parsed"] = best_parsed
    file_payload["rewrite"]["finish_reason"] = best_turn["finish_reason"] if best_turn else None
    file_payload["rewrite"]["attempts"] = rewrite_attempts
    file_payload["rewrite"]["turns"] = rewrite_turns
    file_payload["rewrite"]["rewritten_query"] = best_query

    if best_turn is None:
        file_payload["processing"]["status"] = "rewrite_failed"
        file_payload["processing"]["error"] = rewrite_error
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    if not is_valid_rewritten_query(best_query) or best_turn["finish_reason"] == "length":
        file_payload["processing"]["status"] = "rewrite_failed"
        file_payload["processing"]["completed_at"] = utc_now()
        return file_payload

    file_payload["processing"]["status"] = "completed"
    file_payload["processing"]["completed_at"] = utc_now()
    return file_payload


def process_job(job: SampleJob, args: argparse.Namespace, output_root: Path, request_logger: RequestLogger) -> Dict[str, Any]:
    output_path = output_path_for_job(output_root, job)
    if output_path.exists() and not args.overwrite:
        existing, existing_reasons = load_existing_sample(output_path)
        if existing is not None and not existing_reasons:
            return {"status": "skipped", "source": job.source, "output_path": str(output_path)}

    last_payload: Optional[Dict[str, Any]] = None
    last_reasons: List[str] = []
    for sample_attempt in range(1, args.sample_retries + 1):
        attempt_payload = run_single_attempt(job, args, output_path, request_logger)
        attempt_payload.setdefault("processing", {})
        attempt_payload["processing"]["sample_attempt"] = sample_attempt
        reasons = validate_sample_payload(attempt_payload)
        attempt_payload["processing"]["validation_errors"] = reasons
        last_payload = attempt_payload
        last_reasons = reasons

        if not reasons:
            atomic_write_json(output_path, attempt_payload)
            return {"status": attempt_payload["processing"]["status"], "source": job.source, "output_path": str(output_path)}

    assert last_payload is not None
    last_payload.setdefault("processing", {})
    last_payload["processing"]["status"] = "invalid_sample_exhausted"
    last_payload["processing"]["validation_errors"] = last_reasons
    atomic_write_json(output_path, last_payload)
    return {"status": last_payload["processing"]["status"], "source": job.source, "output_path": str(output_path)}


def write_run_config(output_root: Path, args: argparse.Namespace, jobs: List[SampleJob]) -> None:
    config = {
        "generated_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "current_eval_file": str((POINTARENA_EXTRACT_DIR / "evaluate_pointarena.py").resolve()),
        "model": args.model,
        "api_url": chat_api_url(args.endpoint),
        "limit_per_source": args.limit_per_source,
        "exclude_sources": sorted(parse_csv_sources(args.exclude_sources)),
        "workers": args.workers,
        "timeout": args.timeout,
        "connect_timeout": args.connect_timeout,
        "max_retries": args.max_retries,
        "parse_retries": args.parse_retries,
        "sample_retries": args.sample_retries,
        "rewrite_retries": args.rewrite_retries,
        "workflow": "cleaning->multipoint->three_type_classification->rewrite_same_session_local",
        "valid_categories": get_allowed_categories_for_args(args),
        "classification_prompt_from_current_eval": False,
        "cleaning_prompt_from_user_override": True,
        "multipoint_prompt_enabled": True,
        "rewrite_prompt_from_original_category_templates": False,
        "all_stages_disable_thinking": True,
        "sources": sorted({job.source for job in jobs}),
        "source_files": sorted({job.source_file for job in jobs}),
    }
    atomic_write_json(output_root / "run_config.json", config)


def write_summary(output_root: Path, summary_file: str, args: argparse.Namespace, jobs: List[SampleJob]) -> None:
    by_status: Dict[str, int] = {}
    by_source_status: Dict[str, Dict[str, int]] = {}
    total_output_files = 0

    for sample_file in output_root.glob("*/*.json"):
        total_output_files += 1
        sample, validation_reasons = load_existing_sample(sample_file)
        if sample is None:
            status = "invalid_file"
            source = sample_file.parent.name
        elif validation_reasons:
            status = "invalid_file"
            source = sample.get("processing", {}).get("source", sample_file.parent.name)
        else:
            status = sample.get("processing", {}).get("status", "unknown")
            source = sample.get("processing", {}).get("source", sample_file.parent.name)
        by_status[status] = by_status.get(status, 0) + 1
        by_source_status.setdefault(source, {})
        by_source_status[source][status] = by_source_status[source].get(status, 0) + 1

    summary = {
        "generated_at": utc_now(),
        "model": args.model,
        "api_url": chat_api_url(args.endpoint),
        "requested_jobs": len(jobs),
        "total_output_files": total_output_files,
        "results_by_status": by_status,
        "results_by_source_status": by_source_status,
    }
    atomic_write_json(output_root / summary_file, summary)


def main() -> None:
    args = parse_args()
    args.endpoint = normalize_local_endpoint(args.endpoint)

    output_root = SCRIPT_DIR / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    request_logger = RequestLogger(output_root / args.requests_log)

    print("[Step 1] Selecting the first samples from each by-source file...")
    excluded_sources = parse_csv_sources(args.exclude_sources)
    jobs = select_jobs(args.limit_per_source, exclude_sources=excluded_sources)
    write_run_config(output_root, args, jobs)
    print(f"[Step 1] Prepared {len(jobs)} jobs from {len(sorted({job.source for job in jobs}))} sources.")

    print("[Step 2] Running a shared-session cleaning + multipoint + three-type classification + rewrite pipeline...")
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(process_job, job, args, output_root, request_logger): job
            for job in jobs
        }
        with tqdm(total=len(future_to_job), desc="mix4 clean-mp-cls-rw", ncols=100) as pbar:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "worker_failed", "source": job.source, "error": str(exc)}
                results.append(result)
                pbar.update(1)

    print("[Step 3] Writing run summary...")
    write_summary(output_root, args.summary_file, args, jobs)

    status_counts: Dict[str, int] = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"[Done] Output root: {output_root}")
    print(f"[Done] Status counts: {json.dumps(status_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
