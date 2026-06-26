#!/usr/bin/env python3
"""Prompt templates for the Qwen3-8B local data pipeline.

All prompt templates used across the 4-stage pipeline (cleaning, multipoint,
classification, rewrite) are centralized here for easy reference and reuse.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Stage 1: Cleaning
# ---------------------------------------------------------------------------

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


CLEANING_RETRY_PROMPT_TEMPLATE = """Your previous cleaning response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- "keep": boolean
- "core_target": string
- "reason": string

Rules:
- If the instruction is unusable, set keep to false and core_target to an empty string.
- If kept, core_target must be the pure visual target phrase.
- reason must briefly explain the keep/reject decision.
- Do not use markdown.

Input Instruction:
{raw_instruction}

Previous invalid output:
{previous_output}"""


# ---------------------------------------------------------------------------
# Stage 2: Multipoint Detection
# ---------------------------------------------------------------------------

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


MULTIPOINT_RETRY_PROMPT_TEMPLATE = """Your previous multi-point decision response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- "multi_point": boolean
- "reason": string

Original instruction:
{original_instruction}

Cleaned core_target:
{core_target}

Previous invalid output:
{previous_output}"""


# ---------------------------------------------------------------------------
# Stage 3: Classification
# ---------------------------------------------------------------------------

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


CLASSIFICATION_RETRY_PROMPT_TEMPLATE = """Your previous classification response was invalid or truncated. Try again in the same session.

Output ONLY a flat JSON object with exactly these keys:
- "thought": string
- "category": string

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


# ---------------------------------------------------------------------------
# Stage 4: Rewrite
# ---------------------------------------------------------------------------

REWRITE_PROMPTS: Dict[str, str] = {
    "Reasoning": """You have already classified the original instruction as Reasoning.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Reasoning category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
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
- It must start with "Point to".
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
- It must start with "Point to".
- Keep it short, exact, and PointArena-like.
- Preserve any cue about an existing point, cursor, icon, text/UI element, or very fine-grained pinpointing.
- If the original mentions an existing point, current point, existing location, current location, or cursor, the rewrite MUST keep that cue explicitly.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
}


REWRITE_RETRY_PROMPT_TEMPLATE = """Your previous rewrite response was invalid or truncated. Try again in the same session.

Requirements:
- Preserve the same target semantics and keep the task in the {category} category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
- Output only the rewritten question text.

Original instruction:
{rewrite_input_text}

Previous invalid output:
{previous_output}"""


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


REWRITE_JSON_FALLBACK_PROMPT_TEMPLATE = """Your previous rewrite attempt did not produce a usable answer. Stay in the same session and answer again.

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
