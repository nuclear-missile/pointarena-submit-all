"""Prompt templates for the Gemini 4-stage data processing pipeline.

All prompts are extracted from the original process_mix4.py and
evaluate_pointarena.py files for modularity and reuse.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Cleaning stage
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

# ---------------------------------------------------------------------------
# Multipoint detection stage
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

# ---------------------------------------------------------------------------
# Classification stage
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT_TEMPLATE = """# Role and Objective
You are a highly precise Data Classifier for the Vision-Language Navigation dataset (Point Arena). Your task is to analyze an object-pointing instruction based on its underlying *semantic features and cognitive bottleneck*, classifying it into exactly ONE of FIVE predefined categories.

# CRITICAL FORMATTING RULES
You must output ONLY a valid, flat JSON object.
- NO Markdown formatting (DO NOT use ```json or ```).
- NO conversational filler.
- Output exactly two keys: "thought" (under 20 words analyzing the semantic feature) and "category".

Example output:
{
  "thought": "Focuses on reading text/semantic info (hotel name) to pinpoint an exact UI/text element.",
  "category": "Object Reference"
}

# Semantic Category Definitions & Decision Boundaries
Do not rely on keywords. Evaluate the PRIMARY mental effort required to solve the task.

## 1. Object Reference (UI, Text, and Fine-Grained Pinpointing)
- **Semantic Feature:** The task acts like a precision crosshair. It targets purely digital interfaces, exact text strings, OR ultra-fine-grained extreme points of an object. It also includes relative navigation from a pre-existing cursor.
- **Key Characteristics:**
  - Reading text/labels (e.g., "the name of the hotel", "the input bar").
  - Extreme fine-grained specific points (e.g., "the tip of the shoes", "directly onto the blue lens").
  - Cursor-based relative movement (e.g., "existing point", "moving downward until...").

## 2. Counting (Enumeration and Collections)
- **Semantic Feature:** The task requires scanning the global scene to identify a *collection*, a mass of items, or a specific subset based on exact quantity.
- **Key Characteristics:**
  - Naked plural nouns acting as the main target without strict spatial anchors (e.g., "the statues", "Jeep vehicles").
  - Mass nouns describing a grouped state (e.g., "the sliced fruit").
  - Explicit counting ("two yellow hats", "all the birds").
- *Exception:* If the plural objects are strictly located by a geometric anchor (e.g., "objects *above the sofa*"), the primary task is spatial, so it belongs to Spatial Relation.

## 3. Spatial Relation (Geometric Layout and Distance)
- **Semantic Feature:** The primary cognitive bottleneck is 2D mapping, absolute/relative positioning, or distance comparison. It requires no complex physics or functional deduction.
- **Key Characteristics:**
  - Distance comparisons (e.g., "the nearest boat", "closest table").
  - Topological anchoring (e.g., "above the sofa", "between X and Y", "front wheels of").
  - *Exception:* If the target relies on a part-whole ownership/semantic binding (e.g., "the shoe of the cyclist"), it requires semantic reasoning, NOT pure spatial layout.

## 4. Affordance (Generic Human Tools and Instruments)
- **Semantic Feature:** Extremely strict definition. It applies ONLY to generic, manipulable tools, instruments, or standard devices defined purely by human hand-scale operations (eating, writing, hammering, playing music).
- **Key Characteristics:**
  - Generic objects named by their utility (e.g., "tool that creates music", "tool made of wood", "object used to eat").
  - *Boundary Rule:* If the object is a structural sub-component of a vehicle/room (e.g., "illuminate the car", "carry the baby", "where items are placed on a bike"), it requires scene-level deduction and belongs to Reasoning!

## 5. Reasoning (World Knowledge, States, and Complex Deduction)
- **Semantic Feature:** The catch-all for complex semantics. The target is isolated through physical states, part-whole ownership, causal events, or scene-level structural functionality.
- **Key Characteristics:**
  - Action/Physical states (e.g., "moving", "fastest", "direction in which...").
  - Scene-level functional deduction (e.g., "where people sit", "what illuminates the car", "object used to carry the baby").
  - Part-whole ownership linking specific actors (e.g., "the shoe of the cyclist", "part of the train").
  - Indirect logic/World knowledge (e.g., "caused it to fall", "color of grass").
---
# Input Task:
[INSERT_TASK_HERE]"""

CLASSIFICATION_PROMPT_NO_COUNTING_TEMPLATE = """# Role and Objective
You are a highly precise Data Classifier for the Vision-Language Navigation dataset (Point Arena). Your task is to analyze an object-pointing instruction based on its underlying semantic features and cognitive bottleneck, classifying it into exactly ONE of FOUR predefined categories. If the task is fundamentally a Counting / collection-enumeration task, output category "None" instead.

# CRITICAL FORMATTING RULES
You must output ONLY a valid, flat JSON object.
- NO Markdown formatting (DO NOT use ```json or ```).
- NO conversational filler.
- Output exactly two keys: "thought" (under 20 words analyzing the semantic feature) and "category".

Example output:
{
  "thought": "Focuses on reading text/semantic info to pinpoint an exact UI/text element.",
  "category": "Object Reference"
}

# Valid Category Values For This Run
- Reasoning
- Spatial Relation
- Affordance
- Object Reference
- None

# Special Rule For This Run
There is NO Counting category in this run.
- If the task is fundamentally about multiple instances, all items in a set, exact counts, grouped targets, plural collection scanning, or anything that would normally belong to Counting, output "None".
- Do NOT force a counting-like task into another category.

# Semantic Category Definitions & Decision Boundaries
Do not rely on keywords. Evaluate the PRIMARY mental effort required to solve the task.

## 1. Object Reference (UI, Text, and Fine-Grained Pinpointing)
- Semantic Feature: The task acts like a precision crosshair. It targets purely digital interfaces, exact text strings, or ultra-fine-grained extreme points of an object. It also includes relative navigation from a pre-existing cursor.
- Key Characteristics:
  - Reading text/labels.
  - Extreme fine-grained specific points.
  - Cursor-based relative movement.

## 2. Spatial Relation (Geometric Layout and Distance)
- Semantic Feature: The primary cognitive bottleneck is 2D mapping, absolute/relative positioning, or distance comparison. It requires no complex physics or functional deduction.
- Key Characteristics:
  - Distance comparisons.
  - Topological anchoring such as above, below, between, left of, right of, closest, or farthest.
  - Exception: If the target relies on part-whole ownership or semantic binding, it belongs to Reasoning.

## 3. Affordance (Generic Human Tools and Instruments)
- Semantic Feature: Extremely strict definition. It applies ONLY to generic, manipulable tools, instruments, or standard devices defined purely by human hand-scale operations.
- Key Characteristics:
  - Generic objects named by their utility.
  - Exception: If the object is a structural sub-component of a vehicle or room, it belongs to Reasoning.

## 4. Reasoning (World Knowledge, States, and Complex Deduction)
- Semantic Feature: The catch-all for complex semantics. The target is isolated through physical states, part-whole ownership, causal events, or scene-level structural functionality.
- Key Characteristics:
  - Action or physical states.
  - Scene-level functional deduction.
  - Part-whole ownership linking specific actors.
  - Indirect logic or world knowledge.

# Input Task:
[INSERT_TASK_HERE]"""

# ---------------------------------------------------------------------------
# Rewrite stage
# ---------------------------------------------------------------------------

REWRITE_PROMPTS: Dict[str, str] = {
    "Reasoning": """You have already classified the original instruction as Reasoning.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Reasoning category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
- Keep the cognitive bottleneck on state, causality, world knowledge, physical property, ownership, or scene-level reasoning.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
    "Spatial Relation": """You have already classified the original instruction as Spatial Relation.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Spatial Relation category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
- Preserve the geometric anchor(s), relative position, topological relation, or nearest/farthest style cue.
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
- Keep the focus on what a person can do with the object, or the object's human-use function.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
    "Counting": """You have already classified the original instruction as Counting.

Rewrite the original instruction into PointArena test-query style.

Requirements:
- Preserve the same target semantics and keep the task in the Counting category.
- The rewritten task must be a single natural English sentence.
- It must start with "Point to".
- Preserve plurality, "all", grouped targets, or exact numbers exactly when they matter.
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
- Preserve any cue about an existing point, cursor, icon, text/UI element, or very fine-grained pinpointing.
- Do not mention the category, dataset name, JSON, coordinates, masks, points, or explanations.
- Output only the rewritten question text.

Original instruction:
[INSERT_ORIGINAL_TASK_HERE]""",
}

# ---------------------------------------------------------------------------
# Rewrite JSON fallback rules (used when JSON repair is needed)
# ---------------------------------------------------------------------------

REWRITE_JSON_FALLBACK_RULES: Dict[str, str] = {
    "Reasoning": "Keep the cognitive bottleneck on state, causality, world knowledge, physical property, ownership, or scene-level reasoning.",
    "Spatial Relation": "Preserve the geometric anchor(s), relative position, topological relation, or nearest/farthest style cue.",
    "Affordance": "Keep the focus on what a person can do with the object, or the object's human-use function.",
    "Counting": "Preserve plurality, \"all\", grouped targets, or exact numbers exactly when they matter.",
    "Object Reference": "Preserve any cue about an existing point, cursor, icon, text/UI element, or very fine-grained pinpointing.",
}

# ---------------------------------------------------------------------------
# Extra body configuration to disable thinking on supported models
# ---------------------------------------------------------------------------

NO_THINKING_EXTRA_BODY: Dict[str, Any] = {
    "google": {
        "thinking_config": {
            "include_thoughts": False,
            "thinkingBudget": 0,
        }
    }
}
