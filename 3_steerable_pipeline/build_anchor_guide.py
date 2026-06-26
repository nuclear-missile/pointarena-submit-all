#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDE4_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClassifiedQuestion:
    ordinal: int
    data_index: int
    image_filename: str
    question: str
    primary_style: str
    flags: tuple[str, ...]
    is_anchor_like: bool


ANCHOR_RE = re.compile(
    r"\b("
    r"blue point|existing point|current point|previous point|designated point|"
    r"the point|current location|existing location|current position|previous location|"
    r"previous points?|location in the existing image"
    r")\b",
    re.I,
)
IMPLICIT_MOVE_RE = re.compile(
    r"\b("
    r"moving|move the point|move the current point|move point|slightly move|"
    r"move upward|move down|move left|move right|"
    r"(?:up|down|left|right|upward|downward|upwards|downwards)"
    r"(?:\s+and\s+(?:left|right|up|down)|"
    r"\s*,\s*then\s*(?:left|right|up|down|upward|upwards|downward|downwards)|"
    r"\s+then\s*(?:left|right|up|down|upward|upwards|downward|downwards))?"
    r"\s+until"
    r")\b",
    re.I,
)
NEAREST_RE = re.compile(r"\b(nearest|closest|closer)\b", re.I)
TWO_STEP_RE = re.compile(r"\bthen\b|\band then\b", re.I)
DIAGONAL_RE = re.compile(
    r"\b("
    r"up\s*(?:and|&)\s*(?:left|right)|"
    r"down\s*(?:and|&)\s*(?:left|right)|"
    r"up[- ]right|up[- ]left|down[- ]right|down[- ]left|"
    r"upper[- ]right|upper[- ]left|lower[- ]right|lower[- ]left|"
    r"diagonal|diagonally"
    r")\b",
    re.I,
)
AXIS_RE = re.compile(
    r"\b("
    r"to the right of|to the left of|right of|left of|above|below|under|beneath|"
    r"at the top of|at the bottom of|on the left side|on the right side|"
    r"directly below|directly above|on the left of|on the right of|"
    r"moving (?:slightly )?(?:left|right|up|down|upward|downward|upwards|downwards)|"
    r"move (?:the point )?(?:left|right|up|down|upward|downward|upwards|downwards)|"
    r"move (?:a little |slightly )?to the (?:left|right)|"
    r"moving (?:a little |slightly )?to the (?:left|right)|"
    r"moving left|moving right|moving up|moving down|go (?:left|right|up|down)"
    r")\b",
    re.I,
)
SAME_POSITION_RE = re.compile(
    r"\b("
    r"where the blue point is|where the existing point is|where the blue point is located|"
    r"currently directed|current point is on|existing point indicates|existing point refers|"
    r"existing point is directed|existing point directs|the one the existing point indicates|"
    r"matches the one|base of the existing point|at the bottom of the existing point|"
    r"at the top of the existing point"
    r")\b",
    re.I,
)


def default_data_json() -> Path:
    candidates = [
        REPO_ROOT / "dataset" / "pointarena-data" / "data.json",
        REPO_ROOT / "datasets" / "pointarena-data" / "data.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def classify_question(ordinal: int, data_index: int, row: dict[str, Any]) -> ClassifiedQuestion:
    question = str(row["user_input"])
    flags: list[str] = []
    if ANCHOR_RE.search(question):
        flags.append("explicit_anchor")
    if IMPLICIT_MOVE_RE.search(question):
        flags.append("implicit_move_anchor")
    if NEAREST_RE.search(question):
        flags.append("nearest")
    if TWO_STEP_RE.search(question):
        flags.append("two_step")
    if DIAGONAL_RE.search(question) and not TWO_STEP_RE.search(question):
        flags.append("single_diagonal")
    if AXIS_RE.search(question):
        flags.append("straight_axis")
    if SAME_POSITION_RE.search(question):
        flags.append("same_position_or_existing_object")

    is_anchor_like = bool(
        set(flags)
        & {
            "explicit_anchor",
            "implicit_move_anchor",
            "same_position_or_existing_object",
        }
    )
    if TWO_STEP_RE.search(question):
        primary_style = "two_step_axis_move"
    elif DIAGONAL_RE.search(question):
        primary_style = "single_diagonal_move"
    elif NEAREST_RE.search(question):
        primary_style = "nearest_or_closest"
    elif AXIS_RE.search(question):
        primary_style = "straight_axis_relation_or_move"
    elif SAME_POSITION_RE.search(question):
        primary_style = "same_position_or_existing_object"
    elif is_anchor_like:
        primary_style = "anchor_other"
    else:
        primary_style = "not_anchor_or_unclear"

    return ClassifiedQuestion(
        ordinal=ordinal,
        data_index=data_index,
        image_filename=str(row["image_filename"]),
        question=question,
        primary_style=primary_style,
        flags=tuple(flags),
        is_anchor_like=is_anchor_like,
    )


def load_steerable(data_json: Path) -> list[ClassifiedQuestion]:
    data = json.loads(data_json.read_text(encoding="utf-8"))
    rows: list[ClassifiedQuestion] = []
    ordinal = 0
    for data_index, row in enumerate(data):
        if row.get("category") != "steerable":
            continue
        rows.append(classify_question(ordinal, data_index, row))
        ordinal += 1
    return rows


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def rows_to_table(rows: list[ClassifiedQuestion], include_flags: bool = True) -> list[str]:
    if include_flags:
        lines = ["| # | data_idx | primary_style | flags | question | image |", "| ---: | ---: | --- | --- | --- | --- |"]
    else:
        lines = ["| # | data_idx | primary_style | question | image |", "| ---: | ---: | --- | --- | --- |"]
    for row in rows:
        if include_flags:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.ordinal),
                        str(row.data_index),
                        row.primary_style,
                        ", ".join(row.flags) if row.flags else "-",
                        md_escape(row.question),
                        md_escape(row.image_filename),
                    ]
                )
                + " |"
            )
        else:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.ordinal),
                        str(row.data_index),
                        row.primary_style,
                        md_escape(row.question),
                        md_escape(row.image_filename),
                    ]
                )
                + " |"
            )
    return lines


def examples_for_style(rows: list[ClassifiedQuestion], style: str, limit: int = 8) -> list[str]:
    out = []
    for row in rows:
        if row.primary_style == style:
            out.append(f"- `{md_escape(row.question)}`")
        if len(out) >= limit:
            break
    return out


def build_markdown(data_json: Path, rows: list[ClassifiedQuestion]) -> str:
    anchor_rows = [row for row in rows if row.is_anchor_like]
    non_anchor_rows = [row for row in rows if not row.is_anchor_like]
    primary_counts = Counter(row.primary_style for row in rows)
    flag_counts = Counter(flag for row in rows for flag in row.flags)
    data_json_rel = data_json.relative_to(REPO_ROOT) if data_json.is_relative_to(REPO_ROOT) else data_json

    lines: list[str] = [
        "# PointArena Steerable Anchor Question Template Guide",
        "",
        "This file is generated from the official PointArena steerable test split, not from guide4 smoke outputs.",
        "",
        "Source:",
        "",
        "```text",
        str(data_json_rel),
        "category == steerable",
        "```",
        "",
        "Local note: this checkout stores the dataset under `datasets/pointarena-data`. If another checkout uses `dataset/pointarena-data`, pass that path to the generator with `--data-json`.",
        "",
        "Generated by:",
        "",
        "```bash",
        "python make_steerable1/guide4/scripts/build_pointarena_anchor_guide.py",
        "```",
        "",
        "## Counts",
        "",
        f"- Total official steerable rows: {len(rows)}",
        f"- Anchor-like rows detected by wording: {len(anchor_rows)}",
        f"- Non-anchor or unclear rows: {len(non_anchor_rows)}",
        "",
        "| primary_style | rows |",
        "| --- | ---: |",
    ]
    for style, count in sorted(primary_counts.items()):
        lines.append(f"| {style} | {count} |")
    lines.extend(["", "| detected_flag | rows |", "| --- | ---: |"])
    for flag, count in sorted(flag_counts.items()):
        lines.append(f"| {flag} | {count} |")

    lines.extend(
        [
            "",
            "## Classification Rules",
            "",
            "- `straight_axis_relation_or_move`: wording such as right, left, above, below, under, or one-axis movement.",
            "- `single_diagonal_move`: one diagonal relation or movement, such as up-left or down and left, without a `then` turn.",
            "- `two_step_axis_move`: ordered movement with `then`, such as up then right. This matches the current guide4 `move_then_axis_limited_fixed` geometry: two connected axis-aligned segments.",
            "- `nearest_or_closest`: nearest, closest, or closer wording. This maps to `nearest_among_objects` and must use a programmatic nearest-distance margin.",
            "- `same_position_or_existing_object`: the target is where the point is, on the current point, or an object indicated by the existing point. Guide4 is not currently generating this as a task.",
            "- `anchor_other`: anchor-dependent but not one of the strict generated guide4 tasks.",
            "",
            "The classifier is intentionally conservative for template design. A row can have multiple flags, but the `primary_style` column chooses the style most relevant to guide4 generation.",
            "",
            "## Official Anchor-Like Questions",
            "",
        ]
    )
    lines.extend(rows_to_table(anchor_rows, include_flags=True))

    lines.extend(["", "## Non-Anchor Or Unclear Steerable Rows", ""])
    lines.extend(rows_to_table(non_anchor_rows, include_flags=False))

    lines.extend(
        [
            "",
            "## Official Wording Examples By Style",
            "",
            "### Straight Axis",
            "",
        ]
    )
    lines.extend(examples_for_style(rows, "straight_axis_relation_or_move", limit=12))
    lines.extend(["", "### Single Diagonal", ""])
    diagonal_examples = examples_for_style(rows, "single_diagonal_move", limit=12)
    lines.extend(diagonal_examples if diagonal_examples else ["- No single diagonal rows detected."])
    lines.extend(["", "### Two-Step Axis", ""])
    two_step_examples = examples_for_style(rows, "two_step_axis_move", limit=12)
    lines.extend(two_step_examples if two_step_examples else ["- No two-step rows detected."])
    lines.extend(["", "### Nearest Or Closest", ""])
    lines.extend(examples_for_style(rows, "nearest_or_closest", limit=12))

    lines.extend(
        [
            "",
            "## Guide4 Move Template Requirements",
            "",
            "### 1. Straight-Line Axis Move",
            "",
            "Use this when the generated path is one straight axis-aligned segment from the blue point to the target.",
            "",
            "Allowed direction placeholders:",
            "",
            "```text",
            "{axis_direction}: right | left | up | down",
            "{axis_phrase}: to the right of | to the left of | above | below | under",
            "{label}: exact target label from the data",
            "```",
            "",
            "Geometry requirement:",
            "",
            "- The target must lie on the requested side of the blue point.",
            "- The generated anchor should be sampled by polar angle within +/-10 degrees of the intended anchor-to-target direction.",
            "- There must be no non-target object mask crossing the 3-pixel path band between anchor and target.",
            "- The blue point is only a reference or start point. It must not be described as marking an object.",
            "",
            "Safe template shapes:",
            "",
            "```text",
            "Point to the {label} {axis_phrase} the blue point.",
            "From the blue point, move {axis_direction} until you reach the {label}.",
            "Start at the blue point and go {axis_direction} to the {label}.",
            "Use the blue point only as a reference, then select the {label} {axis_phrase} it.",
            "```",
            "",
            "Do not use vague phrases such as near, around, somewhere, or in that area. Do not say the blue point labels, indicates, points to, or is on another object.",
            "",
            "### 2. Single Diagonal-Line Move",
            "",
            "Use this when the generated path is one straight diagonal segment from the blue point to the target.",
            "",
            "Allowed direction placeholders:",
            "",
            "```text",
            "{diagonal_direction}: up-right | down-right | down-left | up-left",
            "{diagonal_phrase}: up and to the right of | down and to the right of | down and to the left of | up and to the left of",
            "{label}: exact target label from the data",
            "```",
            "",
            "Geometry requirement:",
            "",
            "- The text must describe a single diagonal move, not two ordered turns.",
            "- The anchor should be sampled within +/-10 degrees of the intended diagonal anchor-to-target direction.",
            "- The 3-pixel path band must not cross any non-target object mask.",
            "- The blue point is only a reference or start point.",
            "",
            "Safe template shapes:",
            "",
            "```text",
            "Point to the {label} {diagonal_phrase} the blue point.",
            "Move diagonally {diagonal_direction} from the blue point until you reach the {label}.",
            "Start at the blue point and keep going {diagonal_direction} to the {label}.",
            "Use the blue point as the start; follow the {diagonal_direction} diagonal to the {label}.",
            "```",
            "",
            "Do not write `first right, then up`; that belongs to the two-step task. Do not reverse the relation by saying where the blue point is relative to the target.",
            "",
            "### 3. Two-Step Axis Move",
            "",
            "Use this for guide4 `move_then_axis_limited_fixed`. The overall target relation is diagonal, but the path is two connected axis-aligned segments.",
            "",
            "Important implementation note:",
            "",
            "- Current guide4 code uses `{first_axis}` followed by `{second_axis}`.",
            "- This is not a single diagonal line.",
            "- If we later need a true first-straight-then-diagonal path, that should be a new validator and a new template family, because the current path check rasterizes two axis-aligned line segments.",
            "",
            "Allowed placeholders:",
            "",
            "```text",
            "{first_axis}: right | left | up | down",
            "{second_axis}: right | left | up | down",
            "{diagonal_relation}: up-right | down-right | down-left | up-left",
            "{label}: exact target label from the data",
            "```",
            "",
            "Geometry requirement:",
            "",
            "- Move from the blue point along `{first_axis}`.",
            "- Turn once, then move along `{second_axis}` to reach the target.",
            "- The order must be preserved exactly.",
            "- The union of the two 3-pixel path bands must not cross any non-target object mask.",
            "- The blue point is only the starting reference.",
            "",
            "Safe template shapes:",
            "",
            "```text",
            "From the blue point, move {first_axis}, then {second_axis}, to the {label}.",
            "Start at the blue point: go {first_axis} first, then {second_axis}, and select the {label}.",
            "Point to the {label} reached by going {first_axis} from the blue point and then {second_axis}.",
            "Use the blue point only as the start; move {first_axis}, turn {second_axis}, and point to the {label}.",
            "```",
            "",
            "Do not collapse this into `up-right of the blue point`. Do not use `diagonally` unless the wording still explicitly states the two ordered steps.",
            "",
            "### 4. Nearest Among Objects",
            "",
            "Use this only when there are 2 to 4 same-label objects and the target is programmatically nearest to the blue point by distance with a margin.",
            "",
            "Safe template shapes:",
            "",
            "```text",
            "Point to the {label} nearest to the blue point.",
            "Using the blue point only as a reference, select the closest {label}.",
            "Find the {label} that is closest to the blue point.",
            "```",
            "",
            "Do not use nearest wording if the blue point is inside any same-label mask. Do not ask for a generic nearby object.",
            "",
            "## Prompt For Generating More Templates",
            "",
            "Use this prompt with a large language model to produce more reusable templates. It is aligned with the official PointArena steerable wording above and with the current guide4 validators.",
            "",
            "```text",
            "You are generating reusable English templates for visual pointing instructions in a steerable pointing dataset.",
            "",
            "Core context:",
            "- The image may contain a blue point.",
            "- The blue point is only a reference/start point.",
            "- The blue point must never be described as marking, indicating, labeling, touching, or being on an object.",
            "- The final answer expected from the user/model is a point on the target object named {label}.",
            "- Return templates with placeholders, not filled examples.",
            "- Do not mention red points, masks, coordinates, pixels, SAM, segmentation, validation, or paths being checked.",
            "- Do not ask QA-style questions such as Where is or Which direction.",
            "- Do not ask for counting, descriptions, explanations, comparisons, or object attributes.",
            "- Do not introduce extra attributes not present in {label}.",
            "- Keep every template concise and imperative.",
            "- Keep the wording close to official PointArena steerable questions.",
            "- Prefer openings like: Point to..., From the blue point, move..., Start at the blue point..., Use the blue point only as a reference....",
            "- Avoid UI-specific or device-specific wording such as cursor, click, tap, hover, drag, slide, or button press actions unless the target label itself is a UI element.",
            "- Avoid overly agentic navigation wording such as navigate, trace a path, follow the path, target the, or proceed to.",
            "- Avoid annotation-like wording such as mark the, starting mark, or blue reference mark.",
            "- Avoid verbose or unnatural modifiers such as exactly, directly at, physically closest, residing, or you can find.",
            "",
            "Use these official PointArena steerable examples as style references only:",
            "- Point to the red strip under the blue point.",
            "- Point to the building to the right of the existing point.",
            "- Point to moving up and left until it is on the heart.",
            "- Point to moving up, then right until you are on the leg of the spider.",
            "- Point to the nearest black horse next to the blue point.",
            "",
            "Generate four template groups:",
            "",
            "1. straight_axis_templates",
            "Geometry: one straight axis-aligned segment from the blue point to the target.",
            "Placeholders: {label}, plus either {axis_direction} or {axis_phrase}.",
            "{axis_direction} is one of right, left, up, down.",
            "{axis_phrase} is one of to the right of, to the left of, above, below, under.",
            "Rules: do not use diagonal wording; do not use nearest/closest wording.",
            "Style: prefer Point to / From the blue point, move / Start at the blue point. Avoid click/cursor/navigate/mark/trace/slide.",
            "",
            "2. single_diagonal_templates",
            "Geometry: one straight diagonal segment from the blue point to the target.",
            "Placeholders: {label}, plus either {diagonal_direction} or {diagonal_phrase}.",
            "{diagonal_direction} is one of up-right, down-right, down-left, up-left.",
            "{diagonal_phrase} is one of up and to the right of, down and to the right of, down and to the left of, up and to the left of.",
            "Rules: describe a single diagonal move; do not split into first/then steps.",
            "Style: prefer Point to / Move diagonally / Start at the blue point. Avoid click/cursor/navigate/mark/trace/slide.",
            "",
            "3. two_step_axis_templates",
            "Geometry: two ordered axis-aligned segments. First move along {first_axis}; then turn and move along {second_axis}.",
            "Placeholders: {label}, {first_axis}, {second_axis}.",
            "{first_axis} and {second_axis} are each one of right, left, up, down.",
            "Rules: preserve the order exactly; do not collapse the path into a single diagonal phrase.",
            "Style: prefer From the blue point, move {first_axis}, then {second_axis}... Avoid navigate/trace/follow path/click/cursor/mark wording.",
            "",
            "4. nearest_among_objects_templates",
            "Geometry: choose the same-label object closest to the blue point.",
            "Placeholders: {label}.",
            "Rules: use nearest or closest explicitly; say the blue point is only a reference when possible.",
            "Style: keep nearest/closest as the exact disambiguating relation. Do not add vague near or nearby wording such as near the blue point or around the blue point.",
            "",
            "Output JSON only with exactly these keys:",
            "{",
            "  \"straight_axis_templates\": [\"...\"],",
            "  \"single_diagonal_templates\": [\"...\"],",
            "  \"two_step_axis_templates\": [\"...\"],",
            "  \"nearest_among_objects_templates\": [\"...\"]",
            "}",
            "",
            "Generate 30 templates for each group.",
            "Every template must be unique and reusable.",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", type=Path, default=default_data_json())
    parser.add_argument("--output", type=Path, default=GUIDE4_ROOT / "anchor_question_template_guide.md")
    args = parser.parse_args()

    rows = load_steerable(args.data_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(args.data_json, rows), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"steerable={len(rows)} anchor_like={sum(row.is_anchor_like for row in rows)}")


if __name__ == "__main__":
    main()
