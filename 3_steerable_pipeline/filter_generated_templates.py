#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GROUP_REQUIRED_PLACEHOLDERS = {
    "straight_axis_templates": {"{label}"},
    "single_diagonal_templates": {"{label}"},
    "two_step_axis_templates": {"{label}", "{first_axis}", "{second_axis}"},
    "nearest_among_objects_templates": {"{label}"},
}

GROUP_OPTIONAL_DIRECTION_PLACEHOLDERS = {
    "straight_axis_templates": {"{axis_direction}", "{axis_phrase}"},
    "single_diagonal_templates": {"{diagonal_direction}", "{diagonal_phrase}"},
}

GLOBAL_BANNED_PATTERNS = {
    "blue_point_as_mark": re.compile(r"\b(starting mark|reference mark|blue reference mark|mark the|mark\b)\b", re.I),
    "ui_specific": re.compile(r"\b(cursor|click|tap|hover|drag|slide)\b", re.I),
    "navigation_style": re.compile(r"\b(navigate|proceed|target the|trace a path|follow the path)\b", re.I),
    "guide_wording": re.compile(r"\bguide\b", re.I),
    "unnatural_modifier": re.compile(r"\b(exactly|directly at|physically closest|residing|you can find)\b", re.I),
}

GROUP_BANNED_PATTERNS = {
    "straight_axis_templates": {
        "diagonal_wording": re.compile(r"\b(diagonal|diagonally|up-right|up-left|down-right|down-left)\b", re.I),
        "nearest_wording": re.compile(r"\b(nearest|closest|nearby?)\b", re.I),
    },
    "single_diagonal_templates": {
        "two_step_wording": re.compile(r"\bthen\b|\band then\b", re.I),
    },
    "two_step_axis_templates": {
        "collapsed_diagonal": re.compile(r"\b(diagonal|diagonally|up-right|up-left|down-right|down-left)\b", re.I),
    },
    "nearest_among_objects_templates": {
        "missing_nearest_word": re.compile(r"^(?!.*\b(nearest|closest)\b).*$", re.I),
        "vague_near": re.compile(r"\b(near the blue point|nearby|around the blue point)\b", re.I),
    },
}


def validate_template(group: str, template: str) -> list[str]:
    reasons: list[str] = []

    for placeholder in GROUP_REQUIRED_PLACEHOLDERS.get(group, set()):
        if placeholder not in template:
            reasons.append(f"missing_required_placeholder:{placeholder}")

    optional_placeholders = GROUP_OPTIONAL_DIRECTION_PLACEHOLDERS.get(group)
    if optional_placeholders and not any(token in template for token in optional_placeholders):
        reasons.append("missing_direction_placeholder")

    for reason, pattern in GLOBAL_BANNED_PATTERNS.items():
        if pattern.search(template):
            reasons.append(reason)

    for reason, pattern in GROUP_BANNED_PATTERNS.get(group, {}).items():
        if pattern.search(template):
            reasons.append(reason)

    if group == "nearest_among_objects_templates" and "{label}" in template:
        if "blue point" not in template and "blue reference point" not in template:
            reasons.append("missing_blue_point_reference")

    if group == "two_step_axis_templates":
        if "then" not in template.lower():
            reasons.append("missing_then_step_order")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cleaned: dict[str, list[str]] = {}
    report_lines = []

    for group, templates in payload.items():
        if not isinstance(templates, list):
            report_lines.append(f"[skip] {group}: not a list")
            continue
        kept = []
        rejected = []
        seen = set()
        for idx, template in enumerate(templates):
            text = str(template).strip()
            if not text:
                rejected.append((idx, text, ["empty"]))
                continue
            if text in seen:
                rejected.append((idx, text, ["duplicate"]))
                continue
            seen.add(text)
            reasons = validate_template(group, text)
            if reasons:
                rejected.append((idx, text, reasons))
            else:
                kept.append(text)
        cleaned[group] = kept
        report_lines.append(f"[group] {group}: kept={len(kept)} rejected={len(rejected)}")
        for idx, text, reasons in rejected:
            report_lines.append(f"  - reject#{idx + 1}: {', '.join(reasons)} :: {text}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
