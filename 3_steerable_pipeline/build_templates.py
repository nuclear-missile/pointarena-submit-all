#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "template_library_clean.json"
TARGET_COUNT = 100

USER_EXTRA_STRAIGHT_AXIS = [
    "Show me the {label} {axis_phrase} the blue point.",
    "Place your cursor on the {label} {axis_phrase} the blue point.",
    "Shift your focus {axis_direction} from the blue point and click the {label}.",
    "Treating the blue point as an origin, select the {label} {axis_phrase} it.",
    "Relying on the blue point for direction, move {axis_direction} to the {label}.",
    "Direct your attention {axis_direction} from the blue point to the {label}.",
    "Highlight the {label} {axis_phrase} the blue reference point.",
    "Spot the {label} located {axis_phrase} the blue point and select it.",
    "Go straight {axis_direction} from the blue point to find the {label}.",
    "Let the blue point guide you {axis_direction} to the {label}.",
    "Pinpoint the {label} {axis_phrase} the blue point.",
    "Move in a straight line {axis_direction} from the blue point to the {label}.",
    "Tap the {label} {axis_phrase} the blue point.",
    "Seek out the {label} {axis_phrase} the blue reference point.",
    "Travel {axis_direction} starting from the blue point to select the {label}.",
    "Spot the blue point first, then select the {label} {axis_phrase} it.",
    "Journey {axis_direction} from the blue point until you hit the {label}.",
    "Move straight {axis_direction} from the blue point and pick the {label}.",
    "Focus on the {label} {axis_phrase} the blue point.",
    "Use the blue point as a landmark to find the {label} {axis_phrase} it.",
    "Point directly to the {label} {axis_phrase} the blue point.",
    "Hover over the {label} {axis_phrase} the blue point.",
    "Select the {label} that sits {axis_phrase} the blue point.",
    "Aim for the {label} {axis_phrase} the blue point.",
    "Track {axis_direction} from the blue point to reach the {label}.",
    "Keep moving {axis_direction} from the blue reference point to the {label}.",
    "Touch the {label} positioned {axis_phrase} the blue point.",
    "Indicate the {label} {axis_phrase} the blue point.",
    "Looking {axis_phrase} the blue point, select the {label}.",
    "Starting at the blue point, go straight {axis_direction} and choose the {label}.",
]


def load_library(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(item).strip() for item in value if str(item).strip()] for key, value in payload.items()}


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def top_up(seed: list[str], generated: list[str], target: int = TARGET_COUNT) -> list[str]:
    merged = dedupe(seed + generated)
    if len(merged) < target:
        raise ValueError(f"only produced {len(merged)} templates, need {target}")
    return merged[:target]


def gen_straight_axis() -> list[str]:
    openings = [
        "Use the blue point as your reference and",
        "Take the blue point as the guide and",
        "Starting from the blue point,",
        "Beginning at the blue point,",
        "With the blue point as the origin,",
        "Keeping the blue point fixed in mind,",
        "Relative to the blue point,",
        "From the position of the blue point,",
        "Using the blue point only as orientation,",
        "Let the blue point serve as your guide and",
    ]
    verbs = [
        "select the {label} {axis_phrase} it.",
        "move {axis_direction} until you reach the {label}.",
        "find the {label} {axis_phrase} it.",
        "choose the {label} {axis_phrase} it.",
        "point to the {label} {axis_phrase} it.",
        "go {axis_direction} to the {label}.",
        "look {axis_phrase} it and select the {label}.",
        "head {axis_direction} to the {label}.",
        "identify the {label} {axis_phrase} it.",
        "travel {axis_direction} to the {label}.",
    ]
    return [f"{opening} {verb}" for opening in openings for verb in verbs]


def gen_single_diagonal() -> list[str]:
    openings = [
        "Using the blue point as the reference,",
        "Starting from the blue point,",
        "From the blue point,",
        "Keep the blue point in view and",
        "Treat the blue point as your anchor,",
        "With the blue point as the guide,",
        "Using the blue point only for direction,",
        "Relative to the blue point,",
        "Take the blue point as the origin and",
        "Beginning at the blue point,",
    ]
    verbs = [
        "move diagonally {diagonal_direction} until you reach the {label}.",
        "select the {label} {diagonal_phrase} it.",
        "find the {label} {diagonal_phrase} it.",
        "go {diagonal_direction} to the {label}.",
        "travel {diagonal_direction} to the {label}.",
        "head {diagonal_direction} until you get to the {label}.",
        "locate the {label} {diagonal_phrase} it.",
        "follow the {diagonal_direction} diagonal to the {label}.",
        "identify the {label} {diagonal_phrase} it.",
        "choose the {label} {diagonal_phrase} it.",
    ]
    return [f"{opening} {verb}" for opening in openings for verb in verbs]


def gen_two_step() -> list[str]:
    openings = [
        "From the blue point,",
        "Starting at the blue point,",
        "Use the blue point as your starting reference and",
        "Take the blue point as the start and",
        "Beginning with the blue point,",
        "With the blue point as your guide,",
        "Using the blue point as the origin,",
        "Treat the blue point as the starting mark and",
        "Keeping the blue point fixed as the start,",
        "From the blue point as the start,",
    ]
    bodies = [
        "move {first_axis}, then {second_axis}, to the {label}.",
        "go {first_axis} first, then {second_axis}, and select the {label}.",
        "travel {first_axis} and then {second_axis} until you reach the {label}.",
        "head {first_axis} before turning {second_axis} to the {label}.",
        "first move {first_axis}, then continue {second_axis} to the {label}.",
        "take one step {first_axis} and then one step {second_axis} to the {label}.",
        "follow {first_axis} first, then {second_axis}, until you arrive at the {label}.",
        "shift {first_axis}, then shift {second_axis}, to the {label}.",
        "move {first_axis} and after that move {second_axis} to the {label}.",
        "reach the {label} by going {first_axis} first and {second_axis} second.",
    ]
    return [f"{opening} {body}" for opening in openings for body in bodies]


def gen_nearest() -> list[str]:
    openings = [
        "Using the blue point as reference,",
        "Relative to the blue point,",
        "From the blue point,",
        "Treat the blue point as your guide and",
        "With the blue point fixed as reference,",
        "Keeping the blue point as the anchor,",
        "Using the blue point only for distance,",
        "Take the blue point as the comparison point and",
        "Starting from the blue point as reference,",
        "With the blue point as the origin for distance,",
    ]
    bodies = [
        "select the nearest {label}.",
        "choose the closest {label}.",
        "find the {label} closest to it.",
        "identify the nearest {label}.",
        "point to the closest {label}.",
        "locate the {label} nearest to it.",
        "pick the {label} with the shortest distance to it.",
        "show the nearest {label}.",
        "mark the closest {label}.",
        "indicate the {label} nearest to it.",
    ]
    return [f"{opening} {body}" for opening in openings for body in bodies]


def gen_other_axis() -> list[str]:
    openings = [
        "Using the {anchor_label} as reference,",
        "Starting from the {anchor_label},",
        "From the {anchor_label},",
        "Treat the {anchor_label} as your guide and",
        "With the {anchor_label} as the origin,",
        "Keeping the {anchor_label} fixed in mind,",
        "Relative to the {anchor_label},",
        "Using the {anchor_label} only for direction,",
        "Take the {anchor_label} as the anchor and",
        "Beginning at the {anchor_label},",
    ]
    bodies = [
        "select the {label} {axis_phrase} it.",
        "move {axis_direction} until you reach the {label}.",
        "find the {label} {axis_phrase} it.",
        "choose the {label} {axis_phrase} it.",
        "point to the {label} {axis_phrase} it.",
        "go {axis_direction} to the {label}.",
        "look {axis_phrase} it and select the {label}.",
        "head {axis_direction} to the {label}.",
        "identify the {label} {axis_phrase} it.",
        "travel {axis_direction} to the {label}.",
    ]
    return [f"{opening} {body}" for opening in openings for body in bodies]


def gen_other_diagonal() -> list[str]:
    openings = [
        "Using the {anchor_label} as reference,",
        "Starting from the {anchor_label},",
        "From the {anchor_label},",
        "Treat the {anchor_label} as your guide and",
        "With the {anchor_label} as the origin,",
        "Keeping the {anchor_label} fixed in mind,",
        "Relative to the {anchor_label},",
        "Using the {anchor_label} only for direction,",
        "Take the {anchor_label} as the anchor and",
        "Beginning at the {anchor_label},",
    ]
    bodies = [
        "move diagonally {diagonal_direction} until you reach the {label}.",
        "select the {label} {diagonal_phrase} it.",
        "find the {label} {diagonal_phrase} it.",
        "go {diagonal_direction} to the {label}.",
        "travel {diagonal_direction} to the {label}.",
        "head {diagonal_direction} until you get to the {label}.",
        "locate the {label} {diagonal_phrase} it.",
        "follow the {diagonal_direction} diagonal to the {label}.",
        "identify the {label} {diagonal_phrase} it.",
        "choose the {label} {diagonal_phrase} it.",
    ]
    return [f"{opening} {body}" for opening in openings for body in bodies]


def gen_axis_constrained_nearest() -> list[str]:
    openings = [
        "Using the blue point as reference,",
        "Relative to the blue point,",
        "From the blue point,",
        "Treat the blue point as your guide and",
        "With the blue point fixed as reference,",
        "Keeping the blue point as the anchor,",
        "Using the blue point only for distance and direction,",
        "Take the blue point as the comparison point and",
        "Starting from the blue point as reference,",
        "With the blue point as the origin for distance,",
    ]
    bodies = [
        "select the nearest {label} {axis_phrase} it.",
        "choose the closest {label} on the {axis_direction} side of it.",
        "find the {label} nearest to it among those {axis_phrase} it.",
        "identify the nearest {label} {axis_phrase} it.",
        "point to the closest {label} {axis_phrase} it.",
        "locate the {label} nearest to it on the {axis_direction} side.",
        "pick the nearest {label} that lies {axis_phrase} it.",
        "show the closest {label} {axis_phrase} it.",
        "mark the nearest {label} on the {axis_direction} side of it.",
        "indicate the {label} nearest to it among the ones {axis_phrase} it.",
    ]
    return [f"{opening} {body}" for opening in openings for body in bodies]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build steerable template library (7 groups x 100 templates)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path. Default: template_library_raw.json",
    )
    parser.add_argument(
        "--seed-library",
        type=str,
        default=None,
        help="Optional seed library JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path("template_library_raw.json")

    seed_library: dict[str, list[str]] = {}
    if args.seed_library:
        seed_library = load_library(Path(args.seed_library))
    elif LIB_PATH.exists():
        seed_library = load_library(LIB_PATH)

    expansions = {
        "straight_axis_templates": USER_EXTRA_STRAIGHT_AXIS + gen_straight_axis(),
        "single_diagonal_templates": gen_single_diagonal(),
        "two_step_axis_templates": gen_two_step(),
        "nearest_among_objects_templates": gen_nearest(),
        "other_object_axis_templates": gen_other_axis(),
        "other_object_diagonal_templates": gen_other_diagonal(),
        "axis_constrained_nearest_templates": gen_axis_constrained_nearest(),
    }
    out: dict[str, list[str]] = {}
    for key, generated in expansions.items():
        out[key] = top_up(seed_library.get(key, []), generated, TARGET_COUNT)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in out.items()}, ensure_ascii=False, indent=2))
    print(f"Wrote {sum(len(v) for v in out.values())} templates to {output_path}")


if __name__ == "__main__":
    main()