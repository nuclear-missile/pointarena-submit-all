#!/usr/bin/env python3
"""Build image-level exclusion keys from prior cleaning outputs and mix10000_v2 work items."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_3types_local.image_key_utils import (  # noqa: E402
    normalized_image_key_from_sample,
    normalized_image_key_from_value,
)

POINTARENA_EXTRACT_DIR = PROJECT_ROOT / "pointarena_extract"
STEERABLE_OUTPUTS_DIR = PROJECT_ROOT / "make_steerable1" / "sam_clean" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exclusion image keys for local three-type cleaning.")
    parser.add_argument("--output-json", required=True, help="Where to write the sorted exclusion key list.")
    parser.add_argument("--report-json", default="", help="Optional report path for counts and provenance.")
    return parser.parse_args()


def discover_pointarena_roots() -> list[Path]:
    if not POINTARENA_EXTRACT_DIR.exists():
        return []
    roots: list[Path] = []
    for child in sorted(POINTARENA_EXTRACT_DIR.iterdir()):
        if child.is_dir():
            roots.append(child)
    return roots


def discover_mix10000_v2_jsons() -> list[Path]:
    patterns = [
        "_work_mix10000_v2/batch_*/image_ids.json",
        "_smoke_mix10000_v2/image_ids.json",
        "mix10000_v2/image_ids.json",
    ]
    discovered: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(STEERABLE_OUTPUTS_DIR.glob(pattern)):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(path)
    return discovered


def collect_pointarena_keys(roots: list[Path]) -> tuple[set[str], dict[str, int]]:
    keys: set[str] = set()
    counts: dict[str, int] = {}
    for root in roots:
        root_count = 0
        for sample_file in root.glob("*/*.json"):
            try:
                sample = json.loads(sample_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if sample.get("processing", {}).get("status") != "completed":
                continue
            key = normalized_image_key_from_sample(sample)
            if not key:
                continue
            if key not in keys:
                root_count += 1
            keys.add(key)
        counts[root.name] = root_count
    return keys, counts


def collect_json_list_keys(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    keys: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        path_count = 0
        for item in payload:
            key = normalized_image_key_from_value(item)
            if not key:
                continue
            if key not in keys:
                path_count += 1
            keys.add(key)
        counts[str(path.relative_to(PROJECT_ROOT))] = path_count
    return keys, counts


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    report_json = Path(args.report_json).resolve() if args.report_json else None

    pointarena_roots = discover_pointarena_roots()
    mix_jsons = discover_mix10000_v2_jsons()

    pointarena_keys, pointarena_counts = collect_pointarena_keys(pointarena_roots)
    mix_keys, mix_counts = collect_json_list_keys(mix_jsons)
    all_keys = sorted(pointarena_keys | mix_keys)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_keys, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_json": str(output_json),
        "total_exclusion_keys": len(all_keys),
        "pointarena_extract_root_count": len(pointarena_roots),
        "pointarena_extract_counts": pointarena_counts,
        "pointarena_extract_unique_keys": len(pointarena_keys),
        "mix10000_v2_json_count": len(mix_jsons),
        "mix10000_v2_counts": mix_counts,
        "mix10000_v2_unique_keys": len(mix_keys),
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
