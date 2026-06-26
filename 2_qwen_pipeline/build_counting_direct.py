#!/usr/bin/env python3
"""
Direct counting data builder — no LLM pipeline needed.

Reads pixmo_points.jsonl from rank 8192 (forward frontier),
selects multi-point records, and saves as Counting training samples.

The key insight: multi-point data IS Counting by definition.
We don't need Qwen3-8B classification to tell us that.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_FILE = PROJECT_ROOT / "ans8" / "data" / "cleaned" / "by_source" / "pixmo_points.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "counting_20k_forward" / "pixmo_points"
START_RANK = 8192  # continue from mix4_counting_to_10000_continuation frontier
TARGET_COUNT = 20000
BATCH_LOG_INTERVAL = 500


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = 0
    scanned = 0
    skipped_no_points = 0
    skipped_single_point = 0
    stats = Counter()

    print(f"Source: {SOURCE_FILE}")
    print(f"Start rank: {START_RANK}")
    print(f"Target: {TARGET_COUNT} Counting samples")
    print(f"Output: {OUTPUT_DIR}")
    print()

    with open(SOURCE_FILE) as f:
        for line_idx, line in enumerate(f, start=1):
            if line_idx <= START_RANK:
                continue

            scanned += 1
            record = json.loads(line)
            points_abs = record.get("points_abs", [])
            query = str(record.get("query", "")).strip()
            record_id = record.get("id", f"pixmo_{line_idx}")
            image_path = record.get("image_path", "")

            if not isinstance(points_abs, list) or len(points_abs) == 0:
                skipped_no_points += 1
                continue

            if len(points_abs) <= 1:
                skipped_single_point += 1
                continue

            # Multi-point → Counting
            width = record.get("width", 0)
            height = record.get("height", 0)

            # Build a training sample
            sample = {
                "processing": {
                    "status": "completed",
                    "workflow": "direct_counting_selection",
                    "source": "pixmo_points",
                    "source_rank": line_idx,
                    "completed_at": "2026-05-08T00:00:00Z",
                },
                "record": {
                    "id": record_id,
                    "query": query,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "points_abs": [(float(p[0]), float(p[1])) for p in points_abs],
                },
                "classification": {
                    "category": "Counting",
                    "is_valid_category": True,
                    "allowed_categories": ["Counting"],
                },
                "rewrite": {
                    "rewritten_query": query,
                },
            }

            output_path = OUTPUT_DIR / f"{line_idx:04d}__{record_id}.json"
            output_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2))

            selected += 1
            stats[f"selected"] += 1

            if selected % BATCH_LOG_INTERVAL == 0:
                print(f"  [{selected}/{TARGET_COUNT}] scanned={scanned} "
                      f"skipped(no_pts={skipped_no_points}, single={skipped_single_point})")

            if selected >= TARGET_COUNT:
                break

    print()
    print(f"Done! Selected {selected} Counting samples")
    print(f"Scanned: {scanned} records from rank {START_RANK+1} to {START_RANK+scanned}")
    print(f"Skipped: {skipped_no_points} no points, {skipped_single_point} single point")
    print(f"Multi-point rate: {selected}/{scanned} = {selected/max(1,scanned)*100:.1f}%")

    # Save summary
    summary = {
        "source": str(SOURCE_FILE),
        "start_rank": START_RANK,
        "end_rank": START_RANK + scanned,
        "target_count": TARGET_COUNT,
        "selected": selected,
        "scanned": scanned,
        "skipped_no_points": skipped_no_points,
        "skipped_single_point": skipped_single_point,
    }
    (OUTPUT_DIR.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
