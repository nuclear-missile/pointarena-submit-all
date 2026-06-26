#!/usr/bin/env python3
"""
Reorder val_pointarena.jsonl by expert group for efficient evaluation.

Groups samples by expert (C, B, steer) to minimize model switching.
With do_sample=False, reordering does NOT affect results — inference is deterministic.

Usage:
    python3 reorder_val.py --input data/val_pointarena.jsonl --output data/val_pointarena_grouped.jsonl
"""
import argparse, csv, json, os
from collections import defaultdict

CATEGORY_EXPERT = {
    "affordance": "C", "counting": "C", "reasoning": "B",
    "spatial": "C", "steerability": "steer",
}


def main():
    parser = argparse.ArgumentParser(description="Reorder val JSONL by expert group")
    parser.add_argument("--input", default="data/val_pointarena.jsonl")
    parser.add_argument("--output", default="data/val_pointarena_grouped.jsonl")
    parser.add_argument("--pixmo-csv", default="data/pixmo_metadata.csv")
    args = parser.parse_args()

    # Load steerable index from CSV
    steerable = set()
    if os.path.exists(args.pixmo_csv):
        with open(args.pixmo_csv) as f:
            for row in csv.DictReader(f):
                steerable.add(row["image_filename"])

    # Load samples
    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # Group by expert
    batches = defaultdict(list)
    for row in rows:
        gt = str(row.get("meta", {}).get("category", "")).strip().lower()
        if gt == "affordable": gt = "affordance"
        elif gt == "steerable": gt = "steerability"
        img_fname = os.path.basename(row.get("image_path", ""))
        if steerable and img_fname in steerable:
            gt = "steerability"
        adp = CATEGORY_EXPERT.get(gt, "B")
        batches[adp].append(row)

    # Write grouped: C first, then B, then steer
    with open(args.output, "w") as f:
        count = 0
        for adp in ["C", "B", "steer"]:
            for row in batches[adp]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Input:  {len(rows)} samples (original order)")
    print(f"Output: {count} samples (grouped: C={len(batches['C'])} B={len(batches['B'])} steer={len(batches['steer'])})")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
