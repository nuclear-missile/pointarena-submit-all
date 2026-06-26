#!/usr/bin/env python3
"""
Download PointArena official evaluation data.

Source: https://huggingface.co/datasets/PointArena/pointarena-data

Downloads:
  - val_pointarena.jsonl (982 evaluation samples)
  - images/ (5 categories: affordance, counting, reasoning, spatial, steerable)
  - masks/ (5 categories)

Usage:
  python download_pointarena_eval.py --output_dir ./eval_data
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CATEGORIES = ["affordance", "counting", "reasoning", "spatial", "steerable"]


def download_from_hf(repo: str, output_dir: Path):
    """Clone/download from HuggingFace dataset repo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo} ...")

    # Use huggingface_hub to download
    cmd = [
        sys.executable, "-c", f"""
from huggingface_hub import snapshot_download
path = snapshot_download(
    "{repo}",
    repo_type="dataset",
    local_dir="{output_dir.as_posix()}",
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: {{path}}")
"""
    ]
    subprocess.run(cmd, check=True)


def verify_download(data_dir: Path):
    """Verify that all expected files exist."""
    issues = []

    val_file = data_dir / "val_pointarena.jsonl"
    if not val_file.exists():
        issues.append(f"Missing: {val_file}")
    else:
        with open(val_file) as f:
            lines = f.readlines()
        print(f"  val_pointarena.jsonl: {len(lines)} samples")

    for cat in CATEGORIES:
        img_dir = data_dir / "images" / cat
        mask_dir = data_dir / "masks" / cat

        if not img_dir.exists():
            issues.append(f"Missing image dir: {img_dir}")
        else:
            n_imgs = len(list(img_dir.glob("*")))
            print(f"  images/{cat}: {n_imgs} files")

        if not mask_dir.exists():
            issues.append(f"Missing mask dir: {mask_dir}")
        else:
            n_masks = len(list(mask_dir.glob("*")))
            print(f"  masks/{cat}: {n_masks} files")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Download PointArena eval data")
    parser.add_argument("--output_dir", default="./eval_data",
                        help="Output directory for eval data")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not args.verify_only:
        print("=" * 60)
        print("PointArena Evaluation Data Downloader")
        print("=" * 60)
        print(f"Output: {output_dir.absolute()}")
        print()

        try:
            download_from_hf("PointArena/pointarena-data", output_dir)
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print()
            print("Manual download:")
            print("  1. Visit: https://huggingface.co/datasets/PointArena/pointarena-data")
            print("  2. Download all files")
            print(f"  3. Place in: {output_dir.absolute()}")
            print()
            print("Expected structure:")
            print(f"  {output_dir}/val_pointarena.jsonl")
            print(f"  {output_dir}/images/affordance/")
            print(f"  {output_dir}/images/counting/")
            print(f"  {output_dir}/images/reasoning/")
            print(f"  {output_dir}/images/spatial/")
            print(f"  {output_dir}/images/steerable/")
            print(f"  {output_dir}/masks/affordance/")
            print(f"  {output_dir}/masks/counting/")
            print(f"  {output_dir}/masks/reasoning/")
            print(f"  {output_dir}/masks/spatial/")
            print(f"  {output_dir}/masks/steerable/")

    print("\nVerifying download...")
    issues = verify_download(output_dir)

    if issues:
        print(f"\n[WARN] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] All files present!")


if __name__ == "__main__":
    main()
