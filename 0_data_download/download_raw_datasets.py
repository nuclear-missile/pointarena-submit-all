#!/usr/bin/env python3
"""
Download all 4 raw training datasets for PointArena.

Sources:
  1. allenai/pixmo-points       (1.85M rows) - PixMo pointing data
  2. wentao-yuan/robopoint-data  (667K rows)  - RoboPoint data
  3. JingkunAn/RefSpatial        (1.8K rows)  - RefSpatial data
  4. FlagEval/Where2Place        (100 rows)   - Where2Place data

Usage:
  python download_raw_datasets.py --output_dir ./raw_data
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "pixmo_points": {
        "repo": "allenai/pixmo-points",
        "config": "default",
        "rows": "1.85M",
        "description": "PixMo pointing data - primary source",
    },
    "robopoint": {
        "repo": "wentao-yuan/robopoint-data",
        "config": "default",
        "rows": "667K",
        "description": "RoboPoint robotics pointing data",
    },
    "refspatial": {
        "repo": "JingkunAn/RefSpatial",
        "config": "default",
        "rows": "1.8K",
        "description": "Reference spatial expressions",
    },
    "where2place": {
        "repo": "FlagEval/Where2Place",
        "config": "default",
        "rows": "100",
        "description": "Where-to-place spatial reasoning",
    },
}


def download_hf_dataset(repo: str, output_dir: Path, config: str = "default"):
    """Download a HuggingFace dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo} ...")
    cmd = [
        sys.executable, "-c", f"""
from datasets import load_dataset
ds = load_dataset("{repo}", split="train")
print(f"  Downloaded {{len(ds)}} rows")
ds.to_parquet("{output_dir / 'data.parquet'}")
print(f"  Saved to {output_dir / 'data.parquet'}")
"""
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download PointArena training datasets")
    parser.add_argument("--output_dir", default="./raw_data",
                        help="Output directory for raw datasets")
    parser.add_argument("--datasets", nargs="*",
                        choices=list(DATASETS.keys()),
                        help="Specific datasets to download (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = args.datasets if args.datasets else list(DATASETS.keys())

    print("=" * 60)
    print("PointArena Raw Dataset Downloader")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")
    print(f"Datasets: {', '.join(targets)}")
    print()

    for name in targets:
        info = DATASETS[name]
        ds_dir = output_dir / name
        print(f"\n[{name}] {info['description']}")
        print(f"  Source: {info['repo']} ({info['rows']} rows)")

        try:
            download_hf_dataset(info['repo'], ds_dir, info['config'])
            print(f"  [OK] Downloaded to {ds_dir}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            print(f"  Manual: https://huggingface.co/datasets/{info['repo']}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nNext step: Process with 1_gemini_pipeline or 2_qwen_pipeline")


if __name__ == "__main__":
    main()
