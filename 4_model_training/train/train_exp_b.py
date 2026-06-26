#!/usr/bin/env python3
"""
Experiment B: Coordinate Map as Additional Image + Correction Prompt

Launches training with IterPosCollatorB which:
- Builds coordinate Gaussian map images from deviated points
- Adds them as additional images in the input
- Uses correction prompts (same text format as Experiment A)

This script monkey-patches the collator in the baseline training script,
then calls the baseline's main(). All training logic is reused.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


if __name__ == "__main__":
    # Parse our additional args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_pos_noise_sigma", type=float, default=100.0)
    parser.add_argument("--iter_pos_drop_prob", type=float, default=0.0)
    our_args, remaining = parser.parse_known_args()

    print("=" * 60)
    print("Experiment B: Coord Map Image + Text Correction")
    print(f"Noise sigma: {our_args.iter_pos_noise_sigma}")
    print(f"Drop prob: {our_args.iter_pos_drop_prob}")
    print("=" * 60)

    # Monkey-patch: replace Molmo2PointCollator with IterPosCollatorB
    from filtered_data_ft.plan_iter_pos.collator_iter_pos_B import IterPosCollatorB
    import src.train.collator as collator_module

    OriginalCollator = collator_module.Molmo2PointCollator

    class PatchedCollator(IterPosCollatorB):
        def __init__(self, processor=None, context_len=2048, coordinate_scale="1000",
                     target_text_variant="sample_label", **kwargs):
            super().__init__(
                processor=processor,
                context_len=context_len,
                coordinate_scale=coordinate_scale,
                target_text_variant=target_text_variant,
                noise_sigma=our_args.iter_pos_noise_sigma,
                drop_prob=our_args.iter_pos_drop_prob,
                **kwargs,
            )

    collator_module.Molmo2PointCollator = PatchedCollator

    # Call baseline main with remaining args
    sys.argv = [sys.argv[0]] + remaining
    import filtered_data_ft.run_lora_train_filtered_steerable_d as baseline
    baseline.main()
