#!/usr/bin/env python3
"""
Experiment A: Iterative Coordinate Correction (Text-Only)

Adds deviated point coordinates to the prompt as TEXT NUMBERS.
No model architecture changes. Uses standard Molmo2 LoRA training.

Key difference from baseline:
- On-the-fly perturbation: ground truth points are perturbed with Gaussian noise
- Modified prompt: includes deviated coordinates as text
- Target: original ground truth (model learns to correct)

Usage:
    python3 -m filtered_data_ft.run_lora_train_iter_pos_A \
        --model_path ... \
        --summary_path ... \
        --output_dir ... \
        --noise_sigma 100 \
        [all standard training args]

This script reuses run_lora_train_filtered_steerable_d.py's main() logic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def build_iter_pos_collator_class():
    """Build a collator class that wraps Molmo2PointCollator with correction prompts."""

    from src.train.collator import Molmo2PointCollator

    class IterPosCollatorA(Molmo2PointCollator):
        """
        Wrapper collator that perturbs ground truth points and adds
        deviated coordinates as text in the prompt.

        For each sample:
        - Perturb GT points → deviated points
        - Build correction prompt: original question + deviated coords
        - Target: ground truth points (unchanged)
        """

        def __init__(self, *args, noise_sigma=100.0, drop_prob=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            import numpy as np
            self.noise_sigma = noise_sigma
            self.drop_prob = drop_prob
            self._rng = np.random.RandomState(42)
            self._step_counter = 0

        def _perturb_points(self, points, width, height):
            """Generate deviated version of points."""
            import numpy as np
            # Use deterministic per-step RNG for reproducibility
            step_seed = 42 + self._step_counter
            rng = np.random.RandomState(step_seed)

            deviated = []
            for x, y in points:
                if rng.random() < self.drop_prob:
                    continue
                nx = x + rng.normal(0, self.noise_sigma)
                ny = y + rng.normal(0, self.noise_sigma)
                nx = max(0.0, min(float(width), nx))
                ny = max(0.0, min(float(height), ny))
                deviated.append((round(nx, 1), round(ny, 1)))

            if not deviated:
                deviated = [(float(p[0]), float(p[1])) for p in points[:1]]
            return deviated

        def _build_correction_question(self, original_question, deviated_points):
            """Build a correction prompt that includes deviated coordinates."""
            points_str = ", ".join(
                f"({x:.1f}, {y:.1f})" for x, y in deviated_points
            )
            correction_prompt = (
                f"{original_question}\n\n"
                f"The following point coordinates were predicted but may contain errors: "
                f"[{points_str}]. "
                f"All coordinates are relative to image width/height in [0, 1000] scale. "
                f"Look at the image carefully and output the corrected coordinates."
            )
            return correction_prompt

        def __call__(self, batch):
            # Perturb each sample's points and modify the question
            for sample in batch:
                original_question = sample.get("question", "")
                points = sample.get("points", [])
                width = sample.get("width", 1000)
                height = sample.get("height", 1000)

                deviated = self._perturb_points(points, width, height)

                # Save original for reference
                sample["original_question"] = original_question
                sample["deviated_points"] = deviated
                # Replace question with correction prompt
                sample["question"] = self._build_correction_question(
                    original_question, deviated
                )
                # GT points stay as target (sample["points"] unchanged)
                self._step_counter += 1

            # Use standard collation
            return super().__call__(batch)

    return IterPosCollatorA


def main():
    """Main entry point - mirrors run_lora_train_filtered_steerable_d.main()."""
    # Import the original main after setting up env
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.train.pointarena_rewritten_dataset_steerable_d import (
        PointArenaSteerableDTrainingDataset,
        DEFAULT_STEERABLE_D_SUMMARY_PATH,
        DEFAULT_CLEAN3_ROOTS,
        DEFAULT_SAM_CLEAN_JSONL,
    )
    from src.utils.seed import set_seed
    from src.train.checkpoint_manager import update_leaderboard
    from src.eval.render_report import render_report_markdown

    parser = argparse.ArgumentParser(description="Experiment A - Iterative Coordinate Correction")
    # Model/data paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--summary_path")
    parser.add_argument("--output_base")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clean3_roots_json", default=None)
    parser.add_argument("--sam_clean_jsonl", default=None)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--init_adapter_path", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    # IterPos Experiment A specific
    parser.add_argument("--noise_sigma", type=float, default=100.0,
                        help="Gaussian noise sigma for point perturbation")
    parser.add_argument("--drop_prob", type=float, default=0.0,
                        help="Probability of dropping a point")

    # Training
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", default="cosine_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--keep_top_m", type=int, default=5)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_scope", default="vision+llm")
    parser.add_argument("--lora_include_projector_in_llm_only", action="store_true")
    parser.add_argument("--freeze_lora_branches", action="append", default=[])

    # Precision
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Data
    parser.add_argument("--point_mode", default="all")
    parser.add_argument("--category_balance_json", default=None)
    parser.add_argument("--sampling_strategy", default="shuffle")
    parser.add_argument("--skip_missing_or_invalid_images", action="store_true")
    parser.add_argument("--allow_missing_or_invalid_images", action="store_true")
    parser.add_argument("--exclude_pointarena_eval", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_target_points", type=int, default=64)

    # Eval
    parser.add_argument("--eval_include_counting", action="store_true")
    parser.add_argument("--eval_allow_multi_point", action="store_true")
    parser.add_argument("--no_eval_at_step0", action="store_true")
    parser.add_argument("--fast_eval_subset", type=int, default=0)
    parser.add_argument("--eval_prompt_mode", default="plain")

    # Processor
    parser.add_argument("--max_crops", type=int, default=12)
    parser.add_argument("--high_res_max_crops", type=int, default=24)
    parser.add_argument("--p_high_res", type=float, default=0.7)

    # Misc
    parser.add_argument("--trainable_param_dtype", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_text_variant", default="sample_label")

    args = parser.parse_args()

    # Import baseline training module and reuse its logic
    # We monkey-patch the collator class to use our IterPosCollatorA

    print("=" * 60)
    print("Experiment A: Iterative Coordinate Correction (Text-Only)")
    print(f"Noise sigma: {args.noise_sigma}")
    print(f"Drop prob: {args.drop_prob}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Build the modified collator class
    IterPosCollatorA = build_iter_pos_collator_class()

    # Now import and run the standard training, replacing the collator
    import filtered_data_ft.run_lora_train_filtered_steerable_d as baseline

    # Monkey-patch: replace Molmo2PointCollator with our wrapper
    import src.train.collator as collator_module
    collator_module.Molmo2PointCollator = IterPosCollatorA

    # Also need to pass noise_sigma to the collator constructor
    original_collator_init = IterPosCollatorA.__init__
    def patched_init(self, processor=None, context_len=2048, coordinate_scale="1000",
                     target_text_variant="sample_label", **kwargs):
        original_collator_init(
            self, processor=processor, context_len=context_len,
            coordinate_scale=coordinate_scale,
            target_text_variant=target_text_variant,
            noise_sigma=args.noise_sigma,
            drop_prob=args.drop_prob,
            **kwargs
        )
    IterPosCollatorA.__init__ = patched_init

    # Run baseline main with the same args
    # We need to reconstruct the argparser to match
    print("\nLaunching training with modified collator...")
    sys.argv = [sys.argv[0]]  # Clear args for baseline parser

    # Use baseline's main function directly
    baseline.main()


if __name__ == "__main__":
    main()
