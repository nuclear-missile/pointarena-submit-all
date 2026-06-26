"""
Wrapper dataset for iterative coordinate correction — JOINT TRAINING.

Wraps PointArenaSteerableDTrainingDataset. With mix_ratio < 1.0,
a fraction of samples keep the original question (base pointing);
the rest get a correction prompt with perturbed coordinates.

Perturbation strategy (correction samples):
  - Single-point:   Gaussian noise only, no drop/insert
  - Multi-point:    Gaussian noise + random drop + random insert
                    (rates calibrated to match Counting error rate ~30%)
"""
from __future__ import annotations

import numpy as np
from typing import Any


class IterPosDatasetWrapper:
    """
    Joint training wrapper.

    mix_ratio: fraction of samples that get correction prompt instead of base.
    noise_sigma: Gaussian sigma for coordinate perturbation (in [0, 1000] scale).
    drop_prob:   per-point drop probability (multi-point samples only).
    insert_prob: per-sample probability of inserting one false point (multi-point only).
    """

    def __init__(
        self,
        base_dataset: Any,
        noise_sigma: float = 100.0,
        drop_prob: float = 0.15,
        insert_prob: float = 0.15,
        mix_ratio: float = 0.5,
        seed: int = 42,
    ):
        self._base = base_dataset
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self.insert_prob = insert_prob
        self.mix_ratio = mix_ratio
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._base[idx]
        points = sample.get("points", [])
        question = str(sample.get("question", ""))

        # Deterministic decision per sample
        rng = np.random.RandomState(hash(f"iter_mix_{idx}") % (2**31))
        is_correction = rng.random() < self.mix_ratio

        if not is_correction or not points:
            sample["original_question"] = question
            sample["is_correction_sample"] = False
            return sample

        # ── Correction mode ──────────────────────────────────
        width = int(sample.get("width", 1000))
        height = int(sample.get("height", 1000))
        num_gt = len(points)
        is_multi = num_gt > 1

        deviated = []
        for x, y in points:
            # Per-point drop: only for multi-point samples
            if is_multi and rng.random() < self.drop_prob:
                continue
            # Gaussian perturbation: always applied
            nx = float(x) + rng.normal(0, self.noise_sigma)
            ny = float(y) + rng.normal(0, self.noise_sigma)
            nx = max(0.0, min(float(width), nx))
            ny = max(0.0, min(float(height), ny))
            deviated.append((round(nx, 1), round(ny, 1)))

        # Random insert: only for multi-point samples
        if is_multi and rng.random() < self.insert_prob:
            deviated.append((
                round(rng.uniform(0, float(width)), 1),
                round(rng.uniform(0, float(height)), 1),
            ))

        if not deviated:
            deviated = [(float(p[0]), float(p[1])) for p in points[:1]]

        # Build correction prompt
        label = str(sample.get("label", "target"))
        points_str = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in deviated)
        correction_question = (
            f"{question}\n\n"
            f"The following {label} coordinates were predicted but may contain errors: "
            f"[{points_str}]. "
            f"All coordinates are relative to image width/height in [0, 1000] scale. "
            f"Look at the image carefully and output the corrected {label} coordinates."
        )

        sample["original_question"] = question
        sample["deviated_points"] = deviated
        sample["question"] = correction_question
        sample["is_correction_sample"] = True
        # sample["points"] stays as ground truth (target)

        return sample

    @property
    def summary(self):
        return self._base.summary

    def balance_report(self):
        return self._base.balance_report()

    def category_counts(self):
        return self._base.category_counts()

    def __getattr__(self, name: str):
        return getattr(self._base, name)
