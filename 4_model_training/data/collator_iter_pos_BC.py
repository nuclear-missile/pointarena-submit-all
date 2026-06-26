"""
Collators for Experiments B and C.

B: text correction prompt + point MLP embeddings + ViT visual features
C: B + coordinate Gaussian map image

Point embeddings are injected via <pt>x,y</pt> XML tags in the text.
The model wrapper detects these tokens and replaces their embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.train.collator import Molmo2PointCollator
from .coordinate_map_utils import build_coordinate_map_tensor


class IterPosCollatorB(Molmo2PointCollator):
    """
    Experiment B: Point MLP + Visual Semantics.

    Uses deviated_points from IterPosDatasetWrapper (which handles correction
    prompt building and mix_ratio). Only builds deviated_coords/pixels for
    correction samples; base samples get zero-filled tensors.
    """

    def __init__(self, *args, noise_sigma=100.0, drop_prob=0.15, insert_prob=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self.insert_prob = insert_prob

    def __call__(self, batch):
        deviated_all = []
        pixels_all = []
        for s in batch:
            dev = s.get("deviated_points", [])  # Already computed by IterPosDatasetWrapper
            w = int(s.get("width", 1000))
            h = int(s.get("height", 1000))

            if dev:
                norm = torch.tensor([[x/1000.0, y/1000.0] for x, y in dev], dtype=torch.float32)
                pixel = torch.tensor([[x/1000.0*w, y/1000.0*h] for x, y in dev], dtype=torch.float32)
            else:
                norm = torch.zeros(0, 2, dtype=torch.float32)
                pixel = torch.zeros(0, 2, dtype=torch.float32)
            deviated_all.append(norm)
            pixels_all.append(pixel)

        # Standard Molmo2 collation (wrapper already set question)
        result = super().__call__(batch)

        # Pad and attach point data
        max_pts = max(d.shape[0] for d in deviated_all)
        B = len(deviated_all)

        dev_tensor = torch.zeros(B, max_pts, 2)
        pix_tensor = torch.zeros(B, max_pts, 2)
        mask = torch.zeros(B, max_pts, dtype=torch.bool)
        for i, (d, p) in enumerate(zip(deviated_all, pixels_all)):
            n = d.shape[0]
            if n > 0:
                dev_tensor[i, :n] = d
                pix_tensor[i, :n] = p
                mask[i, :n] = True

        result["deviated_coords"] = dev_tensor
        result["deviated_pixels"] = pix_tensor
        result["deviated_mask"] = mask
        return result


class IterPosCollatorC(IterPosCollatorB):
    """
    Experiment C: B + Coordinate Gaussian Map.

    Adds a 100×100×4 Gaussian heatmap tensor. Only for correction samples;
    base samples get an empty map (single point at center as placeholder).
    """

    def __call__(self, batch):
        result = super().__call__(batch)

        maps = []
        for s in batch:
            dev = s.get("deviated_points", [])
            if not dev:
                dev = [(500, 500)]
            cm = build_coordinate_map_tensor(
                dev, grid_size=100, sigmas=[8.0, 4.0, 2.0, 1.0],
                coordinate_scale=1000.0,
            )
            maps.append(cm)
        result["coord_maps"] = torch.stack(maps)
        return result
