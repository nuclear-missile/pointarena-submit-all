"""
Collator for Steerability with B's anchor point encoding.

- Reads anchor_norm01 from steerability samples
- Encodes anchor coordinates using the same mechanism as B (PointCoordinateMLP + ViT features)
- For iter version: adds correction prompt with perturbed output point
"""
from __future__ import annotations

import numpy as np
import torch
from .collator_iter_pos_BC import IterPosCollatorB


class SteerabilityAnchorCollator(IterPosCollatorB):
    """
    Extends IterPosCollatorB to also encode Steerability anchor points.

    For samples with anchor_norm01 metadata, the anchor coordinates are
    added to deviated_coords for injection via Molmo2IterPosWrapper.
    """

    def __call__(self, batch):
        # Collect anchor points for encoding
        anchor_all = []
        anchor_pixels_all = []
        has_anchor = False

        for s in batch:
            anchor_norm = s.get("meta", {}).get("anchor_norm01", None)
            if anchor_norm is None:
                anchor_norm = s.get("anchor_norm01", None)

            if anchor_norm and len(anchor_norm) == 2:
                has_anchor = True
                w = int(s.get("width", 1000))
                h = int(s.get("height", 1000))
                anchor_all.append(torch.tensor(
                    [[float(anchor_norm[0]), float(anchor_norm[1])]],
                    dtype=torch.float32
                ))
                anchor_pixels_all.append(torch.tensor(
                    [[float(anchor_norm[0]) * w / 1000.0, float(anchor_norm[1]) * h / 1000.0]],
                    dtype=torch.float32
                ))
            else:
                anchor_all.append(torch.zeros(0, 2, dtype=torch.float32))
                anchor_pixels_all.append(torch.zeros(0, 2, dtype=torch.float32))

        # Standard B collation (handles deviation/correction if iter mode)
        result = super().__call__(batch)

        # Combine anchor coords with any existing deviated_coords
        B = len(anchor_all)
        max_anchors = max(a.shape[0] for a in anchor_all)

        if has_anchor and max_anchors > 0:
            anchor_tensor = torch.zeros(B, max_anchors, 2)
            anchor_pix_tensor = torch.zeros(B, max_anchors, 2)
            for i in range(B):
                n = anchor_all[i].shape[0]
                if n > 0:
                    anchor_tensor[i, :n] = anchor_all[i]
                    anchor_pix_tensor[i, :n] = anchor_pixels_all[i]

            # Prepend anchors to existing deviated_coords (if any)
            existing = result.get("deviated_coords", None)
            existing_pix = result.get("deviated_pixels", None)
            if existing is not None and existing.shape[1] > 0:
                result["deviated_coords"] = torch.cat([anchor_tensor, existing], dim=1)
                result["deviated_pixels"] = torch.cat([anchor_pix_tensor, existing_pix], dim=1)
            else:
                result["deviated_coords"] = anchor_tensor
                result["deviated_pixels"] = anchor_pix_tensor

        return result
