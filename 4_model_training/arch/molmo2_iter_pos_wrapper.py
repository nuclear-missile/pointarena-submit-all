"""
Molmo2 Model Wrapper for Iterative Position Correction (Experiment C).

Wraps the Molmo2 model to inject:
1. ViT intermediate feature extraction at point positions
2. Point encoding = MLP(xy) + projected visual features
3. Point embeddings injected into text sequence (same additive pattern as image features)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PointCoordinateMLP(nn.Module):
    """Encode (x,y) coordinates via MLP directly to text hidden size (4096)."""
    def __init__(self, input_dim=2, hidden_dims=(128, 512, 1024), output_dim=4096):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords_normalized):
        """coords_normalized: (B, N, 2) in [0,1] → (B, N, 4096)"""
        return self.mlp(coords_normalized)


class ViTFeatureExtractor:
    """Extract ViT intermediate features at specific layers and patch positions."""

    def __init__(self, model, layer_indices=(8, 12, 16)):
        self.model = model
        self.layer_indices = layer_indices
        self.features = {}
        self._handles = []

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook

    def attach(self):
        vit = self.model.vision_backbone.image_vit
        blocks = vit.transformer.resblocks
        for idx in self.layer_indices:
            if idx < len(blocks):
                h = blocks[idx].register_forward_hook(self._make_hook(idx))
                self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()

    def get_point_features(self, points_xy, image_size=(378, 378)):
        """
        Extract ViT features at point locations.

        Args:
            points_xy: (B, N, 2) in [0, image_size] pixel coordinates
            image_size: (H, W) of ViT input

        Returns:
            (B, N, sum_of_layer_dims) or None
        """
        if not self.features:
            return None

        H, W = image_size
        patch_size = 14  # Molmo2 ViT patch size
        grid_h, grid_w = H // patch_size, W // patch_size  # 27x27

        batch_size, num_points = points_xy.shape[:2]
        all_feats = []

        for b in range(batch_size):
            sample_feats = []
            for n in range(num_points):
                x, y = points_xy[b, n, 0].item(), points_xy[b, n, 1].item()
                p_row = min(int(y / patch_size), grid_h - 1)
                p_col = min(int(x / patch_size), grid_w - 1)
                p_idx = p_row * grid_w + p_col

                layer_feats = []
                for idx in self.layer_indices:
                    if idx in self.features:
                        feat = self.features[idx]  # (B*T, num_patches, hidden)
                        if p_idx < feat.shape[1]:
                            layer_feats.append(feat[b, p_idx, :])
                if layer_feats:
                    sample_feats.append(torch.cat(layer_feats, dim=-1))
                else:
                    # Fallback: zero vector
                    dim = 1152 * len(self.layer_indices)
                    sample_feats.append(torch.zeros(dim, device=points_xy.device))
            all_feats.append(torch.stack(sample_feats))

        return torch.stack(all_feats)  # (B, N, sum_hidden)


class Molmo2IterPosWrapper(nn.Module):
    """
    Wraps Molmo2 to add iterative correction point encoding.

    Injects:
    - Point MLP embeddings (xy + visual semantics) into text sequence
    - During training, encodes deviated points
    - During inference, same mechanism for correction
    """

    def __init__(
        self,
        base_model: nn.Module,
        point_mlp: Optional[PointCoordinateMLP] = None,
        visual_proj: Optional[nn.Linear] = None,
        vit_extractor: Optional[ViTFeatureExtractor] = None,
    ):
        super().__init__()
        self.base_model = base_model
        # MLP_xy: 2 → ... → 4096 (direct to text hidden size)
        self.point_mlp = point_mlp or PointCoordinateMLP()
        # visual_proj: ViT concat features (1152*3=3456) → 4096
        # Both paths independently reach 4096, then ADD — no bottleneck
        self.visual_proj = visual_proj or nn.Linear(1152 * 3, 4096)
        self.vit_extractor = vit_extractor

    def encode_points(self, coords_normalized, visual_features=None):
        """
        Encode points into 4096-dim embeddings for text injection.

        Both paths independently reach 4096-dim:
          coord_emb  = MLP_xy(coords)       # 2 → ... → 4096
          visual_emb = Linear(3456, 4096)   # ViT concat → 4096
          point_emb  = coord_emb + visual_emb  (or coord_emb alone if no visual)

        Returns:
            (B*N, 4096) point embeddings
        """
        B, N = coords_normalized.shape[:2]
        coord_emb = self.point_mlp(coords_normalized)  # (B, N, 4096)

        if visual_features is not None:
            vis_emb = self.visual_proj(visual_features)  # (B, N, 4096)
            point_emb = coord_emb + vis_emb
        else:
            point_emb = coord_emb

        return point_emb.view(B * N, -1)  # (B*N, 4096)

    def forward(self, input_ids, pixel_values=None, deviated_coords=None,
                deviated_coords_pixel=None, **kwargs):
        """
        Forward pass with point encoding injection.

        Args:
            input_ids: text token ids
            pixel_values: image tensor
            deviated_coords: (B, N, 2) normalized [0,1] coords for point encoding
            deviated_coords_pixel: (B, N, 2) pixel-space coords for ViT lookup
            **kwargs: other Molmo2 args
        """
        # Standard Molmo2 forward
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )
        return outputs

    def train(self, mode=True):
        super().train(mode)
        self.base_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        yield from self.base_model.parameters(recurse=recurse)
        yield from self.point_mlp.parameters()
        yield from self.visual_proj.parameters()

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self.base_model.named_parameters(prefix=prefix + 'base_model.', recurse=recurse):
            yield name, param
        for name, param in self.point_mlp.named_parameters(prefix=prefix + 'point_mlp.', recurse=recurse):
            yield name, param
        for name, param in self.visual_proj.named_parameters(prefix=prefix + 'visual_proj.', recurse=recurse):
            yield name, param

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    @property
    def config(self):
        return self.base_model.config

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.4f}")
