"""
Molmo2 Model Wrapper for Iterative Position Correction (Experiments B & C).

Monkey-patches Molmo2Model.build_input_embeddings to inject:
  1. ViT intermediate features at point locations (layers 8,12,16)
  2. Point embedding = MLP_xy(x,y) + Linear(ViT_concat, 4096)
  3. Coord map CNN features (Experiment C only)

Point embeddings are ADDED to text embeddings at special token positions,
following the same additive pattern as Molmo2 image features.

Architecture:
  Exp B: MLP_xy(2→4096) + Linear(ViT_3456→4096) → injected as point tokens
  Exp C: B + CoordMap CNN(100×100×4→N_tokens×4096) → injected as vision tokens
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PointCoordinateMLP(nn.Module):
    """Encode (x,y) to 4096-dim with LayerNorm for numerical stability."""
    def __init__(self, input_dim=2, hidden_dims=(128, 512, 1024), output_dim=4096):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LayerNorm(output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords_norm):
        """coords_norm: (*, 2) in [0,1] → (*, 4096)"""
        coords_norm = torch.nan_to_num(coords_norm, nan=0.5, posinf=1.0, neginf=0.0)
        coords_norm = coords_norm.clamp(-0.1, 1.1)
        return self.mlp(coords_norm)


class ViTFeatureExtractor(nn.Module):
    """Hooks into Molmo2 ViT to capture intermediate layer features."""

    def __init__(self, model, layer_indices=(8, 12, 16)):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self._features = {}
        self._handles = []

    def _make_hook(self, idx):
        def hook(module, input, output):
            self._features[idx] = output.detach()
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
        self._features.clear()

    def get_point_features(self, points_pixel, image_size=(378, 378)):
        """
        Extract ViT features at point locations.
        points_pixel: (B, N, 2) — pixel coordinates
        Returns: (B, N, 1152*len(layer_indices)) or None
        """
        if not self._features:
            return None
        H, W = image_size
        patch_size = 14
        grid_h, grid_w = H // patch_size, W // patch_size
        B, N = points_pixel.shape[:2]
        all_feats = []
        for b in range(B):
            sample_feats = []
            for n in range(N):
                x = float(points_pixel[b, n, 0].clamp(0, W - 1))
                y = float(points_pixel[b, n, 1].clamp(0, H - 1))
                r = min(int(y / patch_size), grid_h - 1)
                c = min(int(x / patch_size), grid_w - 1)
                p_idx = r * grid_w + c

                layer_feats = []
                for idx in self.layer_indices:
                    if idx in self._features:
                        f = self._features[idx]  # (B*T, num_patches, 1152)
                        if p_idx < f.shape[1]:
                            feat = f[b % f.shape[0], p_idx, :]
                            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                            layer_feats.append(feat)
                if layer_feats:
                    sample_feats.append(torch.cat(layer_feats, dim=-1))
                else:
                    sample_feats.append(torch.zeros(
                        1152 * len(self.layer_indices),
                        device=points_pixel.device, dtype=torch.bfloat16))
            all_feats.append(torch.stack(sample_feats))
        return torch.stack(all_feats)  # (B, N, 3456)


class CoordMapEncoder(nn.Module):
    """Encode 100×100×4 Gaussian coord map → N×4096 vision tokens. (Experiment C)"""

    def __init__(self, in_channels=4, out_dim=4096, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, out_dim * num_tokens),
            nn.LayerNorm(out_dim * num_tokens),
        )
        self.token_norm = nn.LayerNorm(out_dim)

    def forward(self, coord_map):
        B = coord_map.shape[0]
        x = self.cnn(coord_map)  # (B, num_tokens*4096)
        x = x.view(B, self.num_tokens, -1)  # (B, num_tokens, 4096)
        return self.token_norm(x)


class Molmo2IterPosWrapper(nn.Module):
    """
    Wraps Molmo2 to inject iterative correction point encodings.

    Usage:
        wrapper = Molmo2IterPosWrapper(base_model, exp_mode="B"|"C")
        outputs = wrapper(
            input_ids=..., pixel_values=...,
            deviated_coords=...,    # (B, N, 2) in [0,1]
            deviated_pixels=...,    # (B, N, 2) in pixel coords (for ViT lookup)
            coord_map=...,          # (B, 4, 100, 100) — Exp C only
            labels=...,
            **kwargs
        )
    """

    def __init__(self, base_model, exp_mode="B"):
        super().__init__()
        self.base_model = base_model
        self.exp_mode = exp_mode

        self.point_mlp = PointCoordinateMLP()
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(1152 * 3),  # normalize raw ViT features BEFORE projection
            nn.Linear(1152 * 3, 4096),
            nn.LayerNorm(4096),
        )
        self.vit_extractor = ViTFeatureExtractor(base_model.model)
        self.point_norm = nn.LayerNorm(4096)  # Final normalization before injection

        self.coord_map_encoder = None
        if exp_mode == "C":
            self.coord_map_encoder = CoordMapEncoder()

        # Register ViT hooks now
        self.vit_extractor.attach()

        # Find Molmo2Model inner module (may be wrapped by PeftModel)
        inner = base_model
        for attr in ['model', 'base_model', 'model.model']:
            candidate = inner
            for part in attr.split('.'):
                if hasattr(candidate, part):
                    candidate = getattr(candidate, part)
                else:
                    candidate = None
                    break
            if candidate is not None and hasattr(candidate, 'build_input_embeddings'):
                inner = candidate
                break
        if not hasattr(inner, 'build_input_embeddings'):
            # Fallback: search recursively
            for name, module in base_model.named_modules():
                if hasattr(module, 'build_input_embeddings'):
                    inner = module
                    break

        self._inner_model = inner
        self._original_build = inner.build_input_embeddings
        inner.build_input_embeddings = self._build_input_embeddings

    def _build_input_embeddings(self, input_ids, images=None, token_pooling=None, **kwargs):
        # Call original
        x, image_features = self._original_build(input_ids, images, token_pooling)

        # Inject coord map tokens (appended as image_patch_id tokens)
        if hasattr(self, '_pending_cm_tokens') and self._pending_cm_tokens is not None:
            cm = self._pending_cm_tokens
            B, N_cm = cm.shape[:2]
            cm_flat = cm.view(B * N_cm, -1).to(dtype=x.dtype, device=x.device)
            patch_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            is_patch = (input_ids.view(-1) == patch_id)
            all_patch_pos = is_patch.nonzero(as_tuple=True)[0]
            if all_patch_pos.numel() >= cm_flat.shape[0]:
                cm_positions = all_patch_pos[-cm_flat.shape[0]:]
                x_flat = x.view(-1, x.shape[-1])
                x_flat[cm_positions] += cm_flat
            self._pending_cm_tokens = None

        # Inject point embeddings from stored attributes (set in forward())
        deviated_coords = getattr(self, '_deviated_coords', None)
        deviated_pixels = getattr(self, '_deviated_pixels', None)

        if deviated_coords is not None and deviated_coords.numel() > 0:
            B, N = deviated_coords.shape[:2]
            # MLP_xy encoding
            coord_emb = self.point_mlp(deviated_coords)  # (B, N, 4096)
            coord_emb = torch.nan_to_num(coord_emb, nan=0.0, posinf=0.0, neginf=0.0)

            # Visual features from ViT hooks
            if deviated_pixels is not None:
                vis_feat = self.vit_extractor.get_point_features(deviated_pixels)
                if vis_feat is not None:
                    vis_feat = torch.nan_to_num(vis_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    # Convert to float32 before projection (module params are float32)
                    vis_emb = self.visual_proj(vis_feat.float())  # (B, N, 4096)
                    vis_emb = torch.nan_to_num(vis_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    point_emb = coord_emb + vis_emb.to(coord_emb.dtype)
                else:
                    point_emb = coord_emb
            else:
                point_emb = coord_emb

            point_emb = self.point_norm(point_emb.view(B * N, -1).float())
            point_emb = point_emb.to(x.dtype)
            point_emb = torch.nan_to_num(point_emb, nan=0.0, posinf=0.0, neginf=0.0)

            # Find point token positions in input_ids
            point_token_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            is_point = input_ids.view(-1) == point_token_id

            if is_point.sum() >= point_emb.shape[0]:
                x_flat = x.view(-1, x.shape[-1])
                point_positions = is_point.nonzero(as_tuple=True)[0][:point_emb.shape[0]]
                x_flat[point_positions] += point_emb.to(x.dtype)

            # Clear stored attributes
            self._deviated_coords = None
            self._deviated_pixels = None

        return x, image_features

    def forward(self, input_ids=None, pixel_values=None, deviated_coords=None,
                deviated_pixels=None, coord_map=None, labels=None, attention_mask=None, **kwargs):
        # Store for injection in build_input_embeddings (Molmo2Model.forward doesn't pass these)
        if deviated_coords is not None:
            self._deviated_coords = deviated_coords
        if deviated_pixels is not None:
            self._deviated_pixels = deviated_pixels

        # Encode coord map BEFORE model forward to extend sequence
        if coord_map is not None and self.coord_map_encoder is not None:
            coord_map = torch.nan_to_num(coord_map, nan=0.0, posinf=1.0, neginf=0.0)
            cm_tokens = self.coord_map_encoder(coord_map)  # (B, 16, 4096)
            cm_tokens = torch.nan_to_num(cm_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            B, N_cm = cm_tokens.shape[:2]
            cm_token_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            cm_ids = torch.full((B, N_cm), cm_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, cm_ids], dim=1)
            if attention_mask is not None:
                cm_mask = torch.ones(B, N_cm, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, cm_mask], dim=1)
            if labels is not None:
                cm_labels = torch.full((B, N_cm), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, cm_labels], dim=1)
            self._pending_cm_tokens = cm_tokens

        # Don't pass extra kwargs to base_model — they're stored as attributes
        return self.base_model(
            input_ids=input_ids, pixel_values=pixel_values,
            labels=labels, attention_mask=attention_mask, **kwargs)

    # Delegate generation-related methods to base_model
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    def can_generate(self):
        return self.base_model.can_generate()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, *a, **kw):
        self.base_model.gradient_checkpointing_enable(*a, **kw)

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self):
        return next(self.parameters()).device

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || all: {total:,} || {100*trainable/total:.4f}%")

    def parameters(self, recurse=True):
        yield from self.base_model.parameters(recurse=recurse)
        yield from self.point_mlp.parameters()
        yield from self.visual_proj.parameters()
        if self.coord_map_encoder is not None:
            yield from self.coord_map_encoder.parameters()

    def named_parameters(self, prefix='', recurse=True):
        for n, p in self.base_model.named_parameters(prefix=prefix+'base_model.', recurse=recurse):
            yield n, p
        for n, p in self.point_mlp.named_parameters(prefix=prefix+'point_mlp.', recurse=recurse):
            yield n, p
        for n, p in self.visual_proj.named_parameters(prefix=prefix+'visual_proj.', recurse=recurse):
            yield n, p
        if self.coord_map_encoder is not None:
            for n, p in self.coord_map_encoder.named_parameters(prefix=prefix+'coord_map.', recurse=recurse):
                yield n, p
