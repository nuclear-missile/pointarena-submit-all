"""
Architecture B/C: Molmo2 with Point Coordinate Injection.

B: PointCoordinateMLP(x,y) + ViT visual features → additive injection
C: B + CoordMap CNN (100×100×4 → N×4096 tokens) → appended as vision tokens

This module reconstructs the Molmo2IterPosWrapper architecture for inference,
given a base model + saved sub-module weights.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sub-modules (must match training architecture exactly) ──────

class PointCoordinateMLP(nn.Module):
    """Encode (x,y) normalized coords → 4096-dim embedding."""
    def __init__(self, input_dim=2, hidden_dims=(128, 512, 1024), output_dim=4096):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(("linear_{}".format(i), nn.Linear(dims[i], dims[i+1])))
            layers.append(("ln_{}".format(i), nn.LayerNorm(dims[i+1])))
            layers.append(("act_{}".format(i), nn.SiLU()))
        layers.append(("linear_final", nn.Linear(hidden_dims[-1], output_dim)))
        layers.append(("ln_final", nn.LayerNorm(output_dim)))
        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, coords_norm):
        coords_norm = torch.nan_to_num(coords_norm, nan=0.5, posinf=1.0, neginf=0.0)
        coords_norm = coords_norm.clamp(-0.1, 1.1)
        return self.mlp(coords_norm)


class ViTFeatureExtractor(nn.Module):
    """Hooks into Molmo2 ViT to capture features at point locations (layers 8,12,16)."""
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

    @torch.no_grad()
    def get_point_features(self, points_pixel, image_size=(378, 378)):
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
                        f = self._features[idx]
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
        return torch.stack(all_feats)


class CoordMapEncoder(nn.Module):
    """CNN encodes 100×100×4 coord map → N×4096 vision tokens (Experiment C only)."""
    def __init__(self, in_channels=4, out_dim=4096, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.cnn = nn.Sequential(OrderedDict([
            ("0", nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)),
            ("1", nn.BatchNorm2d(64)),
            ("2", nn.SiLU()),
            ("3", nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ("4", nn.BatchNorm2d(128)),
            ("5", nn.SiLU()),
            ("6", nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            ("7", nn.BatchNorm2d(256)),
            ("8", nn.SiLU()),
        ]))
        self.proj = nn.Sequential(OrderedDict([
            ("pool", nn.AdaptiveAvgPool2d((4, 4))),
            ("flatten", nn.Flatten()),
            ("linear", nn.Linear(256 * 16, out_dim * num_tokens)),
            ("norm", nn.LayerNorm(out_dim * num_tokens)),
        ]))
        self.token_norm = nn.LayerNorm(out_dim)

    def forward(self, coord_map):
        B = coord_map.shape[0]
        x = self.cnn(coord_map)
        x = self.proj[0](x)   # pool
        x = self.proj[1](x)   # flatten
        x = self.proj[2](x)   # linear
        x = self.proj[3](x)   # norm
        x = x.view(B, self.num_tokens, -1)
        return self.token_norm(x)


# ── Wrapper (reconstructed from saved weights) ─────────────────

class PointInjectionWrapper(nn.Module):
    """
    Wraps Molmo2 to inject point coordinate encodings (B/C architecture).

    Load a base Molmo2 model, then call load_arch_weights() with saved
    sub-module state dicts to reconstruct the full architecture.
    """

    def __init__(self, base_model, exp_mode="B"):
        super().__init__()
        self.base_model = base_model
        self.exp_mode = exp_mode

        # Sub-modules (weights loaded separately)
        self.point_mlp = PointCoordinateMLP()
        self.visual_proj = nn.Sequential(OrderedDict([
            ("0", nn.LayerNorm(3456)),
            ("1", nn.Linear(3456, 4096)),
            ("2", nn.LayerNorm(4096)),
        ]))
        self.vit_extractor = ViTFeatureExtractor(base_model.model)
        self.point_norm = nn.LayerNorm(4096)

        self.coord_map_encoder = None
        if exp_mode == "C":
            self.coord_map_encoder = CoordMapEncoder()

        # Hook into ViT
        self.vit_extractor.attach()

        # Find inner Molmo2Model and monkey-patch build_input_embeddings
        self._patch_build_input_embeddings()

    def load_arch_weights(self, arch_state: dict):
        """Load saved sub-module weights into this wrapper."""
        self.point_mlp.load_state_dict(arch_state["point_mlp"], strict=True)

        vp = arch_state["visual_proj"]
        vp_mapped = {
            "0.weight": vp["0.weight"], "0.bias": vp["0.bias"],
            "1.weight": vp["1.weight"], "1.bias": vp["1.bias"],
            "2.weight": vp["2.weight"], "2.bias": vp["2.bias"],
        }
        self.visual_proj.load_state_dict(vp_mapped)
        self.point_norm.load_state_dict(arch_state["point_norm"])

        if self.coord_map_encoder is not None and "coord_map_encoder" in arch_state:
            self.coord_map_encoder.load_state_dict(
                arch_state["coord_map_encoder"], strict=True
            )

    def _patch_build_input_embeddings(self):
        inner = self.base_model
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
            for _name, module in self.base_model.named_modules():
                if hasattr(module, 'build_input_embeddings'):
                    inner = module
                    break

        self._inner_model = inner
        self._original_build = inner.build_input_embeddings
        inner.build_input_embeddings = self._build_input_embeddings

    def _build_input_embeddings(self, input_ids, images=None, token_pooling=None, **kwargs):
        x, image_features = self._original_build(input_ids, images, token_pooling)

        # Inject coord map tokens
        if hasattr(self, '_pending_cm_tokens') and self._pending_cm_tokens is not None:
            cm = self._pending_cm_tokens
            B_cm, N_cm = cm.shape[:2]
            cm_flat = cm.reshape(B_cm * N_cm, -1).to(dtype=x.dtype, device=x.device)
            patch_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            is_patch = (input_ids.view(-1) == patch_id)
            all_patch_pos = is_patch.nonzero(as_tuple=True)[0]
            if all_patch_pos.numel() >= cm_flat.shape[0]:
                cm_positions = all_patch_pos[-cm_flat.shape[0]:]
                x_flat = x.view(-1, x.shape[-1])
                x_flat[cm_positions] += cm_flat
            self._pending_cm_tokens = None

        # Inject point embeddings
        deviated_coords = getattr(self, '_deviated_coords', None)
        deviated_pixels = getattr(self, '_deviated_pixels', None)

        if deviated_coords is not None and deviated_coords.numel() > 0:
            B_pt, N_pt = deviated_coords.shape[:2]
            coord_emb = self.point_mlp(deviated_coords)
            coord_emb = torch.nan_to_num(coord_emb, nan=0.0, posinf=0.0, neginf=0.0)

            if deviated_pixels is not None:
                vis_feat = self.vit_extractor.get_point_features(deviated_pixels)
                if vis_feat is not None:
                    vis_feat = torch.nan_to_num(vis_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    vis_emb = self.visual_proj(vis_feat.float())
                    vis_emb = torch.nan_to_num(vis_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    point_emb = coord_emb + vis_emb.to(coord_emb.dtype)
                else:
                    point_emb = coord_emb
            else:
                point_emb = coord_emb

            point_emb = self.point_norm(point_emb.reshape(B_pt * N_pt, -1).float())
            point_emb = point_emb.to(x.dtype)
            point_emb = torch.nan_to_num(point_emb, nan=0.0, posinf=0.0, neginf=0.0)

            point_token_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            is_point = input_ids.view(-1) == point_token_id
            if is_point.sum() >= point_emb.shape[0]:
                x_flat = x.view(-1, x.shape[-1])
                point_positions = is_point.nonzero(as_tuple=True)[0][:point_emb.shape[0]]
                x_flat[point_positions] += point_emb.to(x.dtype)

            self._deviated_coords = None
            self._deviated_pixels = None

        return x, image_features

    def forward(self, input_ids=None, pixel_values=None, deviated_coords=None,
                deviated_pixels=None, coord_map=None, labels=None,
                attention_mask=None, **kwargs):
        if deviated_coords is not None:
            self._deviated_coords = deviated_coords
        if deviated_pixels is not None:
            self._deviated_pixels = deviated_pixels

        if coord_map is not None and self.coord_map_encoder is not None:
            coord_map = torch.nan_to_num(coord_map, nan=0.0, posinf=1.0, neginf=0.0)
            cm_tokens = self.coord_map_encoder(coord_map)
            cm_tokens = torch.nan_to_num(cm_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            B_cm, N_cm = cm_tokens.shape[:2]
            cm_token_id = getattr(self.base_model.config, 'image_patch_id', 151938)
            cm_ids = torch.full((B_cm, N_cm), cm_token_id, dtype=input_ids.dtype,
                                device=input_ids.device)
            input_ids = torch.cat([input_ids, cm_ids], dim=1)
            if attention_mask is not None:
                cm_mask = torch.ones(B_cm, N_cm, dtype=attention_mask.dtype,
                                     device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, cm_mask], dim=1)
            if labels is not None:
                cm_labels = torch.full((B_cm, N_cm), -100, dtype=labels.dtype,
                                       device=labels.device)
                labels = torch.cat([labels, cm_labels], dim=1)
            self._pending_cm_tokens = cm_tokens

        return self.base_model(
            input_ids=input_ids, pixel_values=pixel_values,
            labels=labels, attention_mask=attention_mask, **kwargs)

    # Delegate to base_model
    def prepare_inputs_for_generation(self, *a, **kw):
        return self.base_model.prepare_inputs_for_generation(*a, **kw)
    def can_generate(self):
        return self.base_model.can_generate()
    def generate(self, *a, **kw):
        return self.base_model.generate(*a, **kw)
    def gradient_checkpointing_enable(self, *a, **kw):
        self.base_model.gradient_checkpointing_enable(*a, **kw)
    def save_pretrained(self, *a, **kw):
        return self.base_model.save_pretrained(*a, **kw)

    @property
    def config(self):
        return self.base_model.config
    @property
    def device(self):
        return next(self.parameters()).device

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
