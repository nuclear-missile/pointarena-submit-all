"""
Experiment B Collator: Coordinate Map as Additional Image

Extends Molmo2PointCollator to:
1. Build coordinate Gaussian map images from deviated points
2. Add them as a second image in each message
3. Use correction prompts (same as Experiment A)
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.train.collator import Molmo2PointCollator
from .coordinate_map_utils import build_coordinate_map_image


class IterPosCollatorB(Molmo2PointCollator):
    """
    Experiment B: Coordinate Map + Text-Based Deviated Points.

    For each sample:
    - Perturb GT points → deviated points
    - Build coordinate map image from deviated points
    - Add coordinate map as second image in user message
    - Build correction prompt from deviated points (text)
    - Target: ground truth points (unchanged)
    """

    def __init__(self, *args, noise_sigma=100.0, drop_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self._step_counter = 0

    def _perturb_points(self, points, width, height):
        step_seed = 42 + self._step_counter
        rng = np.random.RandomState(step_seed)
        deviated = []
        for px, py in points:
            if rng.random() < self.drop_prob:
                continue
            nx = float(px) + rng.normal(0, self.noise_sigma)
            ny = float(py) + rng.normal(0, self.noise_sigma)
            nx = max(0.0, min(float(width), nx))
            ny = max(0.0, min(float(height), ny))
            deviated.append((round(nx, 1), round(ny, 1)))
        if not deviated:
            deviated = [(float(p[0]), float(p[1])) for p in points[:1]]
        return deviated

    def _build_correction_question(self, original_question, deviated_points):
        points_str = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in deviated_points)
        return (
            f"{original_question}\n\n"
            f"The following point coordinates were predicted but may contain errors: "
            f"[{points_str}]. "
            f"All coordinates are relative to image width/height in [0, 1000] scale. "
            f"Look at the image and the coordinate map carefully, "
            f"then output the corrected coordinates."
        )

    def _build_coord_map_image(self, deviated_points):
        """Build coordinate map as PIL Image."""
        if not deviated_points:
            return Image.new("RGB", (378, 378), color=(0, 0, 0))
        return build_coordinate_map_image(
            deviated_points,
            output_size=(378, 378),
            sigmas=[8.0, 4.0, 2.0, 1.0],
            coordinate_scale=1000.0,
        )

    def __call__(self, batch):
        # Build coord maps and modify prompts
        coord_maps = []
        for sample in batch:
            original_question = sample.get("question", "")
            points = sample.get("points", [])
            width = sample.get("width", 1000)
            height = sample.get("height", 1000)

            deviated = self._perturb_points(points, width, height)
            sample["original_question"] = original_question
            sample["deviated_points"] = deviated
            sample["question"] = self._build_correction_question(
                original_question, deviated
            )

            cm = self._build_coord_map_image(deviated)
            coord_maps.append(cm)
            self._step_counter += 1

        # Override message building to include coord map as second image
        # We monkey-patch the image list to prepend the coord map
        messages = []
        prompt_messages = []
        sample_sources = []
        sample_ids = []
        sample_targets = []

        for i, sample in enumerate(batch):
            question = str(sample["question"]).strip()
            style = str(sample.get("style", "pointing")).strip().lower() or "pointing"
            label = str(sample.get("label", "target")).strip() or "target"
            width = int(sample["width"])
            height = int(sample["height"])
            points = sample["points"]
            metadata = sample.get("metadata", {}) or {}
            image = self._to_rgb_image(sample["image"])
            coord_map_img = coord_maps[i]

            target = self._build_target_text(
                points=points, width=width, height=height, label=label, style=style
            )

            # User message with BOTH images: coord map first, then original
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": coord_map_img},
                    {"type": "image", "image": image},
                ],
            }
            assistant_turn = {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }
            messages.append([user_turn, assistant_turn])
            prompt_messages.append([user_turn])
            sample_sources.append(str(metadata.get("source", "unknown")))
            sample_ids.append(str(sample.get("id", "")))
            sample_targets.append(target)

        # Use processor with messages containing both images
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_len,
            padding=True,
        )

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_len,
            padding=True,
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        attention_mask = model_inputs.get("attention_mask")
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if "token_type_ids" in model_inputs:
            mm_mask = model_inputs["token_type_ids"].to(torch.bool)
            labels[mm_mask] = -100

        prompt_attn = prompt_inputs.get("attention_mask")
        for i in range(labels.size(0)):
            if prompt_attn is not None:
                prompt_len = int(prompt_attn[i].sum().item())
            elif pad_id is not None:
                prompt_len = int((prompt_inputs["input_ids"][i] != pad_id).sum().item())
            else:
                prompt_len = int(prompt_inputs["input_ids"][i].numel())
            prompt_len = max(0, min(prompt_len, labels.size(1)))
            labels[i, :prompt_len] = -100

        model_inputs["labels"] = labels
        model_inputs["_meta_sources"] = sample_sources
        model_inputs["_meta_ids"] = sample_ids
        model_inputs["_meta_targets"] = sample_targets
        return model_inputs
