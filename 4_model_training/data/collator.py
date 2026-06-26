from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from PIL import ImageFile

from src.train.molmo2_point_formatter import Molmo2PointFormatter

# Some training images may have truncated JPEG tails.
# Allow PIL to decode them instead of crashing DataLoader workers.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class Molmo2PointCollator:
    processor: Any
    context_len: int = 2048
    coordinate_scale: str = "1000"
    # Controls assistant target text protocol.
    # - sample_label: current behavior (uses sample label)
    # - html_target/html_empty/html_dot/html_x/html_self_closing: controlled ablations
    # - coord_csv: minimal experimental protocol "x,y" (0~1000 space)
    target_text_variant: str = "sample_label"

    def __post_init__(self) -> None:
        self.point_formatter = Molmo2PointFormatter(coordinate_scale=self.coordinate_scale)

    def _to_rgb_image(self, image_obj: Any) -> Image.Image:
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")

        if isinstance(image_obj, (str, Path)):
            with Image.open(str(image_obj)) as im:
                return im.convert("RGB")

        # Fallback for ndarray-like inputs.
        try:
            return Image.fromarray(image_obj).convert("RGB")
        except Exception as exc:  # pragma: no cover
            raise TypeError(f"Unsupported image field type: {type(image_obj)}") from exc

    def _build_target_text(
        self,
        *,
        points: list[tuple[float, float]],
        width: int,
        height: int,
        label: str,
        style: str,
    ) -> str:
        variant = (self.target_text_variant or "sample_label").strip().lower()

        if variant == "sample_label":
            return self.point_formatter.format_image_points(
                points=points,
                scale=[width, height],
                label=label,
                mode=style,
            )

        if not points:
            return "There are none."

        # Coordinate string in Molmo2 html protocol.
        coord = self.point_formatter._format_single_image_coordinates(points, [width, height])
        coord = f"1 {coord}"

        if variant == "html_self_closing":
            point_str = f'<points coords="{coord}"/>'
            return self.point_formatter._build_output(point_str, len(points), style)

        if variant in {"html_target", "html_empty", "html_dot", "html_x"}:
            label_text = "target"
            if variant == "html_empty":
                label_text = ""
            elif variant == "html_dot":
                label_text = "."
            elif variant == "html_x":
                label_text = "x"
            point_str = f'<points coords="{coord}">{label_text}</points>'
            return self.point_formatter._build_output(point_str, len(points), style)

        if variant == "coord_csv":
            scaled = [self.point_formatter._scale_point(p, [width, height]) for p in points]
            scaled = sorted(scaled, key=lambda p: (p[0], p[1]))
            return "; ".join(f"{x},{y}" for x, y in scaled)

        # Safe fallback
        return self.point_formatter.format_image_points(
            points=points,
            scale=[width, height],
            label=label,
            mode=style,
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        messages = []
        prompt_messages = []
        sample_sources: list[str] = []
        sample_ids: list[str] = []
        sample_targets: list[str] = []

        for sample in batch:
            question = str(sample["question"]).strip()
            style = str(sample.get("style", "pointing")).strip().lower() or "pointing"
            label = str(sample.get("label", "target")).strip() or "target"
            width = int(sample["width"])
            height = int(sample["height"])
            points = sample["points"]
            metadata = sample.get("metadata", {}) or {}

            target = self._build_target_text(
                points=points,
                width=width,
                height=height,
                label=label,
                style=style,
            )
            image = self._to_rgb_image(sample["image"])

            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
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
        # Attach lightweight metadata for sampling diagnostics.
        model_inputs["_meta_sources"] = sample_sources
        model_inputs["_meta_ids"] = sample_ids
        model_inputs["_meta_targets"] = sample_targets
        return model_inputs
