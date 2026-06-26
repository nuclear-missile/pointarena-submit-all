#!/usr/bin/env python3
"""SAM3 ONNX segmentation utilities for the PointArena steerable pipeline.

Extracted from guide4/run_pipeline.py -- provides the SAM3 mask generation
and geometry verification functions used by the anchor-relative task pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3TrackerProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_ANGLES: dict[str, float] = {
    "right": 0.0,
    "down-right": 45.0,
    "down": 90.0,
    "down-left": 135.0,
    "left": 180.0,
    "up-left": 225.0,
    "up": 270.0,
    "up-right": 315.0,
}

AXIS_DIRECTIONS = ["right", "down", "left", "up"]
DIAGONAL_DIRECTIONS = ["up-right", "down-right", "down-left", "up-left"]
MOVE_DIRECTIONS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class Paths:
    """Directory layout for pipeline stages."""
    root: Path
    filtered: Path
    sam_masks: Path
    task_pool: Path
    rewrites: Path
    visuals: Path
    final: Path
    reports: Path
    cache: Path


@dataclass
class ObjectRecord:
    """A single detected object with label, geometry, and point annotations."""
    object_id: str
    image_order: int
    object_order: int
    image_id: str
    image_path: str
    width: int
    height: int
    row_id: str
    label_raw: str
    label_clean: str
    label_display: str
    same_class_key: str
    head_noun: str
    points_xy: list[list[float]]
    points_norm: list[list[float]]
    center_xy: list[float]
    center_norm: list[float]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def safe_name(text: str, max_len: int = 90) -> str:
    """Sanitize a string for use as a filename."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return value[:max_len].strip("._") or "object"


def stable_hash(payload: Any) -> int:
    """Deterministic hash of an arbitrary JSON-serialisable payload."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return int.from_bytes(__import__("hashlib").sha256(raw.encode("utf-8")).digest()[:8], "big")


def deterministic_rng(seed: int, *parts: Any) -> random.Random:
    """Create a seeded RNG that depends on every part."""
    return random.Random(stable_hash({"seed": seed, "parts": list(parts)}))


def json_default(obj: Any) -> Any:
    """JSON serializer fallback for Path, numpy arrays, etc."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def color_for_index(index: int) -> tuple[int, int, int]:
    """Deterministic RGB colour from an integer index (golden-ratio hue)."""
    hue = (index * 0.618033988749895) % 1.0
    saturation = 0.72
    value = 0.96
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - f * saturation)
    t = value * (1.0 - (1.0 - f) * saturation)
    r, g, b = {
        0: (value, t, p),
        1: (q, value, p),
        2: (p, value, t),
        3: (p, q, value),
        4: (t, p, value),
        5: (value, p, q),
    }[i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a truetype font, falling back to default."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def objects_from_df(df: "pd.DataFrame") -> dict[str, list[ObjectRecord]]:
    """Convert a DataFrame of object rows into per-image ObjectRecord lists.

    Requires the input DataFrame to have columns matching ObjectRecord fields.
    Points must be stored as JSON strings in points_xy_json / points_norm_json.
    """
    import pandas as pd

    out: dict[str, list[ObjectRecord]] = defaultdict(list)
    for row in df.to_dict(orient="records"):
        obj = ObjectRecord(
            object_id=str(row["object_id"]),
            image_order=int(row["image_order"]),
            object_order=int(row["object_order"]),
            image_id=str(row["image_id"]),
            image_path=str(row["image_path"]),
            width=int(row["width"]),
            height=int(row["height"]),
            row_id=str(row["row_id"]),
            label_raw=str(row["label_raw"]),
            label_clean=str(row["label_clean"]),
            label_display=str(row["label_display"]),
            same_class_key=str(row["same_class_key"]),
            head_noun=str(row["head_noun"]),
            points_xy=json.loads(row["points_xy_json"]),
            points_norm=json.loads(row["points_norm_json"]),
            center_xy=[float(row["center_x"]), float(row["center_y"])],
            center_norm=[float(row["center_x_norm"]), float(row["center_y_norm"])],
        )
        out[obj.image_id].append(obj)
    return out


# ---------------------------------------------------------------------------
# SAM3 ONNX inference
# ---------------------------------------------------------------------------


def ensure_sam_model_dir(args: argparse.Namespace) -> Path:
    """Download (or locate cached) SAM3 ONNX files and return the model directory."""
    if args.sam_model_dir:
        return args.sam_model_dir.resolve()
    files = [
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        f"onnx/vision_encoder_{args.sam_dtype}.onnx",
        f"onnx/vision_encoder_{args.sam_dtype}.onnx_data",
        f"onnx/prompt_encoder_mask_decoder_{args.sam_dtype}.onnx",
        f"onnx/prompt_encoder_mask_decoder_{args.sam_dtype}.onnx_data",
    ]
    paths = []
    for filename in files:
        paths.append(
            Path(
                hf_hub_download(
                    repo_id=args.sam_repo_id,
                    filename=filename,
                    endpoint=args.hf_endpoint,
                    local_files_only=args.sam_local_files_only,
                )
            )
        )
    return paths[0].parent


class Sam3OnnxSegmenter:
    """Segment images with SAM3 via ONNX Runtime.

    Uses a vision encoder + prompt encoder/mask decoder ONNX model pair.
    Supports batched point prompts for multiple label groups.
    """

    def __init__(
        self, model_dir: Path, dtype: str, providers: list[str], prompt_batch_size: int
    ) -> None:
        self.model_dir = model_dir
        self.dtype = dtype
        self.prompt_batch_size = prompt_batch_size
        self.processor = Sam3TrackerProcessor.from_pretrained(
            str(model_dir), local_files_only=True
        )
        onnx_root = model_dir / "onnx"
        self.vision_session = ort.InferenceSession(
            str(onnx_root / f"vision_encoder_{dtype}.onnx"), providers=providers
        )
        self.prompt_session = ort.InferenceSession(
            str(onnx_root / f"prompt_encoder_mask_decoder_{dtype}.onnx"),
            providers=providers,
        )

    def segment_label_groups(
        self, image: Image.Image, groups: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run SAM3 on an image for each label group and return masks with metadata."""
        input_points = [[group["points_xy"] for group in groups]]
        input_labels = [[[1] * len(group["points_xy"]) for group in groups]]
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].detach().cpu().numpy().astype(np.float32)
        image_embeddings = self.vision_session.run(None, {"pixel_values": pixel_values})
        point_tensor = inputs["input_points"].squeeze(0)
        label_tensor = inputs["input_labels"].squeeze(0)
        original_size = tuple(int(x) for x in inputs["original_sizes"][0].tolist())
        results: list[dict[str, Any]] = []
        for start in range(0, len(groups), self.prompt_batch_size):
            end = min(start + self.prompt_batch_size, len(groups))
            chunk_size = end - start
            feed: dict[str, np.ndarray] = {
                "input_points": point_tensor[start:end]
                .unsqueeze(1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "input_labels": label_tensor[start:end]
                .unsqueeze(1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                "input_boxes": np.zeros((chunk_size, 0, 4), dtype=np.float32),
            }
            for embedding_index, embedding in enumerate(image_embeddings):
                feed[f"image_embeddings.{embedding_index}"] = (
                    np.repeat(embedding, chunk_size, axis=0).astype(np.float32)
                )
            iou_scores, pred_masks, object_score_logits = self.prompt_session.run(
                None, feed
            )
            for chunk_index in range(chunk_size):
                best = int(np.argmax(iou_scores[chunk_index, 0]))
                import torch

                selected_masks = [
                    torch.from_numpy(pred_masks[chunk_index : chunk_index + 1, 0:1, best])
                ]
                selected_meta = {
                    "selected_mask_index": best,
                    "iou_scores": [
                        float(x) for x in iou_scores[chunk_index, 0].tolist()
                    ],
                    "selected_iou_score": float(
                        iou_scores[chunk_index, 0, best]
                    ),
                    "object_score_logit": float(
                        object_score_logits[chunk_index, 0, 0]
                    ),
                }
                resized = self.processor.post_process_masks(
                    selected_masks, [original_size]
                )
                group_index = start + chunk_index
                results.append(
                    {
                        "group": groups[group_index],
                        "mask": resized[0][0, 0].detach().cpu().numpy().astype(bool),
                        **selected_meta,
                    }
                )
        return results


def make_label_groups(objects: list[ObjectRecord]) -> list[dict[str, Any]]:
    """Group ObjectRecords by same_class_key for SAM3 batch inference."""
    by_key: dict[str, list[ObjectRecord]] = defaultdict(list)
    for obj in objects:
        by_key[obj.same_class_key].append(obj)
    groups = []
    for index, (key, items) in enumerate(sorted(by_key.items())):
        points: list[list[float]] = []
        for obj in items:
            points.extend(obj.points_xy)
        groups.append(
            {
                "label_group_id": f"label_{index:03d}",
                "same_class_key": key,
                "label_display": items[0].label_display,
                "object_ids": [obj.object_id for obj in items],
                "points_xy": points,
            }
        )
    return groups


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask with an elliptical kernel."""
    if radius <= 0:
        return mask.astype(bool)
    import cv2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: list[list[float]],
    color: tuple[int, int, int],
    radius: int = 6,
) -> None:
    """Draw point annotations on a PIL ImageDraw surface."""
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(20, 20, 20),
            width=2,
        )
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255))


def make_segmentation_preview(
    image: Image.Image,
    semantic: Image.Image,
    overlay: Image.Image,
    groups: list[dict[str, Any]],
) -> Image.Image:
    """Build a side-by-side preview with original image, semantic map, overlay, and legend."""
    legend_width = 520
    width = image.width + semantic.width + overlay.width + legend_width + 36
    height = max(image.height, 48 + 28 * len(groups))
    canvas = Image.new("RGB", (width, height), (246, 246, 240))
    x = 0
    canvas.paste(image, (x, 0))
    x += image.width + 12
    canvas.paste(semantic, (x, 0))
    x += semantic.width + 12
    canvas.paste(overlay, (x, 0))
    x += overlay.width + 12
    draw = ImageDraw.Draw(canvas)
    title = load_font(22)
    font = load_font(17)
    draw.text((x, 16), "SAM3 label masks", fill=(20, 20, 20), font=title)
    y = 54
    for group in groups:
        color = tuple(group["color_rgb"])
        draw.rounded_rectangle((x, y + 4, x + 24, y + 24), radius=4, fill=color)
        text = f"{group['label_group_id']} {group['label_display']} objs={len(group['object_ids'])} pts={len(group['points_xy'])}"
        draw.text((x + 34, y + 2), text, fill=(30, 30, 30), font=font)
        y += 28
    return canvas


def build_sam_masks(
    args: argparse.Namespace, paths: Paths, object_df: "pd.DataFrame | None" = None
) -> None:
    """Run SAM3 on every image in object_df and save masks + previews.

    Saves per-label-group mask PNGs (binary and dilated), a semantic RGB map,
    an overlay with point annotations, and a combined preview.
    """
    import pandas as pd

    if object_df is None:
        object_df = pd.read_parquet(paths.filtered / "object_records.parquet")
    objects_by_image = objects_from_df(object_df)
    model_dir = ensure_sam_model_dir(args)
    segmenter = Sam3OnnxSegmenter(
        model_dir, args.sam_dtype, args.sam_providers, args.prompt_batch_size
    )
    index_rows = []
    for image_index, (image_id, objects) in enumerate(objects_by_image.items()):
        image_path = Path(objects[0].image_path)
        image = Image.open(image_path).convert("RGB")
        groups = make_label_groups(objects)
        segs = segmenter.segment_label_groups(image, groups)
        image_dir = paths.sam_masks / image_id
        mask_dir = image_dir / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        semantic = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        overlay = np.asarray(image).copy()
        meta_groups = []
        for idx, seg in enumerate(segs):
            group = seg["group"]
            color = color_for_index(idx)
            mask = seg["mask"]
            dilated = dilate_mask(mask, args.mask_dilate_px)
            semantic[mask] = color
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.55
                + np.asarray(color, dtype=np.float32) * 0.45
            ).astype(np.uint8)
            mask_path = (
                mask_dir
                / f"{group['label_group_id']}_{safe_name(group['label_display'])}.png"
            )
            dilated_path = (
                mask_dir
                / f"{group['label_group_id']}_{safe_name(group['label_display'])}_dilated.png"
            )
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            Image.fromarray(dilated.astype(np.uint8) * 255, mode="L").save(
                dilated_path
            )
            meta_groups.append(
                {
                    **{k: v for k, v in group.items() if k != "points_xy"},
                    "points_xy": group["points_xy"],
                    "mask_path": str(mask_path),
                    "dilated_mask_path": str(dilated_path),
                    "area_pixels": int(mask.sum()),
                    "dilated_area_pixels": int(dilated.sum()),
                    "color_rgb": list(color),
                    "selected_iou_score": seg["selected_iou_score"],
                    "object_score_logit": seg["object_score_logit"],
                }
            )
        semantic_img = Image.fromarray(semantic)
        overlay_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_img)
        for group in meta_groups:
            draw_points(draw, group["points_xy"], tuple(group["color_rgb"]), radius=6)
        semantic_img.save(image_dir / "semantic.png")
        overlay_img.save(image_dir / "overlay.png")
        preview = make_segmentation_preview(image, semantic_img, overlay_img, meta_groups)
        preview_path = image_dir / "preview.png"
        preview.save(preview_path)
        write_json(
            image_dir / "masks.json",
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "width": image.width,
                "height": image.height,
                "sam_repo_id": args.sam_repo_id,
                "sam_dtype": args.sam_dtype,
                "mask_dilate_px": args.mask_dilate_px,
                "label_groups": meta_groups,
            },
        )
        index_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "num_objects": len(objects),
                "num_label_groups": len(groups),
                "preview_path": str(preview_path),
            }
        )
        print(
            f"[sam] {image_index:03d} {image_id} objects={len(objects)} labels={len(groups)}"
        )
    pd.DataFrame(index_rows).to_csv(paths.sam_masks / "sam_mask_index.csv", index=False)


def load_mask(path: str) -> np.ndarray:
    """Load a binary mask from a PNG file."""
    return np.asarray(Image.open(path).convert("L")) > 0


def load_sam_meta(paths: Paths, image_id: str) -> dict[str, Any]:
    """Load the masks.json metadata for an image."""
    return json.loads((paths.sam_masks / image_id / "masks.json").read_text(encoding="utf-8"))


def point_in_mask(point_xy: list[float], mask: np.ndarray) -> bool:
    """Check whether a point falls inside a binary mask."""
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def point_coverage_ratio(points_xy: list[list[float]], mask: np.ndarray) -> float:
    """Fraction of points that fall within the mask."""
    if not points_xy:
        return 0.0
    covered = sum(1 for point_xy in points_xy if point_in_mask(point_xy, mask))
    return covered / max(1, len(points_xy))


def raster_line_mask(
    shape: tuple[int, int], points: list[list[float]], width: int
) -> np.ndarray:
    """Rasterize a polyline as a binary mask."""
    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, 1, thickness=max(1, width))
    return mask.astype(bool)


def connected_components_count(mask: np.ndarray) -> int:
    """Count connected components (8-connectivity) in a binary mask."""
    import cv2
    num, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(num - 1)


def combine_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Union of multiple binary masks."""
    if not masks:
        return np.zeros(shape, dtype=bool)
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return union


def path_is_unambiguous(
    path_mask: np.ndarray,
    target_union_mask: np.ndarray,
    other_groups: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Check whether a path from anchor to target is unobstructed.

    A path is unambiguous if:
    - After removing the target mask, the path has at most 1 connected component.
    - Other-object masks cover at most 2 pixels of the path.

    Returns (passed, debug_info).
    """
    remainder = path_mask & ~target_union_mask
    count = connected_components_count(remainder)
    pixels = int(remainder.sum())
    other_hits = []
    other_union = np.zeros_like(path_mask, dtype=bool)
    for group in other_groups:
        hit_mask = path_mask & group["raw_mask"] & ~target_union_mask
        hit_pixels = int(hit_mask.sum())
        if hit_pixels > 0:
            other_hits.append(
                {
                    "same_class_key": group["same_class_key"],
                    "label_display": group["label_display"],
                    "hit_pixels": hit_pixels,
                }
            )
        other_union |= group["raw_mask"]
    other_hit_pixels = int((path_mask & other_union & ~target_union_mask).sum())
    passed = count <= 1 and other_hit_pixels <= 2
    return passed, {
        "same_class_remainder_components": count,
        "same_class_remainder_pixels": pixels,
        "other_object_hit_pixels": other_hit_pixels,
        "other_object_hit_count": len(other_hits),
        "other_object_hit_labels": [item["label_display"] for item in other_hits[:8]],
        "path_pixels": int(path_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Geometry helpers (anchor-relative)
# ---------------------------------------------------------------------------


def anchor_from_target_polar(
    target_xy: list[float],
    image_size: tuple[int, int],
    direction: str,
    rng: random.Random,
) -> tuple[list[float], float, float]:
    """Generate an anchor point opposite the given direction from the target.

    Returns (anchor_xy, angle_deg, radius) or (empty_list, angle, radius) if out of bounds.
    """
    width, height = image_size
    base_angle = DIRECTION_ANGLES[direction]
    angle = (base_angle + rng.uniform(-10.0, 10.0)) % 360.0
    radius = rng.uniform(
        max(45.0, min(width, height) * 0.08), min(width, height) * 0.28
    )
    rad = math.radians(angle)
    x = target_xy[0] - math.cos(rad) * radius
    y = target_xy[1] - math.sin(rad) * radius
    if x < 8 or y < 8 or x > width - 8 or y > height - 8:
        return [], angle, radius
    return [round(x, 3), round(y, 3)], round(angle, 3), round(radius, 3)


def direction_from_anchor(anchor_xy: list[float], target_xy: list[float]) -> str:
    """Determine the cardinal/diagonal direction from anchor toward target."""
    dx = target_xy[0] - anchor_xy[0]
    dy = target_xy[1] - anchor_xy[1]
    deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    names = list(DIRECTION_ANGLES.keys())
    return min(
        names,
        key=lambda name: min(
            abs(deg - DIRECTION_ANGLES[name]), 360 - abs(deg - DIRECTION_ANGLES[name])
        ),
    )


def build_two_step_path(
    anchor: list[float], target: list[float], first_axis: str
) -> list[list[float]]:
    """Build an L-shaped path: anchor -> midpoint (first axis) -> target."""
    if first_axis in {"right", "left"}:
        mid = [target[0], anchor[1]]
    else:
        mid = [anchor[0], target[1]]
    return [anchor, mid, target]
