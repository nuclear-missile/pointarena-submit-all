#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3TrackerProcessor

try:
    from .task_logic import (
        axis_constrained_nearest_meta,
        direction_angle_deg,
        direction_matches,
        other_object_anchor_candidates,
    )
except ImportError:
    from task_logic import axis_constrained_nearest_meta, direction_angle_deg, direction_matches, other_object_anchor_candidates


ROOT = Path(__file__).resolve().parents[1]
MAKE_ROOT = ROOT.parent
DEFAULT_POINT_RECORDS = MAKE_ROOT / "guide1" / "outputs_full" / "01_filtered" / "point_records.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_TEMPLATE_LIBRARY = ROOT / "template_library_clean.json"
DEFAULT_SAM_REPO_ID = "onnx-community/sam3-tracker-ONNX"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SAM_DTYPE = "q4f16"

TASK_TYPES = [
    "move_until_axis_fixed",
    "move_until_axis_from_other_object",
    "move_then_axis_limited_fixed",
    "nearest_among_objects",
    "nearest_among_objects_axis_constrained",
]

TASK_ORDER = {name: index for index, name in enumerate(TASK_TYPES)}

DIRECTION_ANGLES = {
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

STUFF_WORDS = {
    "background",
    "wall",
    "floor",
    "ceiling",
    "sky",
    "ground",
    "road",
    "street",
    "sidewalk",
    "grass",
    "water",
    "sand",
    "snow",
    "table",
    "desk",
    "counter",
    "shelf",
    "shelves",
    "cabinet",
    "door",
    "window",
    "mirror",
    "stairs",
    "step",
    "carpet",
    "rug",
    "plate",
    "bowl",
    "tray",
    "screen",
    "monitor",
    "paper",
    "papers",
    "sink",
    "backsplash",
    "cupboard",
    "cupboards",
    "stovetop",
    "countertop",
    "cabinetry",
    "tile",
    "tiles",
    "pavement",
    "concrete",
    "cement",
    "wood",
    "wooden",
}

SPATIAL_WORDS = {
    "corner",
    "edge",
    "top",
    "bottom",
    "left",
    "right",
    "middle",
    "center",
    "upper",
    "lower",
    "front",
    "back",
    "side",
    "line",
    "area",
    "space",
}

TEXTLIKE_WORDS = {
    "text",
    "word",
    "words",
    "letter",
    "letters",
    "number",
    "numbers",
    "writing",
    "logo",
    "logos",
    "watermark",
    "copyright",
    "caption",
    "title",
    "headline",
    "font",
    "fonts",
    "icon",
    "icons",
    "symbol",
    "symbols",
}

ARTIFACT_WORDS = {
    "reflection",
    "reflections",
    "glare",
    "shadow",
    "shadows",
    "highlight",
    "highlights",
    "shine",
    "blur",
    "blurry",
    "texture",
    "textured",
    "lighting",
}

REGION_WORDS = {
    "surface",
    "area",
    "section",
    "piece",
    "pieces",
    "part",
    "parts",
    "spot",
    "spots",
    "mark",
    "marks",
    "line",
    "lines",
    "point",
    "points",
    "tip",
    "tips",
    "base",
    "center",
    "middle",
    "rim",
    "edge",
    "edges",
}

RELATIONAL_WORDS = {
    "of",
    "on",
    "in",
    "with",
    "behind",
    "under",
    "inside",
    "outside",
    "between",
    "from",
    "near",
    "around",
    "across",
    "through",
}

FUNCTIONAL_WORDS = {
    "where",
    "used",
    "use",
    "would",
    "hold",
    "holds",
    "holding",
    "go",
    "goes",
    "open",
    "close",
    "drink",
    "read",
    "write",
    "sit",
    "stand",
}

BAD_HEAD_WORDS = TEXTLIKE_WORDS | ARTIFACT_WORDS | REGION_WORDS | {
    "thing",
    "things",
    "object",
    "objects",
    "item",
    "items",
    "image",
    "photo",
    "picture",
    "scene",
}

ATTRIBUTE_ONLY_WORDS = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "brown",
    "orange",
    "pink",
    "purple",
    "gold",
    "silver",
    "gray",
    "grey",
    "sharp",
    "silky",
    "curious",
    "furry",
    "fur",
}

ABSTRACT_OR_CATEGORY_WORDS = {
    "food",
    "meal",
    "dinner",
    "supper",
    "lunch",
    "breakfast",
    "cuisine",
    "time",
    "brand",
    "drawing",
    "painting",
    "artwork",
    "bubble",
    "horizon",
    "light",
    "female",
}


@dataclass
class Paths:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guide4 strict SAM3-filtered task pipeline.")
    parser.add_argument("--stage", choices=["full", "filter", "sam", "candidates", "rewrite", "visualize"], default="full")
    parser.add_argument("--point-records", type=Path, default=DEFAULT_POINT_RECORDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--template-library-json", type=Path, default=DEFAULT_TEMPLATE_LIBRARY)
    parser.add_argument("--templates-per-candidate", type=int, default=1)
    parser.add_argument("--limit-images", type=int, default=10)
    parser.add_argument("--target-per-task", type=int, default=20)
    parser.add_argument("--task-targets-json", type=str, default="")
    parser.add_argument("--image-ids-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sam-repo-id", type=str, default=DEFAULT_SAM_REPO_ID)
    parser.add_argument("--hf-endpoint", type=str, default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--sam-model-dir", type=Path, default=None)
    parser.add_argument("--sam-dtype", type=str, default=DEFAULT_SAM_DTYPE)
    parser.add_argument("--sam-local-files-only", action="store_true")
    parser.add_argument("--sam-providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--mask-dilate-px", type=int, default=5)
    parser.add_argument("--path-width-px", type=int, default=3)
    parser.add_argument("--max-objects-per-image", type=int, default=60)
    parser.add_argument("--enable-api-rewrite", action="store_true")
    parser.add_argument("--rewrite-api-key-env", type=str, default="GUIDE4_REWRITE_API_KEY")
    parser.add_argument("--rewrite-api-base-url", type=str, default="https://api.vectorengine.ai/v1")
    parser.add_argument("--rewrite-model", type=str, default="")
    parser.add_argument("--rewrite-timeout-s", type=int, default=120)
    parser.add_argument("--rewrite-retries", type=int, default=3)
    return parser.parse_args()


def resolve_task_targets(args: argparse.Namespace) -> dict[str, int]:
    payload: dict[str, Any] = {}
    raw = str(getattr(args, "task_targets_json", "") or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(raw)
    targets = {task: int(args.target_per_task) for task in TASK_TYPES}
    for task in TASK_TYPES:
        if task in payload:
            targets[task] = max(0, int(payload[task]))
    return targets


def ensure_paths(root: Path) -> Paths:
    paths = Paths(
        root=root,
        filtered=root / "01_filtered",
        sam_masks=root / "02_sam_masks",
        task_pool=root / "03_task_pool",
        rewrites=root / "04_rewrites",
        visuals=root / "05_visuals",
        final=root / "07_final",
        reports=root / "reports",
        cache=root / "_cache",
    )
    for value in paths.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)
    (paths.visuals / "candidates").mkdir(parents=True, exist_ok=True)
    (paths.visuals / "rewrites").mkdir(parents=True, exist_ok=True)
    return paths


def clean_label(text: str) -> str:
    value = text or ""
    value = value.replace("_", " ").replace("/", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def label_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_text(clean_label(text)))


def head_noun_from_label(text: str) -> str:
    tokens = [token for token in label_tokens(text) if token not in {"the", "a", "an"}]
    return tokens[-1] if tokens else clean_text(clean_label(text))


def normalize_key(text: str) -> str:
    value = clean_text(clean_label(text))
    value = re.sub(r"^(a|an|the)\s+", "", value)
    return value


def safe_name(text: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return value[:max_len].strip("._") or "object"


def stable_hash(payload: Any) -> int:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return int.from_bytes(__import__("hashlib").sha256(raw.encode("utf-8")).digest()[:8], "big")


def deterministic_rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(stable_hash({"seed": seed, "parts": list(parts)}))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def load_template_library(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [str(item).strip() for item in value if str(item).strip()]
    return out


def color_for_index(index: int) -> tuple[int, int, int]:
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
    for candidate in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def label_is_strict_instance_candidate(label: str) -> tuple[bool, str]:
    low = clean_text(clean_label(label))
    tokens = label_tokens(low)
    head = head_noun_from_label(low)
    if not low:
        return False, "empty_label"
    if len(low) > 70:
        return False, "label_too_long"
    if re.search(r"https?://|www\.|@|\.com|\\|[{}\[\]]", low):
        return False, "text_artifact_or_url"
    if re.search(r"[,'\"“”‘’:;()]", low):
        return False, "caption_like_punctuation"
    if low.isdigit():
        return False, "numeric_only_label"
    if re.fullmatch(r"[a-z]", low):
        return False, "single_character_label"
    if re.search(r"\d", low) and len(tokens) >= 2:
        return False, "numeric_phrase_label"
    if any(low.startswith(prefix) for prefix in ["where ", "what ", "which ", "who ", "how "]):
        return False, "qa_or_functional_phrase"
    words = set(tokens)
    if words & STUFF_WORDS:
        return False, "stuff_label"
    if words & SPATIAL_WORDS:
        return False, "spatial_label"
    if words & TEXTLIKE_WORDS:
        return False, "text_like_label"
    if words & ARTIFACT_WORDS:
        return False, "appearance_artifact_label"
    if head in BAD_HEAD_WORDS:
        return False, "generic_head_noun"
    if len(tokens) > 4:
        return False, "too_many_tokens"
    if len(tokens) == 1 and head in ATTRIBUTE_ONLY_WORDS:
        return False, "attribute_only_label"
    if len(tokens) == 1 and head in ABSTRACT_OR_CATEGORY_WORDS:
        return False, "abstract_category_label"
    if len(tokens) >= 2 and words & FUNCTIONAL_WORDS:
        return False, "functional_phrase"
    if len(tokens) >= 3 and words & RELATIONAL_WORDS:
        return False, "relational_phrase"
    if low in {"text", "words", "letters", "numbers", "logo", "background", "shadow", "shadows", "glare"}:
        return False, "generic_non_instance"
    return True, "keep"


def collect_first_images(point_records: Path, limit_images: int, batch_size: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    pf = pq.ParquetFile(point_records)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["image_id"]):
        for row in batch.to_pylist():
            image_id = str(row.get("image_id") or "")
            if image_id and image_id not in seen:
                seen.add(image_id)
                ids.append(image_id)
                if len(ids) >= limit_images:
                    return ids
    return ids


def load_rows_for_images(point_records: Path, image_ids: list[str], batch_size: int) -> dict[str, list[dict[str, Any]]]:
    columns = [
        "image_id",
        "image_path",
        "image_width",
        "image_height",
        "row_id",
        "label_raw",
        "label_clean",
        "point_x_norm",
        "point_y_norm",
    ]
    selected = set(image_ids)
    rows: dict[str, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    pf = pq.ParquetFile(point_records)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            image_id = str(row.get("image_id") or "")
            if image_id in selected:
                rows[image_id].append(row)
    return rows


def filter_object_records(args: argparse.Namespace, paths: Paths) -> pd.DataFrame:
    if getattr(args, "image_ids_json", None):
        image_ids = [str(item) for item in json.loads(Path(args.image_ids_json).read_text(encoding="utf-8"))]
    else:
        image_ids = collect_first_images(args.point_records, args.limit_images, args.batch_size)
    rows_by_image = load_rows_for_images(args.point_records, image_ids, args.batch_size)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for image_order, image_id in enumerate(image_ids):
        rows = rows_by_image.get(image_id, [])
        if not rows:
            continue
        image_path = Path(str(rows[0]["image_path"]))
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("row_id") or "")].append(row)
        if len(grouped) > args.max_objects_per_image:
            dropped.append({"image_id": image_id, "row_id": "", "drop_reason": "too_many_raw_objects_in_image", "num_objects": len(grouped)})
            continue
        object_index = 0
        for row_id, group in grouped.items():
            label_raw = str(group[0].get("label_raw") or "")
            label_clean = str(group[0].get("label_clean") or clean_label(label_raw))
            keep, reason = label_is_strict_instance_candidate(label_raw or label_clean)
            points_xy: list[list[float]] = []
            points_norm: list[list[float]] = []
            for row in group:
                x = float(row["point_x_norm"])
                y = float(row["point_y_norm"])
                x = min(1.0, max(0.0, x))
                y = min(1.0, max(0.0, y))
                points_norm.append([round(x, 6), round(y, 6)])
                points_xy.append([round(x * width, 3), round(y * height, 3)])
            if not keep:
                dropped.append({"image_id": image_id, "row_id": row_id, "label_raw": label_raw, "drop_reason": reason, "point_count": len(points_xy)})
                continue
            if len(points_xy) > 8:
                dropped.append({"image_id": image_id, "row_id": row_id, "label_raw": label_raw, "drop_reason": "too_many_points_for_single_instance", "point_count": len(points_xy)})
                continue
            arr = np.asarray(points_norm, dtype=float)
            if len(points_xy) >= 3:
                spread = float(np.hypot(arr[:, 0].max() - arr[:, 0].min(), arr[:, 1].max() - arr[:, 1].min()))
                if spread > 0.25:
                    dropped.append({"image_id": image_id, "row_id": row_id, "label_raw": label_raw, "drop_reason": "multi_point_spread_too_large", "point_count": len(points_xy), "spread_diag_norm": spread})
                    continue
            center_norm = arr.mean(axis=0)
            object_id = f"{image_order:04d}_{object_index:03d}"
            object_index += 1
            kept.append(
                {
                    "object_id": object_id,
                    "image_order": image_order,
                    "object_order": object_index - 1,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "row_id": row_id,
                    "label_raw": label_raw,
                    "label_clean": label_clean,
                    "label_display": label_raw.strip() or label_clean,
                    "same_class_key": normalize_key(label_clean),
                    "head_noun": head_noun_from_label(label_clean),
                    "point_count": len(points_xy),
                    "points_xy_json": json.dumps(points_xy),
                    "points_norm_json": json.dumps(points_norm),
                    "center_x": round(float(center_norm[0] * width), 3),
                    "center_y": round(float(center_norm[1] * height), 3),
                    "center_x_norm": round(float(center_norm[0]), 6),
                    "center_y_norm": round(float(center_norm[1]), 6),
                }
            )
    kept_df = pd.DataFrame(kept)
    dropped_df = pd.DataFrame(dropped)
    kept_df.to_parquet(paths.filtered / "object_records.parquet", index=False)
    dropped_df.to_parquet(paths.filtered / "dropped_objects.parquet", index=False)
    summary = {
        "input_point_records": str(args.point_records),
        "limit_images": args.limit_images,
        "kept_objects": int(len(kept_df)),
        "dropped_objects": int(len(dropped_df)),
        "drop_reasons": dropped_df["drop_reason"].value_counts().to_dict() if len(dropped_df) else {},
    }
    write_json(paths.reports / "filter_summary.json", summary)
    return kept_df


def objects_from_df(df: pd.DataFrame) -> dict[str, list[ObjectRecord]]:
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


def ensure_sam_model_dir(args: argparse.Namespace) -> Path:
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
    def __init__(self, model_dir: Path, dtype: str, providers: list[str], prompt_batch_size: int) -> None:
        self.model_dir = model_dir
        self.dtype = dtype
        self.prompt_batch_size = prompt_batch_size
        self.processor = Sam3TrackerProcessor.from_pretrained(str(model_dir), local_files_only=True)
        onnx_root = model_dir / "onnx"
        self.vision_session = ort.InferenceSession(str(onnx_root / f"vision_encoder_{dtype}.onnx"), providers=providers)
        self.prompt_session = ort.InferenceSession(str(onnx_root / f"prompt_encoder_mask_decoder_{dtype}.onnx"), providers=providers)

    def segment_label_groups(self, image: Image.Image, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        input_points = [[group["points_xy"] for group in groups]]
        input_labels = [[[1] * len(group["points_xy"]) for group in groups]]
        inputs = self.processor(images=image, input_points=input_points, input_labels=input_labels, return_tensors="pt")
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
                "input_points": point_tensor[start:end].unsqueeze(1).detach().cpu().numpy().astype(np.float32),
                "input_labels": label_tensor[start:end].unsqueeze(1).detach().cpu().numpy().astype(np.int64),
                "input_boxes": np.zeros((chunk_size, 0, 4), dtype=np.float32),
            }
            for embedding_index, embedding in enumerate(image_embeddings):
                feed[f"image_embeddings.{embedding_index}"] = np.repeat(embedding, chunk_size, axis=0).astype(np.float32)
            iou_scores, pred_masks, object_score_logits = self.prompt_session.run(None, feed)
            selected_masks = []
            selected_meta = []
            for chunk_index in range(chunk_size):
                best = int(np.argmax(iou_scores[chunk_index, 0]))
                selected_masks.append(torch.from_numpy(pred_masks[chunk_index : chunk_index + 1, 0:1, best]))
                selected_meta.append(
                    {
                        "selected_mask_index": best,
                        "iou_scores": [float(x) for x in iou_scores[chunk_index, 0].tolist()],
                        "selected_iou_score": float(iou_scores[chunk_index, 0, best]),
                        "object_score_logit": float(object_score_logits[chunk_index, 0, 0]),
                    }
                )
            resized = self.processor.post_process_masks(selected_masks, [original_size] * len(selected_masks))
            for local_index, mask_tensor in enumerate(resized):
                group_index = start + local_index
                results.append(
                    {
                        "group": groups[group_index],
                        "mask": mask_tensor[0, 0].detach().cpu().numpy().astype(bool),
                        **selected_meta[local_index],
                    }
                )
        return results


def make_label_groups(objects: list[ObjectRecord]) -> list[dict[str, Any]]:
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


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def draw_points(draw: ImageDraw.ImageDraw, points: list[list[float]], color: tuple[int, int, int], radius: int = 6) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 20, 20), width=2)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255))


def build_sam_masks(args: argparse.Namespace, paths: Paths, object_df: pd.DataFrame | None = None) -> None:
    if object_df is None:
        object_df = pd.read_parquet(paths.filtered / "object_records.parquet")
    objects_by_image = objects_from_df(object_df)
    model_dir = ensure_sam_model_dir(args)
    segmenter = Sam3OnnxSegmenter(model_dir, args.sam_dtype, args.sam_providers, args.prompt_batch_size)
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
            overlay[mask] = (overlay[mask].astype(np.float32) * 0.55 + np.asarray(color, dtype=np.float32) * 0.45).astype(np.uint8)
            mask_path = mask_dir / f"{group['label_group_id']}_{safe_name(group['label_display'])}.png"
            dilated_path = mask_dir / f"{group['label_group_id']}_{safe_name(group['label_display'])}_dilated.png"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            Image.fromarray(dilated.astype(np.uint8) * 255, mode="L").save(dilated_path)
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
        for idx, group in enumerate(meta_groups):
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
        index_rows.append({"image_id": image_id, "image_path": str(image_path), "num_objects": len(objects), "num_label_groups": len(groups), "preview_path": str(preview_path)})
        print(f"[sam] {image_index:03d} {image_id} objects={len(objects)} labels={len(groups)}")
    pd.DataFrame(index_rows).to_csv(paths.sam_masks / "sam_mask_index.csv", index=False)


def make_segmentation_preview(image: Image.Image, semantic: Image.Image, overlay: Image.Image, groups: list[dict[str, Any]]) -> Image.Image:
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


def load_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def load_sam_meta(paths: Paths, image_id: str) -> dict[str, Any]:
    return json.loads((paths.sam_masks / image_id / "masks.json").read_text(encoding="utf-8"))


def point_in_mask(point_xy: list[float], mask: np.ndarray) -> bool:
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def point_coverage_ratio(points_xy: list[list[float]], mask: np.ndarray) -> float:
    if not points_xy:
        return 0.0
    covered = sum(1 for point_xy in points_xy if point_in_mask(point_xy, mask))
    return covered / max(1, len(points_xy))


def anchor_from_target_polar(target_xy: list[float], image_size: tuple[int, int], direction: str, rng: random.Random) -> tuple[list[float], float, float]:
    width, height = image_size
    base_angle = DIRECTION_ANGLES[direction]
    angle = (base_angle + rng.uniform(-10.0, 10.0)) % 360.0
    radius = rng.uniform(max(45.0, min(width, height) * 0.08), min(width, height) * 0.28)
    rad = math.radians(angle)
    # Direction is from anchor to target, so anchor is the opposite vector from target.
    x = target_xy[0] - math.cos(rad) * radius
    y = target_xy[1] - math.sin(rad) * radius
    if x < 8 or y < 8 or x > width - 8 or y > height - 8:
        return [], angle, radius
    return [round(x, 3), round(y, 3)], round(angle, 3), round(radius, 3)


def direction_from_anchor(anchor_xy: list[float], target_xy: list[float]) -> str:
    dx = target_xy[0] - anchor_xy[0]
    dy = target_xy[1] - anchor_xy[1]
    deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    names = list(DIRECTION_ANGLES.keys())
    return min(names, key=lambda name: min(abs(deg - DIRECTION_ANGLES[name]), 360 - abs(deg - DIRECTION_ANGLES[name])))


def raster_line_mask(shape: tuple[int, int], points: list[list[float]], width: int) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, 1, thickness=max(1, width))
    return mask.astype(bool)


def connected_components_count(mask: np.ndarray) -> int:
    import cv2

    num, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(num - 1)


def combine_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
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


def build_two_step_path(anchor: list[float], target: list[float], first_axis: str) -> list[list[float]]:
    if first_axis in {"right", "left"}:
        mid = [target[0], anchor[1]]
    else:
        mid = [anchor[0], target[1]]
    return [anchor, mid, target]


def nearest_candidate_meta(obj: ObjectRecord, same_label_objects: list[ObjectRecord], anchor_xy: list[float]) -> dict[str, Any]:
    distances = [(other.object_id, math.hypot(other.center_xy[0] - anchor_xy[0], other.center_xy[1] - anchor_xy[1])) for other in same_label_objects]
    distances.sort(key=lambda item: item[1])
    if not distances:
        return {"pass": False, "reason": "no_same_class_objects"}
    closest_id, closest_dist = distances[0]
    second_dist = distances[1][1] if len(distances) > 1 else float("inf")
    margin = second_dist - closest_dist
    min_margin = max(12.0, closest_dist * 0.12)
    return {
        "pass": closest_id == obj.object_id and margin >= min_margin,
        "closest_id": closest_id,
        "closest_dist": round(float(closest_dist), 3),
        "second_dist": round(float(second_dist), 3) if math.isfinite(second_dist) else None,
        "margin_px": round(float(margin), 3) if math.isfinite(margin) else None,
        "min_margin_px": round(float(min_margin), 3),
    }


def candidate_id(task_type: str, image_id: str, object_id: str, direction: str) -> str:
    return f"{task_type}::{image_id}::{object_id}::{direction}"


def build_candidates(args: argparse.Namespace, paths: Paths, object_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if object_df is None:
        object_df = pd.read_parquet(paths.filtered / "object_records.parquet")
    objects_by_image = objects_from_df(object_df)
    rows: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    task_counts = Counter()
    task_targets = resolve_task_targets(args)

    for image_order, (image_id, objects) in enumerate(objects_by_image.items()):
        if all(task_counts[t] >= task_targets[t] for t in TASK_TYPES):
            break
        sam_meta = load_sam_meta(paths, image_id)
        mask_entries = []
        for group in sam_meta["label_groups"]:
            mask_entries.append(
                {
                    "same_class_key": group["same_class_key"],
                    "label_display": group["label_display"],
                    "object_ids": list(group.get("object_ids", [])),
                    "raw_mask": load_mask(group["mask_path"]),
                    "dilated_mask": load_mask(group["dilated_mask_path"]),
                    "area_pixels": int(group["area_pixels"]),
                    "dilated_area_pixels": int(group["dilated_area_pixels"]),
                }
            )
        masks_by_key = {group["same_class_key"]: group["dilated_mask"] for group in mask_entries}
        raw_masks_by_key = {group["same_class_key"]: group["raw_mask"] for group in mask_entries}
        meta_by_key = {group["same_class_key"]: group for group in mask_entries}
        by_key: dict[str, list[ObjectRecord]] = defaultdict(list)
        for obj in objects:
            by_key[obj.same_class_key].append(obj)
        object_payloads = [
            {
                "object_id": other.object_id,
                "same_class_key": other.same_class_key,
                "label_display": other.label_display,
                "center_xy": list(other.center_xy),
                "object_order": other.object_order,
            }
            for other in objects
        ]
        for obj in objects:
            target_xy = obj.center_xy
            target_mask = masks_by_key.get(obj.same_class_key)
            raw_target_mask = raw_masks_by_key.get(obj.same_class_key)
            if target_mask is None or raw_target_mask is None:
                drops.append({"image_id": image_id, "object_id": obj.object_id, "drop_reason": "missing_sam_mask"})
                continue
            if not point_in_mask(target_xy, target_mask):
                drops.append({"image_id": image_id, "object_id": obj.object_id, "drop_reason": "target_center_not_in_sam_mask", "task_type": "all"})
                continue
            raw_point_coverage = point_coverage_ratio(obj.points_xy, raw_target_mask)
            dilated_point_coverage = point_coverage_ratio(obj.points_xy, target_mask)
            if raw_point_coverage < 0.5 or dilated_point_coverage < 1.0:
                drops.append(
                    {
                        "image_id": image_id,
                        "object_id": obj.object_id,
                        "task_type": "all",
                        "drop_reason": "sam_mask_does_not_cover_object_points",
                        "raw_point_coverage": round(raw_point_coverage, 3),
                        "dilated_point_coverage": round(dilated_point_coverage, 3),
                    }
                )
                continue
            other_groups = [group for group in mask_entries if group["same_class_key"] != obj.same_class_key]
            if task_counts["move_until_axis_fixed"] < task_targets["move_until_axis_fixed"]:
                for direction in MOVE_DIRECTIONS:
                    rng = deterministic_rng(args.seed, "move_until", image_id, obj.object_id, direction)
                    accepted = try_direction_candidate(
                        args,
                        obj,
                        image_id,
                        target_xy,
                        target_mask,
                        raw_target_mask,
                        other_groups,
                        raw_point_coverage,
                        dilated_point_coverage,
                        direction,
                        rng,
                        task_type="move_until_axis_fixed",
                        two_step=False,
                    )
                    if accepted:
                        rows.append(accepted)
                        task_counts["move_until_axis_fixed"] += 1
                        break
            if task_counts["move_until_axis_from_other_object"] < task_targets["move_until_axis_from_other_object"]:
                direction_order = MOVE_DIRECTIONS[:]
                rng0 = deterministic_rng(args.seed, "move_until_other_order", image_id, obj.object_id)
                rng0.shuffle(direction_order)
                for direction in direction_order:
                    rng = deterministic_rng(args.seed, "move_until_other", image_id, obj.object_id, direction)
                    accepted = try_other_object_direction_candidate(
                        args,
                        obj,
                        image_id,
                        target_xy,
                        target_mask,
                        raw_target_mask,
                        other_groups,
                        object_payloads,
                        raw_point_coverage,
                        dilated_point_coverage,
                        direction,
                        rng,
                    )
                    if accepted:
                        rows.append(accepted)
                        task_counts["move_until_axis_from_other_object"] += 1
                        break
            if task_counts["move_then_axis_limited_fixed"] < task_targets["move_then_axis_limited_fixed"]:
                diagonal_order = DIAGONAL_DIRECTIONS[:]
                rng0 = deterministic_rng(args.seed, "move_then_order", image_id, obj.object_id)
                rng0.shuffle(diagonal_order)
                for direction in diagonal_order:
                    rng = deterministic_rng(args.seed, "move_then", image_id, obj.object_id, direction)
                    accepted = try_direction_candidate(
                        args,
                        obj,
                        image_id,
                        target_xy,
                        target_mask,
                        raw_target_mask,
                        other_groups,
                        raw_point_coverage,
                        dilated_point_coverage,
                        direction,
                        rng,
                        task_type="move_then_axis_limited_fixed",
                        two_step=True,
                    )
                    if accepted:
                        rows.append(accepted)
                        task_counts["move_then_axis_limited_fixed"] += 1
                        break
            if task_counts["nearest_among_objects"] < task_targets["nearest_among_objects"]:
                same = by_key[obj.same_class_key]
                if 2 <= len(same) < 5:
                    accepted = try_nearest_candidate(args, obj, same, target_mask, raw_point_coverage, dilated_point_coverage)
                    if accepted:
                        rows.append(accepted)
                        task_counts["nearest_among_objects"] += 1
                else:
                    drops.append({"image_id": image_id, "object_id": obj.object_id, "task_type": "nearest_among_objects", "drop_reason": "same_object_count_not_2_to_4", "same_count": len(same)})
            if task_counts["nearest_among_objects_axis_constrained"] < task_targets["nearest_among_objects_axis_constrained"]:
                same = by_key[obj.same_class_key]
                if 2 <= len(same) < 5:
                    direction_order = AXIS_DIRECTIONS[:]
                    rng0 = deterministic_rng(args.seed, "nearest_axis_order", image_id, obj.object_id)
                    rng0.shuffle(direction_order)
                    for direction in direction_order:
                        accepted = try_axis_constrained_nearest_candidate(
                            args,
                            obj,
                            same,
                            target_mask,
                            raw_point_coverage,
                            dilated_point_coverage,
                            direction,
                        )
                        if accepted:
                            rows.append(accepted)
                            task_counts["nearest_among_objects_axis_constrained"] += 1
                            break
                else:
                    drops.append(
                        {
                            "image_id": image_id,
                            "object_id": obj.object_id,
                            "task_type": "nearest_among_objects_axis_constrained",
                            "drop_reason": "same_object_count_not_2_to_4",
                            "same_count": len(same),
                        }
                    )

    candidate_df = pd.DataFrame(rows)
    if len(candidate_df):
        candidate_df = (
            candidate_df.drop_duplicates(subset=["candidate_id"])
            .sort_values(["image_order", "object_order", "task_rank", "direction_text", "candidate_id"])
            .reset_index(drop=True)
        )
    dropped_df = pd.DataFrame(drops)
    candidate_df.to_parquet(paths.task_pool / "task_candidates.parquet", index=False)
    dropped_df.to_parquet(paths.task_pool / "dropped_task_candidates.parquet", index=False)
    write_json(
        paths.reports / "candidate_summary.json",
        {
            "task_targets": task_targets,
            "accepted_by_task": dict(Counter(candidate_df["task_type"]) if len(candidate_df) else {}),
            "dropped_reasons": dropped_df["drop_reason"].value_counts().to_dict() if len(dropped_df) else {},
        },
    )
    return candidate_df


def try_direction_candidate(
    args: argparse.Namespace,
    obj: ObjectRecord,
    image_id: str,
    target_xy: list[float],
    target_mask: np.ndarray,
    raw_target_mask: np.ndarray,
    other_groups: list[dict[str, Any]],
    raw_point_coverage: float,
    dilated_point_coverage: float,
    direction: str,
    rng: random.Random,
    task_type: str,
    two_step: bool,
) -> dict[str, Any] | None:
    for attempt in range(30):
        anchor, angle, radius = anchor_from_target_polar(target_xy, (obj.width, obj.height), direction, rng)
        if not anchor:
            continue
        if point_in_mask(anchor, target_mask):
            continue
        if two_step:
            first_axis = "right" if "right" in direction else "left"
            if "up" in direction or "down" in direction:
                first_axis = rng.choice([first_axis, "up" if "up" in direction else "down"])
            path_points = build_two_step_path(anchor, target_xy, first_axis)
        else:
            first_axis = direction
            path_points = [anchor, target_xy]
        path_mask = raster_line_mask((obj.height, obj.width), path_points, args.path_width_px)
        pass_path, path_meta = path_is_unambiguous(path_mask, raw_target_mask, other_groups)
        if not pass_path:
            continue
        question_templates = template_questions(
            args,
            task_type,
            obj.label_display,
            direction,
            first_axis,
            image_id,
            obj.object_id,
        )
        return {
            "candidate_id": candidate_id(task_type, image_id, obj.object_id, direction),
            "task_type": task_type,
            "task_rank": TASK_ORDER[task_type],
            "image_id": image_id,
            "image_path": obj.image_path,
            "image_order": obj.image_order,
            "object_id": obj.object_id,
            "object_order": obj.object_order,
            "target_label": obj.label_display,
            "same_class_key": obj.same_class_key,
            "target_point": [round(target_xy[0] / obj.width, 6), round(target_xy[1] / obj.height, 6)],
            "target_point_xy": target_xy,
            "anchor_point": [round(anchor[0] / obj.width, 6), round(anchor[1] / obj.height, 6)],
            "anchor_point_xy": anchor,
            "direction_text": direction,
            "first_axis": first_axis,
            "anchor_angle_deg": angle,
            "anchor_radius_px": radius,
            "path_points_xy_json": json.dumps(path_points),
            "path_judge_json": json.dumps(
                {
                    "pass": True,
                    "raw_point_coverage": round(raw_point_coverage, 3),
                    "dilated_point_coverage": round(dilated_point_coverage, 3),
                    **path_meta,
                }
            ),
            "template_questions_json": json.dumps(question_templates, ensure_ascii=False),
            "judge_method": "sam3_anchor_exclusion_path_same_class_connectivity_and_other_object_blocking",
        }
    return None


def try_other_object_direction_candidate(
    args: argparse.Namespace,
    obj: ObjectRecord,
    image_id: str,
    target_xy: list[float],
    target_mask: np.ndarray,
    raw_target_mask: np.ndarray,
    other_groups: list[dict[str, Any]],
    object_payloads: list[dict[str, Any]],
    raw_point_coverage: float,
    dilated_point_coverage: float,
    direction: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    candidates = other_object_anchor_candidates(
        target_object_id=obj.object_id,
        target_same_class_key=obj.same_class_key,
        target_xy=target_xy,
        objects=object_payloads,
        direction=direction,
        tolerance_deg=10.0,
        require_unique_class=True,
    )
    if not candidates:
        return None
    order = list(range(len(candidates)))
    rng.shuffle(order)
    for index in order:
        anchor_obj = candidates[index]
        anchor = [round(float(anchor_obj["center_xy"][0]), 3), round(float(anchor_obj["center_xy"][1]), 3)]
        anchor_key = str(anchor_obj["same_class_key"])
        if point_in_mask(anchor, target_mask):
            continue
        filtered_other_groups = [
            group
            for group in other_groups
            if group["same_class_key"] not in {anchor_key, obj.same_class_key}
        ]
        path_points = [anchor, target_xy]
        path_mask = raster_line_mask((obj.height, obj.width), path_points, args.path_width_px)
        pass_path, path_meta = path_is_unambiguous(path_mask, raw_target_mask, filtered_other_groups)
        if not pass_path:
            continue
        question_templates = template_questions(
            args,
            "move_until_axis_from_other_object",
            obj.label_display,
            direction,
            direction,
            image_id,
            obj.object_id,
            anchor_label=str(anchor_obj["label_display"]),
        )
        return {
            "candidate_id": f"move_until_axis_from_other_object::{image_id}::{obj.object_id}::{anchor_obj['object_id']}::{direction}",
            "task_type": "move_until_axis_from_other_object",
            "task_rank": TASK_ORDER["move_until_axis_from_other_object"],
            "image_id": image_id,
            "image_path": obj.image_path,
            "image_order": obj.image_order,
            "object_id": obj.object_id,
            "object_order": obj.object_order,
            "target_label": obj.label_display,
            "same_class_key": obj.same_class_key,
            "target_point": [round(target_xy[0] / obj.width, 6), round(target_xy[1] / obj.height, 6)],
            "target_point_xy": target_xy,
            "anchor_point": [round(anchor[0] / obj.width, 6), round(anchor[1] / obj.height, 6)],
            "anchor_point_xy": anchor,
            "anchor_object_id": str(anchor_obj["object_id"]),
            "anchor_label": str(anchor_obj["label_display"]),
            "anchor_same_class_key": anchor_key,
            "direction_text": direction,
            "first_axis": direction,
            "anchor_angle_deg": round(direction_angle_deg(anchor, target_xy), 3),
            "anchor_radius_px": round(math.hypot(target_xy[0] - anchor[0], target_xy[1] - anchor[1]), 3),
            "path_points_xy_json": json.dumps(path_points),
            "path_judge_json": json.dumps(
                {
                    "pass": True,
                    "anchor_label": str(anchor_obj["label_display"]),
                    "anchor_object_id": str(anchor_obj["object_id"]),
                    "anchor_source": "other_object_center",
                    "ignored_blocking_same_class_keys": [anchor_key],
                    "raw_point_coverage": round(raw_point_coverage, 3),
                    "dilated_point_coverage": round(dilated_point_coverage, 3),
                    **path_meta,
                }
            ),
            "template_questions_json": json.dumps(question_templates, ensure_ascii=False),
            "judge_method": "sam3_other_object_anchor_center_path_same_class_connectivity_and_other_object_blocking",
        }
    return None


def try_nearest_candidate(
    args: argparse.Namespace,
    obj: ObjectRecord,
    same: list[ObjectRecord],
    target_mask: np.ndarray,
    raw_point_coverage: float,
    dilated_point_coverage: float,
) -> dict[str, Any] | None:
    rng = deterministic_rng(args.seed, "nearest", obj.image_id, obj.object_id)
    for attempt in range(30):
        angle = rng.uniform(0.0, 360.0)
        radius = rng.uniform(max(35.0, min(obj.width, obj.height) * 0.05), min(obj.width, obj.height) * 0.18)
        anchor = [obj.center_xy[0] + math.cos(math.radians(angle)) * radius, obj.center_xy[1] + math.sin(math.radians(angle)) * radius]
        if anchor[0] < 8 or anchor[1] < 8 or anchor[0] > obj.width - 8 or anchor[1] > obj.height - 8:
            continue
        if point_in_mask(anchor, target_mask):
            continue
        nearest_meta = nearest_candidate_meta(obj, same, anchor)
        if not nearest_meta["pass"]:
            continue
        question_templates = template_questions(
            args,
            "nearest_among_objects",
            obj.label_display,
            "nearest",
            "",
            obj.image_id,
            obj.object_id,
        )
        distances = {other.object_id: round(math.hypot(other.center_xy[0] - anchor[0], other.center_xy[1] - anchor[1]), 3) for other in same}
        return {
            "candidate_id": f"nearest_among_objects::{obj.image_id}::{obj.object_id}",
            "task_type": "nearest_among_objects",
            "task_rank": TASK_ORDER["nearest_among_objects"],
            "image_id": obj.image_id,
            "image_path": obj.image_path,
            "image_order": obj.image_order,
            "object_id": obj.object_id,
            "object_order": obj.object_order,
            "target_label": obj.label_display,
            "same_class_key": obj.same_class_key,
            "target_point": [round(obj.center_xy[0] / obj.width, 6), round(obj.center_xy[1] / obj.height, 6)],
            "target_point_xy": obj.center_xy,
            "anchor_point": [round(anchor[0] / obj.width, 6), round(anchor[1] / obj.height, 6)],
            "anchor_point_xy": [round(anchor[0], 3), round(anchor[1], 3)],
            "direction_text": "nearest",
            "first_axis": "",
            "anchor_angle_deg": round(angle, 3),
            "anchor_radius_px": round(radius, 3),
            "path_points_xy_json": json.dumps([anchor, obj.center_xy]),
            "path_judge_json": json.dumps(
                {
                    "pass": True,
                    "same_count": len(same),
                    "nearest_distances": distances,
                    "raw_point_coverage": round(raw_point_coverage, 3),
                    "dilated_point_coverage": round(dilated_point_coverage, 3),
                    **nearest_meta,
                }
            ),
            "template_questions_json": json.dumps(question_templates, ensure_ascii=False),
            "judge_method": "programmatic_nearest_distance_margin_and_sam3_anchor_exclusion",
        }
    return None


def try_axis_constrained_nearest_candidate(
    args: argparse.Namespace,
    obj: ObjectRecord,
    same: list[ObjectRecord],
    target_mask: np.ndarray,
    raw_point_coverage: float,
    dilated_point_coverage: float,
    direction: str,
) -> dict[str, Any] | None:
    rng = deterministic_rng(args.seed, "nearest_axis", obj.image_id, obj.object_id, direction)
    same_payloads = [{"object_id": other.object_id, "center_xy": list(other.center_xy)} for other in same]
    for attempt in range(30):
        anchor, angle, radius = anchor_from_target_polar(obj.center_xy, (obj.width, obj.height), direction, rng)
        if not anchor:
            continue
        if point_in_mask(anchor, target_mask):
            continue
        nearest_meta = axis_constrained_nearest_meta(
            target_object_id=obj.object_id,
            target_xy=obj.center_xy,
            same_label_objects=same_payloads,
            anchor_xy=anchor,
            direction=direction,
            target_tolerance_deg=10.0,
            competitor_sector_deg=45.0,
        )
        if not nearest_meta["pass"]:
            continue
        question_templates = template_questions(
            args,
            "nearest_among_objects_axis_constrained",
            obj.label_display,
            direction,
            direction,
            obj.image_id,
            obj.object_id,
        )
        distances = {
            other.object_id: round(math.hypot(other.center_xy[0] - anchor[0], other.center_xy[1] - anchor[1]), 3)
            for other in same
        }
        return {
            "candidate_id": f"nearest_among_objects_axis_constrained::{obj.image_id}::{obj.object_id}::{direction}",
            "task_type": "nearest_among_objects_axis_constrained",
            "task_rank": TASK_ORDER["nearest_among_objects_axis_constrained"],
            "image_id": obj.image_id,
            "image_path": obj.image_path,
            "image_order": obj.image_order,
            "object_id": obj.object_id,
            "object_order": obj.object_order,
            "target_label": obj.label_display,
            "same_class_key": obj.same_class_key,
            "target_point": [round(obj.center_xy[0] / obj.width, 6), round(obj.center_xy[1] / obj.height, 6)],
            "target_point_xy": obj.center_xy,
            "anchor_point": [round(anchor[0] / obj.width, 6), round(anchor[1] / obj.height, 6)],
            "anchor_point_xy": [round(anchor[0], 3), round(anchor[1], 3)],
            "direction_text": direction,
            "first_axis": direction,
            "anchor_angle_deg": round(angle, 3),
            "anchor_radius_px": round(radius, 3),
            "path_points_xy_json": json.dumps([anchor, obj.center_xy]),
            "path_judge_json": json.dumps(
                {
                    "pass": True,
                    "same_count": len(same),
                    "nearest_distances": distances,
                    "raw_point_coverage": round(raw_point_coverage, 3),
                    "dilated_point_coverage": round(dilated_point_coverage, 3),
                    **nearest_meta,
                }
            ),
            "template_questions_json": json.dumps(question_templates, ensure_ascii=False),
            "judge_method": "directional_nearest_sector_and_sam3_anchor_exclusion",
        }
    return None


def fallback_template_questions(task_type: str, label: str, direction: str, first_axis: str, anchor_label: str = "") -> list[str]:
    phrase = direction_phrase(direction)
    if task_type == "move_until_axis_fixed":
        return [
            f"Point to the {label} {phrase} the blue point.",
            f"Use the blue point only as a reference and move {direction.replace('-', ' ')} until you reach the {label}.",
            f"Start at the blue point and keep going {direction.replace('-', ' ')} to the {label}.",
        ]
    if task_type == "move_until_axis_from_other_object":
        return [
            f"Point to the {label} {phrase} the {anchor_label}.",
            f"From the {anchor_label}, move {direction.replace('-', ' ')} until you reach the {label}.",
            f"Start at the {anchor_label} and keep going {direction.replace('-', ' ')} to the {label}.",
        ]
    if task_type == "nearest_among_objects":
        return [
            f"Point to the {label} nearest to the blue point.",
            f"Using the blue point only as a reference, select the closest {label}.",
            f"Find the {label} that is closest to the blue point.",
        ]
    if task_type == "nearest_among_objects_axis_constrained":
        return [
            f"Point to the nearest {label} {phrase} the blue point.",
            f"Find the closest {label} on the {direction.replace('-', ' ')} side of the blue point.",
            f"Using the blue point only as a reference, select the nearest {label} {phrase} it.",
        ]
    second_axis = second_axis_from_direction(direction, first_axis)
    return [
        f"From the blue point, move {first_axis}, then {second_axis}, to the {label}.",
        f"Use the blue point only as a starting reference: go {first_axis} first, then {second_axis}, and point to the {label}.",
        f"Point to the {label} reached by going {first_axis} from the blue point and then {second_axis}.",
    ]


def template_group_and_values(task_type: str, label: str, direction: str, first_axis: str, anchor_label: str = "") -> tuple[str | None, dict[str, str]]:
    second_axis = second_axis_from_direction(direction, first_axis)
    if task_type == "nearest_among_objects":
        return "nearest_among_objects_templates", {"label": label}
    if task_type == "nearest_among_objects_axis_constrained":
        return (
            "axis_constrained_nearest_templates",
            {
                "label": label,
                "axis_direction": direction,
                "axis_phrase": direction_phrase(direction),
            },
        )
    if task_type == "move_then_axis_limited_fixed":
        return (
            "two_step_axis_templates",
            {
                "label": label,
                "first_axis": first_axis,
                "second_axis": second_axis,
                "diagonal_relation": direction,
            },
        )
    if task_type == "move_until_axis_from_other_object":
        if direction in DIAGONAL_DIRECTIONS:
            return (
                "other_object_diagonal_templates",
                {
                    "label": label,
                    "anchor_label": anchor_label,
                    "diagonal_direction": direction,
                    "diagonal_phrase": direction_phrase(direction),
                },
            )
        return (
            "other_object_axis_templates",
            {
                "label": label,
                "anchor_label": anchor_label,
                "axis_direction": direction,
                "axis_phrase": direction_phrase(direction),
            },
        )
    if task_type == "move_until_axis_fixed":
        if direction in DIAGONAL_DIRECTIONS:
            return (
                "single_diagonal_templates",
                {
                    "label": label,
                    "diagonal_direction": direction,
                    "diagonal_phrase": direction_phrase(direction),
                },
            )
        return (
            "straight_axis_templates",
            {
                "label": label,
                "axis_direction": direction,
                "axis_phrase": direction_phrase(direction),
            },
        )
    return None, {"label": label}


def render_selected_templates(
    args: argparse.Namespace,
    group_name: str,
    values: dict[str, str],
    sample_key_parts: list[Any],
) -> list[str]:
    library = getattr(args, "template_library", {}) or {}
    source_templates = list(library.get(group_name, []))
    if not source_templates:
        return []
    max_count = max(1, int(args.templates_per_candidate))
    chosen_templates = source_templates
    if len(source_templates) > max_count:
        rng = deterministic_rng(args.seed, "template_select", group_name, *sample_key_parts)
        indices = list(range(len(source_templates)))
        rng.shuffle(indices)
        chosen_templates = [source_templates[index] for index in indices[:max_count]]
    rendered: list[str] = []
    seen: set[str] = set()
    for template in chosen_templates:
        try:
            question = template.format(**values).strip()
        except Exception:
            continue
        if question and question not in seen:
            seen.add(question)
            rendered.append(question)
    return rendered


def template_questions(
    args: argparse.Namespace,
    task_type: str,
    label: str,
    direction: str,
    first_axis: str,
    image_id: str,
    object_id: str,
    anchor_label: str = "",
) -> list[str]:
    group_name, values = template_group_and_values(task_type, label, direction, first_axis, anchor_label=anchor_label)
    if group_name:
        rendered = render_selected_templates(
            args,
            group_name,
            values,
            [task_type, image_id, object_id, direction, first_axis, label, anchor_label],
        )
        if rendered:
            return rendered
    fallback = fallback_template_questions(task_type, label, direction, first_axis, anchor_label=anchor_label)
    return fallback[: max(1, int(args.templates_per_candidate))]


def direction_phrase(direction: str) -> str:
    return {
        "right": "to the right of",
        "left": "to the left of",
        "up": "above",
        "down": "below",
        "up-right": "up and to the right of",
        "down-right": "down and to the right of",
        "down-left": "down and to the left of",
        "up-left": "up and to the left of",
    }.get(direction, direction)


def second_axis_from_direction(direction: str, first_axis: str) -> str:
    axes = []
    if "right" in direction:
        axes.append("right")
    if "left" in direction:
        axes.append("left")
    if "up" in direction:
        axes.append("up")
    if "down" in direction:
        axes.append("down")
    for axis in axes:
        if axis != first_axis:
            return axis
    return axes[0] if axes else first_axis


def rewrite_candidates(args: argparse.Namespace, paths: Paths, candidate_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if candidate_df is None:
        candidate_df = pd.read_parquet(paths.task_pool / "task_candidates.parquet")
    rows = []
    final_rows = []
    for candidate in candidate_df.to_dict(orient="records"):
        templates = json.loads(candidate["template_questions_json"])
        questions = list(templates)
        if args.enable_api_rewrite:
            questions.extend(call_rewrite_api(args, candidate, templates))
        for style_index, question in enumerate(questions):
            row = {
                "candidate_id": candidate["candidate_id"],
                "task_type": candidate["task_type"],
                "object_id": candidate["object_id"],
                "style_index": style_index,
                "question": question,
                "rewrite_source": "template" if style_index < len(templates) else "api",
            }
            rows.append(row)
            final_rows.append(
                {
                    "id": f"{candidate['candidate_id']}::{style_index}",
                    "candidate_id": candidate["candidate_id"],
                    "image_id": candidate["image_id"],
                    "image_path": candidate["image_path"],
                    "task_type": candidate["task_type"],
                    "question": question,
                    "target": candidate["target_point"],
                    "anchor": candidate["anchor_point"],
                    "meta": {
                        "target_label": candidate["target_label"],
                        "direction_text": candidate["direction_text"],
                        "first_axis": candidate.get("first_axis", ""),
                        "judge_method": candidate["judge_method"],
                        "path_judge": json.loads(candidate["path_judge_json"]),
                    },
                }
            )
    rewrite_df = pd.DataFrame(rows)
    rewrite_df.to_parquet(paths.rewrites / "rewrite_results.parquet", index=False)
    write_jsonl(paths.final / "accepted_samples.jsonl", final_rows)
    return rewrite_df


def call_rewrite_api(args: argparse.Namespace, candidate: dict[str, Any], templates: list[str]) -> list[str]:
    api_key = os.environ.get(args.rewrite_api_key_env, "").strip()
    if not api_key or not args.rewrite_model:
        return []
    url = args.rewrite_api_base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1") + "/v1/chat/completions"
    system = (
        "You rewrite visual pointing instructions for a steerable pointing dataset. "
        "Return JSON only in the form {\"questions\":[\"...\",\"...\"]}. "
        "Do not judge samples, do not mention validation, and do not add explanations."
    )
    task_requirements = [
        "Keep the exact target label and spatial relation.",
        "The blue point is only a reference/start point.",
        "Never say or imply that the blue point marks, labels, indicates, touches, or is on an object.",
        "Use concise imperative pointing language.",
        "Keep the wording close to official PointArena steerable questions.",
        "Prefer openings like 'Point to ...', 'From the blue point, move ...', 'Start at the blue point ...', or 'Use the blue point only as a reference ...'.",
        "Do not ask QA-style questions such as Where is or Which direction.",
        "Do not mention red points, masks, coordinates, pixels, SAM, segmentation, or validation.",
        "Do not introduce new visual attributes that are not present in the target label.",
        "Avoid UI-specific or annotation-like wording such as cursor, click, tap, hover, drag, slide, navigate, trace a path, follow the path, target the, mark the, or starting mark.",
    ]
    task_type = candidate["task_type"]
    if task_type == "move_until_axis_fixed":
        if "-" in str(candidate["direction_text"]):
            task_requirements.extend(
                [
                    "This is one straight diagonal movement from the blue point to the target.",
                    "Preserve the diagonal direction exactly.",
                    "Do not split the instruction into first/then steps.",
                ]
            )
        else:
            task_requirements.extend(
                [
                    "This is one straight axis-aligned movement from the blue point to the target.",
                    "Preserve the axis direction exactly.",
                    "Do not use diagonal wording.",
                ]
            )
    elif task_type == "move_until_axis_from_other_object":
        task_requirements.extend(
            [
                f"Use the anchor object label exactly as '{candidate.get('anchor_label', '')}'.",
                "Use the anchor object as the textual reference instead of the blue point.",
                "Do not mention a blue point in the rewritten question.",
                "Do not imply the anchor object becomes the target object.",
            ]
        )
        if "-" in str(candidate["direction_text"]):
            task_requirements.extend(
                [
                    "This is one straight diagonal movement from the anchor object to the target.",
                    "Preserve the diagonal direction exactly.",
                    "Do not split the instruction into first/then steps.",
                ]
            )
        else:
            task_requirements.extend(
                [
                    "This is one straight axis-aligned movement from the anchor object to the target.",
                    "Preserve the axis direction exactly.",
                    "Do not use diagonal wording.",
                ]
            )
    elif task_type == "move_then_axis_limited_fixed":
        second_axis = second_axis_from_direction(candidate["direction_text"], candidate.get("first_axis", ""))
        task_requirements.extend(
            [
                "This is a two-step path, not a single diagonal line.",
                f"Preserve the ordered steps exactly: first {candidate.get('first_axis', '')}, then {second_axis}.",
                "Do not collapse the instruction into one diagonal phrase.",
            ]
        )
    elif task_type == "nearest_among_objects":
        task_requirements.extend(
            [
                "This asks for the same-label object nearest or closest to the blue point.",
                "Use nearest/closest explicitly.",
                "Do not use vague nearby wording such as near, nearby, or around the blue point.",
            ]
        )
    elif task_type == "nearest_among_objects_axis_constrained":
        task_requirements.extend(
            [
                "This asks for the same-label object that is nearest within one axis direction from the blue point.",
                "Use nearest/closest explicitly.",
                "Mention the requested axis side explicitly.",
                "Do not collapse the instruction into a generic nearest-object question.",
            ]
        )
    user = {
        "task_type": task_type,
        "target_label": candidate["target_label"],
        "anchor_label": candidate.get("anchor_label", ""),
        "direction": candidate["direction_text"],
        "first_axis": candidate.get("first_axis", ""),
        "templates": templates,
        "requirements": task_requirements,
        "num_variants": 2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": args.rewrite_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    for attempt in range(max(1, args.rewrite_retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=args.rewrite_timeout_s)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return [str(q) for q in data.get("questions", []) if str(q).strip()][:2]
        except Exception:
            time.sleep(2.0)
    return []


def visualize_candidates(paths: Paths, candidate_df: pd.DataFrame | None = None, rewrite_df: pd.DataFrame | None = None) -> None:
    if candidate_df is None:
        candidate_df = pd.read_parquet(paths.task_pool / "task_candidates.parquet")
    object_df = pd.read_parquet(paths.filtered / "object_records.parquet")
    objects = {(row["image_id"], row["object_id"]): row for row in object_df.to_dict(orient="records")}
    rows = []
    for idx, candidate in enumerate(candidate_df.to_dict(orient="records")):
        obj_row = objects[(candidate["image_id"], candidate["object_id"])]
        out = paths.visuals / "candidates" / f"{idx:04d}_{safe_name(candidate['task_type'])}_{safe_name(candidate['object_id'])}.png"
        draw_candidate_visual(candidate, obj_row, paths, out, question=None)
        rows.append({"candidate_id": candidate["candidate_id"], "task_type": candidate["task_type"], "visual_path": str(out)})
    pd.DataFrame(rows).to_csv(paths.visuals / "candidate_visual_index.csv", index=False)
    if rewrite_df is None and (paths.rewrites / "rewrite_results.parquet").exists():
        rewrite_df = pd.read_parquet(paths.rewrites / "rewrite_results.parquet")
    if rewrite_df is not None and len(rewrite_df):
        by_candidate = {row["candidate_id"]: row for row in candidate_df.to_dict(orient="records")}
        rewrite_rows = []
        for idx, row in enumerate(rewrite_df.to_dict(orient="records")):
            candidate = by_candidate[row["candidate_id"]]
            obj_row = objects[(candidate["image_id"], candidate["object_id"])]
            out = paths.visuals / "rewrites" / f"{idx:04d}_{safe_name(candidate['task_type'])}_{safe_name(candidate['object_id'])}.png"
            draw_candidate_visual(candidate, obj_row, paths, out, question=row["question"])
            rewrite_rows.append({"candidate_id": row["candidate_id"], "question": row["question"], "visual_path": str(out)})
        pd.DataFrame(rewrite_rows).to_csv(paths.visuals / "rewrite_visual_index.csv", index=False)


def make_path_debug_panel(
    image_size: tuple[int, int],
    target_mask: np.ndarray,
    other_masks: list[np.ndarray],
    path_points: list[list[float]],
    anchor: list[float],
    target: list[float],
    path_width_px: int = 3,
) -> Image.Image:
    width, height = image_size
    target_mask = target_mask.astype(bool)
    other_union = combine_masks(other_masks, (height, width))
    path_mask = raster_line_mask((height, width), path_points, path_width_px)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    canvas[other_union & ~target_mask] = np.array([152, 120, 214], dtype=np.uint8)
    canvas[target_mask] = np.array([255, 96, 96], dtype=np.uint8)
    canvas[path_mask & ~target_mask & ~other_union] = np.array([255, 220, 40], dtype=np.uint8)
    canvas[path_mask & other_union & ~target_mask] = np.array([255, 145, 0], dtype=np.uint8)
    panel = Image.fromarray(canvas)
    draw = ImageDraw.Draw(panel)
    draw.line([tuple(p) for p in path_points], fill=(255, 235, 0), width=max(3, path_width_px))
    draw.ellipse((anchor[0] - 8, anchor[1] - 8, anchor[0] + 8, anchor[1] + 8), fill=(40, 90, 255), outline=(255, 255, 255), width=2)
    draw.ellipse((target[0] - 8, target[1] - 8, target[0] + 8, target[1] + 8), fill=(255, 48, 48), outline=(255, 255, 255), width=2)
    return panel


def format_judge_lines(candidate: dict[str, Any]) -> list[str]:
    judge = json.loads(candidate["path_judge_json"])
    judge_short = {
        "programmatic_nearest_distance_margin_and_sam3_anchor_exclusion": "nearest margin + sam3",
        "directional_nearest_sector_and_sam3_anchor_exclusion": "directional nearest + sam3",
        "sam3_anchor_exclusion_path_same_class_connectivity_and_other_object_blocking": "sam3 + path block",
        "sam3_other_object_anchor_center_path_same_class_connectivity_and_other_object_blocking": "other-object anchor + sam3",
    }.get(candidate["judge_method"], candidate["judge_method"])
    lines = [
        f"target: {candidate['target_label']}",
        f"direction: {candidate['direction_text']}",
        f"anchor xy: {tuple(round(float(v), 1) for v in candidate['anchor_point_xy'])}",
        f"target xy: {tuple(round(float(v), 1) for v in candidate['target_point_xy'])}",
        f"anchor angle: {candidate['anchor_angle_deg']}",
        f"anchor radius: {candidate['anchor_radius_px']}",
        f"judge: {judge_short}",
    ]
    if candidate["task_type"] == "move_until_axis_from_other_object":
        lines.extend(
            [
                f"anchor label: {candidate.get('anchor_label', '')}",
                f"anchor object id: {candidate.get('anchor_object_id', '')}",
                f"ignored block labels: {', '.join(judge.get('ignored_blocking_same_class_keys', [])) or 'none'}",
                f"path pixels: {judge.get('path_pixels')}",
                f"same-class comps: {judge.get('same_class_remainder_components')}",
                f"other-hit px: {judge.get('other_object_hit_pixels')}",
            ]
        )
    elif candidate["task_type"] == "nearest_among_objects":
        lines.extend(
            [
                f"same count: {judge.get('same_count')}",
                f"nearest dist: {judge.get('closest_dist')}",
                f"second dist: {judge.get('second_dist')}",
                f"margin px: {judge.get('margin_px')} / {judge.get('min_margin_px')}",
            ]
        )
    elif candidate["task_type"] == "nearest_among_objects_axis_constrained":
        lines.extend(
            [
                f"same count: {judge.get('same_count')}",
                f"target dist: {judge.get('target_distance_px')}",
                f"target dir delta: {judge.get('target_direction_delta_deg')}",
                f"sector deg: {judge.get('direction_sector_deg')}",
                f"blocking ids: {', '.join(judge.get('blocking_object_ids', [])) or 'none'}",
            ]
        )
    else:
        lines.extend(
            [
                f"path pixels: {judge.get('path_pixels')}",
                f"same-class comps: {judge.get('same_class_remainder_components')}",
                f"other-hit px: {judge.get('other_object_hit_pixels')}",
                f"other-hit labels: {', '.join(judge.get('other_object_hit_labels', [])) or 'none'}",
            ]
        )
    lines.extend(
        [
            f"raw point coverage: {judge.get('raw_point_coverage')}",
            f"dilated point coverage: {judge.get('dilated_point_coverage')}",
        ]
    )
    return lines


def acceptance_status(candidate: dict[str, Any]) -> tuple[bool, str]:
    judge = json.loads(candidate["path_judge_json"])
    passed = bool(judge.get("pass", False))
    return passed, "accepted" if passed else "rejected"


def draw_candidate_visual(candidate: dict[str, Any], obj_row: dict[str, Any], paths: Paths, out: Path, question: str | None) -> None:
    image = Image.open(candidate["image_path"]).convert("RGB")
    sam_meta = load_sam_meta(paths, candidate["image_id"])
    group = next(g for g in sam_meta["label_groups"] if g["same_class_key"] == candidate["same_class_key"])
    target_mask = load_mask(group["mask_path"])
    dilated_mask = load_mask(group["dilated_mask_path"])
    semantic = Image.open(paths.sam_masks / candidate["image_id"] / "semantic.png").convert("RGB")
    panel = image.copy()
    overlay = np.asarray(panel).copy()
    color = np.asarray([255, 60, 60], dtype=np.float32)
    overlay[dilated_mask] = (overlay[dilated_mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
    panel = Image.fromarray(overlay)
    draw = ImageDraw.Draw(panel)
    anchor = candidate["anchor_point_xy"]
    target = candidate["target_point_xy"]
    path_points = json.loads(candidate["path_points_xy_json"])
    draw.line([tuple(p) for p in path_points], fill=(255, 210, 0), width=4)
    draw.ellipse((anchor[0] - 8, anchor[1] - 8, anchor[0] + 8, anchor[1] + 8), fill=(30, 90, 255), outline=(255, 255, 255), width=2)
    draw.ellipse((target[0] - 8, target[1] - 8, target[0] + 8, target[1] + 8), fill=(255, 30, 30), outline=(255, 255, 255), width=2)
    raw_points = json.loads(obj_row["points_xy_json"])
    for point_x, point_y in raw_points:
        draw.ellipse((point_x - 4, point_y - 4, point_x + 4, point_y + 4), fill=(255, 255, 255), outline=(10, 10, 10), width=1)
    ignored_keys = {candidate["same_class_key"]}
    if candidate["task_type"] == "move_until_axis_from_other_object" and candidate.get("anchor_same_class_key"):
        ignored_keys.add(candidate["anchor_same_class_key"])
    other_masks = [load_mask(g["mask_path"]) for g in sam_meta["label_groups"] if g["same_class_key"] not in ignored_keys]
    judge_panel = make_path_debug_panel((image.width, image.height), target_mask, other_masks, path_points, anchor, target, path_width_px=3)
    width = image.width * 3 + 680
    height = max(image.height, 260)
    canvas = Image.new("RGB", (width, height), (246, 246, 240))
    canvas.paste(semantic, (0, 0))
    canvas.paste(panel, (image.width, 0))
    canvas.paste(judge_panel, (image.width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    title = load_font(22)
    font = load_font(17)
    x = image.width * 3 + 44
    y = 20
    accepted, acceptance_text = acceptance_status(candidate)
    badge_fill = (50, 150, 90) if accepted else (205, 70, 70)
    badge_text = f"acceptance: {acceptance_text}"
    draw.rounded_rectangle((x, y, x + 240, y + 34), radius=10, fill=badge_fill)
    draw.text((x + 14, y + 6), badge_text, font=font, fill=(255, 255, 255))
    y += 48
    draw.text((x, y), candidate["task_type"], font=title, fill=(20, 20, 20))
    y += 36
    draw.text((image.width * 2 + 16, 16), "Path Judge", font=title, fill=(20, 20, 20))
    legend_y = 52
    for label, color_box in [
        ("target mask", (255, 96, 96)),
        ("other masks", (152, 120, 214)),
        ("clear path", (255, 220, 40)),
        ("blocked path", (255, 145, 0)),
    ]:
        draw.rounded_rectangle((image.width * 2 + 16, legend_y + 4, image.width * 2 + 38, legend_y + 24), radius=4, fill=color_box)
        draw.text((image.width * 2 + 48, legend_y + 2), label, font=font, fill=(30, 30, 30))
        legend_y += 28
    for line in format_judge_lines(candidate):
        draw.text((x, y), line, font=font, fill=(30, 30, 30))
        y += 26
    if question:
        y += 10
        draw.text((x, y), "question:", font=title, fill=(20, 20, 20))
        y += 34
        for line in wrap_text(question, 42):
            draw.text((x, y), line, font=font, fill=(10, 10, 10))
            y += 24
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def wrap_text(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        if sum(len(x) + 1 for x in cur) + len(word) > width and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def run_full(args: argparse.Namespace, paths: Paths) -> None:
    object_df = filter_object_records(args, paths)
    build_sam_masks(args, paths, object_df)
    candidate_df = build_candidates(args, paths, object_df)
    rewrite_df = rewrite_candidates(args, paths, candidate_df)
    visualize_candidates(paths, candidate_df, rewrite_df)


def main() -> None:
    args = parse_args()
    args.template_library = load_template_library(args.template_library_json)
    if args.output_root.exists() and args.force:
        shutil.rmtree(args.output_root)
    paths = ensure_paths(args.output_root)
    write_json(
        paths.reports / "template_library_summary.json",
        {
            "template_library_json": str(args.template_library_json),
            "templates_per_candidate": int(args.templates_per_candidate),
            "template_counts": {key: len(value) for key, value in (args.template_library or {}).items()},
        },
    )
    if args.stage == "full":
        run_full(args, paths)
    elif args.stage == "filter":
        filter_object_records(args, paths)
    elif args.stage == "sam":
        build_sam_masks(args, paths)
    elif args.stage == "candidates":
        build_candidates(args, paths)
    elif args.stage == "rewrite":
        rewrite_candidates(args, paths)
    elif args.stage == "visualize":
        visualize_candidates(paths)
    print(f"[done] {args.stage}: {paths.root}")


if __name__ == "__main__":
    main()
