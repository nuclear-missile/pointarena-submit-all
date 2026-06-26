from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from src.train.prompting import build_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTARENA_EXTRACT_DIR = PROJECT_ROOT / "pointarena_extract"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "cache" / "pointarena_rewritten_training_summary_latest.json"
DEFAULT_STEERABLE_D_SUMMARY_PATH = PROJECT_ROOT / "data" / "cache" / "pointarena_rewritten_training_summary_steerable_d.json"
DEFAULT_SAM_CLEAN_JSONL = (
    PROJECT_ROOT / "make_steerable1" / "sam_clean" / "outputs" / "mix10000_v2" / "07_final" / "accepted_samples.jsonl"
)
DEFAULT_CLEAN3_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "clean_3types_local" / "qwen3_8b_three_types_2000_each_nodup_reverse",
    PROJECT_ROOT / "clean_3types_local" / "qwen3_8b_three_types_add2000_nodup_reverse",
)
LATEST_VALID_CATEGORIES: tuple[str, ...] = (
    "Reasoning",
    "Spatial Relation",
    "Affordance",
    "Counting",
    "Object Reference",
    "Steerable",
)
STEERABLE_D_ACTIVE_CATEGORIES: tuple[str, ...] = (
    "Affordance",
    "Counting",
    "Object Reference",
    "Reasoning",
    "Steerable",
)
POINT_MODES = {"single", "multi", "all"}
SUMMARY_VERSION = 1
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class SampleCandidate:
    sample_id: str
    root_name: str
    sample_path: Path
    timestamp_key: float
    timestamp_iso: str
    version_kind: str
    decision_rank: int
    is_judged: bool
    status: str
    category: str | None
    filter_stage: str | None
    source: str
    source_file: str
    source_rank: int | None
    record_id: str
    sample: dict[str, Any]


def normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    alias = {
        "reasoning": "Reasoning",
        "spatial relation": "Spatial Relation",
        "spatial": "Spatial Relation",
        "affordance": "Affordance",
        "counting": "Counting",
        "object reference": "Object Reference",
        "objectref": "Object Reference",
        "reference": "Object Reference",
        "steerable": "Steerable",
        "steerability": "Steerable",
        "none": None,
        "": None,
    }
    if key in alias:
        return alias[key]
    for label in LATEST_VALID_CATEGORIES:
        if key == label.lower():
            return label
    return None


def normalize_point_mode(point_mode: str) -> str:
    value = (point_mode or "all").strip().lower()
    return value if value in POINT_MODES else "all"


def should_keep_by_point_mode(num_points: int, point_mode: str) -> bool:
    if num_points <= 0:
        return False
    if point_mode == "single":
        return num_points == 1
    if point_mode == "multi":
        return num_points > 1
    return True


def strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_rewritten_text(raw_text: Any) -> str:
    if raw_text is None:
        return ""
    cleaned = strip_code_fence(str(raw_text))
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in (
            "rewritten_query",
            "rewritten_instruction",
            "rewritten",
            "rewrite",
            "query",
            "rewritten_task",
            "text",
        ):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                break
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]
    cleaned = cleaned.strip().strip('"').strip("'")
    match = re.search(r"Point to[\s\S]*", cleaned)
    if match:
        cleaned = match.group(0).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def is_valid_rewritten_query(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip()) and text.strip().startswith("Point to")


def parse_timestamp(value: Any, fallback_epoch: float) -> tuple[float, str]:
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp(), dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    fallback_dt = datetime.fromtimestamp(fallback_epoch, tz=timezone.utc)
    return fallback_epoch, fallback_dt.isoformat().replace("+00:00", "Z")


def canonical_sample_id(sample: dict[str, Any], sample_path: Path) -> str:
    processing = sample.get("processing", {}) or {}
    record = sample.get("record", {}) or {}
    for value in (processing.get("sample_id"), record.get("id")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = processing.get("source") or record.get("source") or sample_path.parent.name
    source_file = processing.get("source_file") or record.get("source_file") or ""
    source_rank = processing.get("source_rank")
    return f"{source}::{source_file}::{source_rank if source_rank is not None else sample_path.stem}"


def detect_version_kind(sample: dict[str, Any]) -> str:
    keys = set(sample.keys())
    if {"cleaning", "multipoint", "classification", "rewrite"}.issubset(keys):
        return "full_workflow"
    if {"classification", "rewrite"}.issubset(keys):
        return "legacy_classify_rewrite"
    return "unknown"


def extract_rewrite_query(sample: dict[str, Any]) -> str:
    rewrite = sample.get("rewrite")
    if not isinstance(rewrite, dict):
        return ""
    for key in (
        "rewritten_query",
        "rewritten_instruction",
        "rewritten",
        "rewrite",
        "query",
        "rewritten_task",
        "text",
    ):
        value = rewrite.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_rewritten_text(value)
    raw_output = rewrite.get("raw_output")
    if isinstance(raw_output, str) and raw_output.strip():
        return normalize_rewritten_text(raw_output)
    return ""


def is_full_workflow_judged(sample: dict[str, Any]) -> bool:
    processing = sample.get("processing")
    cleaning = sample.get("cleaning")
    if not isinstance(processing, dict) or not isinstance(cleaning, dict):
        return False

    status = str(processing.get("status") or "").strip()
    if status not in {"completed", "filtered_out"}:
        return False

    keep = cleaning.get("keep")
    reason = cleaning.get("reason")
    if not isinstance(keep, bool):
        parsed = cleaning.get("parsed")
        if isinstance(parsed, dict):
            keep = parsed.get("keep")
    if not isinstance(keep, bool):
        return False
    if not isinstance(reason, str) or not reason.strip():
        parsed = cleaning.get("parsed")
        if isinstance(parsed, dict):
            reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False

    if keep is False:
        return status == "filtered_out"

    multipoint = sample.get("multipoint")
    classification = sample.get("classification")
    rewrite = sample.get("rewrite")
    if not isinstance(multipoint, dict) or not isinstance(classification, dict) or not isinstance(rewrite, dict):
        return False

    multi_point = multipoint.get("multi_point")
    if not isinstance(multi_point, bool):
        parsed = multipoint.get("parsed")
        if isinstance(parsed, dict):
            multi_point = parsed.get("multi_point")
    if not isinstance(multi_point, bool):
        return False

    category = normalize_category(classification.get("category"))
    filter_stage = str(processing.get("filter_stage") or "").strip()
    if status == "filtered_out":
        if filter_stage == "classification":
            return category is None
        return False

    if category not in LATEST_VALID_CATEGORIES:
        return False
    rewritten_query = extract_rewrite_query(sample)
    return is_valid_rewritten_query(rewritten_query)


def is_legacy_judged(sample: dict[str, Any]) -> bool:
    processing = sample.get("processing")
    classification = sample.get("classification")
    if not isinstance(processing, dict) or not isinstance(classification, dict):
        return False
    if str(processing.get("status") or "").strip() != "completed":
        return False
    category = normalize_category(classification.get("category"))
    if category not in LATEST_VALID_CATEGORIES:
        return False
    rewritten_query = extract_rewrite_query(sample)
    return is_valid_rewritten_query(rewritten_query)


def to_float_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            return float(point[0]), float(point[1])
        except Exception:
            return None
    if isinstance(point, dict) and "x" in point and "y" in point:
        try:
            return float(point["x"]), float(point["y"])
        except Exception:
            return None
    return None


def clip_pixel_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    max_w = float(max(int(width) - 1, 0))
    max_h = float(max(int(height) - 1, 0))
    return max(0.0, min(max_w, float(x))), max(0.0, min(max_h, float(y)))


def convert_norm01_to_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    den_w = float(max(int(width) - 1, 1))
    den_h = float(max(int(height) - 1, 1))
    return max(0.0, min(den_w, x * den_w)), max(0.0, min(den_h, y * den_h))


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def load_image_size(image_path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception:
        return (0, 0)

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return (0, 0)
    try:
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        return (0, 0)


def is_openable_local_image(image_path: str | Path) -> bool:
    try:
        from PIL import Image, ImageFile
    except Exception:
        return False

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
        return False

    try:
        with Image.open(path) as image:
            image.load()
            _ = image.size
        return True
    except Exception:
        return False


def extract_label(sample: dict[str, Any], category: str | None) -> str:
    record = sample.get("record", {}) or {}
    if isinstance(record.get("ground_truth_label"), str) and record["ground_truth_label"].strip():
        return record["ground_truth_label"].strip()

    cleaning = sample.get("cleaning", {}) or {}
    for value in (cleaning.get("core_target"), ((cleaning.get("parsed") or {}).get("core_target") if isinstance(cleaning.get("parsed"), dict) else None)):
        if isinstance(value, str) and value.strip():
            return value.strip()

    raw_meta = record.get("raw_meta", {}) or {}
    for source in (record, raw_meta, raw_meta.get("raw_meta", {}) or {}, record.get("meta", {}) or {}):
        if isinstance(source, dict):
            value = source.get("label")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return category or "target"


def extract_absolute_points(record: dict[str, Any], width: int, height: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    direct_points = record.get("points_abs")
    if isinstance(direct_points, list):
        for point in direct_points:
            xy = to_float_xy(point)
            if xy is None:
                continue
            points.append(clip_pixel_point(xy[0], xy[1], width, height))
        if points:
            return points

    target = record.get("target") or {}
    target_points = target.get("points") if isinstance(target, dict) else None
    point_format = str((target.get("point_format") if isinstance(target, dict) else "") or "").strip().lower()
    image_size = target.get("image_size") if isinstance(target, dict) else None
    if (width <= 0 or height <= 0) and isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        width = coerce_int(image_size[0], width)
        height = coerce_int(image_size[1], height)

    if isinstance(target_points, list):
        for point in target_points:
            xy = to_float_xy(point)
            if xy is None:
                continue
            x0, y0 = xy
            if point_format == "xy_norm_01":
                x1, y1 = convert_norm01_to_pixels(x0, y0, width, height)
            else:
                x1, y1 = clip_pixel_point(x0, y0, width, height)
            points.append((x1, y1))
        if points:
            return points

    generic_points = record.get("points")
    if isinstance(generic_points, list):
        for point in generic_points:
            xy = to_float_xy(point)
            if xy is None:
                continue
            points.append(clip_pixel_point(xy[0], xy[1], width, height))
    return points


def build_structured_training_row(
    sample: dict[str, Any],
    sample_id: str,
    root_name: str,
    sample_path: Path,
    version_kind: str,
) -> dict[str, Any] | None:
    processing = sample.get("processing", {}) or {}
    record = sample.get("record", {}) or {}
    classification = sample.get("classification", {}) or {}
    rewrite = sample.get("rewrite", {}) or {}

    category = normalize_category(classification.get("category"))
    if category not in LATEST_VALID_CATEGORIES:
        return None

    question = extract_rewrite_query(sample)
    if not is_valid_rewritten_query(question):
        return None

    image_path = str(record.get("image_path") or "").strip()
    width = coerce_int(record.get("width"), 0)
    height = coerce_int(record.get("height"), 0)

    target = record.get("target") or {}
    if (width <= 0 or height <= 0) and isinstance(target, dict):
        image_size = target.get("image_size")
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            width = coerce_int(image_size[0], width)
            height = coerce_int(image_size[1], height)

    if (width <= 0 or height <= 0) and image_path:
        width, height = load_image_size(image_path)
    if width <= 0 or height <= 0:
        return None

    points = extract_absolute_points(record, width, height)
    if not points:
        return None

    label = extract_label(sample, category)
    style = "point_count" if category == "Counting" else "pointing"
    original_query = str(record.get("query") or "").strip()

    metadata = {
        "category": category,
        "source": processing.get("source") or record.get("source") or "",
        "source_file": processing.get("source_file") or "",
        "source_rank": processing.get("source_rank"),
        "record_id": record.get("id") or sample_id,
        "selected_root": root_name,
        "selected_sample_path": str(sample_path.resolve()),
        "selected_version_kind": version_kind,
        "selected_completed_at": processing.get("completed_at") or processing.get("started_at") or "",
        "workflow": processing.get("workflow") or "",
        "original_query": original_query,
        "rewritten_query": question,
        "ground_truth_label": record.get("ground_truth_label") or "",
        "classification_allowed_categories": classification.get("allowed_categories") or list(LATEST_VALID_CATEGORIES),
        "rewrite_finish_reason": rewrite.get("finish_reason"),
    }

    return {
        "id": str(record.get("id") or sample_id),
        "image": image_path,
        "style": style,
        "label": label,
        "question": question,
        "points": [(float(x), float(y)) for x, y in points],
        "width": int(width),
        "height": int(height),
        "metadata": metadata,
    }


def summarize_history(candidates: Sequence[SampleCandidate]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item.timestamp_key, item.sample_path.as_posix())):
        history.append(
            {
                "root": candidate.root_name,
                "path": str(candidate.sample_path.resolve()),
                "status": candidate.status,
                "category": candidate.category,
                "version_kind": candidate.version_kind,
                "decision_rank": candidate.decision_rank,
                "is_judged": candidate.is_judged,
                "timestamp": candidate.timestamp_iso,
                "filter_stage": candidate.filter_stage,
            }
        )
    return history


def parse_candidate(sample_path: Path, root_name: str) -> SampleCandidate | None:
    try:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(sample, dict):
        return None

    processing = sample.get("processing", {}) or {}
    record = sample.get("record", {}) or {}
    status = str(processing.get("status") or "").strip() or "unknown"
    version_kind = detect_version_kind(sample)
    decision_rank = 0
    is_judged = False
    if version_kind == "full_workflow" and is_full_workflow_judged(sample):
        decision_rank = 2
        is_judged = True
    elif version_kind == "legacy_classify_rewrite" and is_legacy_judged(sample):
        decision_rank = 1
        is_judged = True

    completed_at = processing.get("completed_at") or processing.get("started_at")
    timestamp_key, timestamp_iso = parse_timestamp(completed_at, sample_path.stat().st_mtime)
    category = normalize_category(((sample.get("classification") or {}).get("category") if isinstance(sample.get("classification"), dict) else None))
    source = str(processing.get("source") or record.get("source") or root_name)
    source_file = str(processing.get("source_file") or "")
    source_rank_raw = processing.get("source_rank")
    source_rank = coerce_int(source_rank_raw, 0) if source_rank_raw is not None else None
    record_id = str(record.get("id") or "").strip()
    return SampleCandidate(
        sample_id=canonical_sample_id(sample, sample_path),
        root_name=root_name,
        sample_path=sample_path,
        timestamp_key=timestamp_key,
        timestamp_iso=timestamp_iso,
        version_kind=version_kind,
        decision_rank=decision_rank,
        is_judged=is_judged,
        status=status,
        category=category,
        filter_stage=str(processing.get("filter_stage") or "").strip() or None,
        source=source,
        source_file=source_file,
        source_rank=source_rank,
        record_id=record_id,
        sample=sample,
    )


def discover_requested_roots(output_base: str | Path = POINTARENA_EXTRACT_DIR) -> list[str]:
    base = Path(output_base)
    discovered: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name == "__pycache__" or name.startswith("tmp_") or "smoke" in name:
            continue
        if any(child.glob("*/*.json")):
            discovered.append(name)
    return discovered


def choose_latest_candidate(candidates: Sequence[SampleCandidate]) -> SampleCandidate | None:
    judged = [candidate for candidate in candidates if candidate.is_judged]
    if not judged:
        return None
    max_rank = max(candidate.decision_rank for candidate in judged)
    ranked = [candidate for candidate in judged if candidate.decision_rank == max_rank]
    return max(ranked, key=lambda item: (item.timestamp_key, item.sample_path.as_posix()))


def build_summary_cache(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    *,
    output_base: str | Path = POINTARENA_EXTRACT_DIR,
    roots: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    output_root = Path(output_base)
    root_names = list(roots) if roots is not None else discover_requested_roots(output_root)

    grouped: dict[str, list[SampleCandidate]] = defaultdict(list)
    per_root_file_counts: dict[str, int] = {}
    scanned_files = 0

    for root_name in root_names:
        sample_count = 0
        for sample_path in sorted((output_root / root_name).glob("*/*.json")):
            sample_count += 1
            scanned_files += 1
            candidate = parse_candidate(sample_path, root_name)
            if candidate is None:
                continue
            grouped[candidate.sample_id].append(candidate)
        per_root_file_counts[root_name] = sample_count

    summary_rows: list[dict[str, Any]] = []
    selected_status_counts: Counter[str] = Counter()
    selected_category_counts: Counter[str] = Counter()
    selected_root_counts: Counter[str] = Counter()
    trainable_completed = 0

    for sample_id in sorted(grouped.keys()):
        candidates = grouped[sample_id]
        chosen = choose_latest_candidate(candidates)
        history = summarize_history(candidates)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "history": history,
            "history_count": len(history),
        }

        if chosen is None:
            row.update(
                {
                    "status": "no_usable_judgement",
                    "category": None,
                    "selected_root": None,
                    "selected_path": None,
                    "selected_version_kind": None,
                    "selected_timestamp": None,
                    "trainable": False,
                    "row": None,
                }
            )
            selected_status_counts[row["status"]] += 1
            summary_rows.append(row)
            continue

        row.update(
            {
                "status": chosen.status,
                "category": chosen.category,
                "selected_root": chosen.root_name,
                "selected_path": str(chosen.sample_path.resolve()),
                "selected_version_kind": chosen.version_kind,
                "selected_timestamp": chosen.timestamp_iso,
                "selected_filter_stage": chosen.filter_stage,
                "source": chosen.source,
                "source_file": chosen.source_file,
                "source_rank": chosen.source_rank,
                "record_id": chosen.record_id,
            }
        )

        if chosen.status == "completed":
            structured_row = build_structured_training_row(
                chosen.sample,
                sample_id=sample_id,
                root_name=chosen.root_name,
                sample_path=chosen.sample_path,
                version_kind=chosen.version_kind,
            )
            row["trainable"] = structured_row is not None
            row["row"] = structured_row
            if structured_row is not None:
                trainable_completed += 1
        else:
            row["trainable"] = False
            row["row"] = None

        selected_status_counts[row["status"]] += 1
        selected_root_counts[chosen.root_name] += 1
        if row["status"] == "completed" and row.get("category") in LATEST_VALID_CATEGORIES:
            selected_category_counts[str(row["category"])] += 1
        summary_rows.append(row)

    payload = {
        "summary_version": SUMMARY_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_root": str(PROJECT_ROOT),
        "output_base": str(output_root.resolve()),
        "source_roots": root_names,
        "scanned_files": scanned_files,
        "unique_sample_ids": len(grouped),
        "trainable_completed_rows": trainable_completed,
        "latest_valid_categories": list(LATEST_VALID_CATEGORIES),
        "selection_policy": {
            "prefer_full_workflow_over_legacy": True,
            "choose_latest_timestamp_within_same_workflow_rank": True,
            "filtered_out_in_latest_full_workflow_excludes_older_completed_versions": True,
            "legacy_no_clean_only_used_when_no_newer_full_workflow_judgement_exists": True,
        },
        "counts": {
            "selected_status": dict(sorted(selected_status_counts.items())),
            "selected_category_completed": dict(sorted(selected_category_counts.items())),
            "selected_root": dict(sorted(selected_root_counts.items())),
            "files_per_root": dict(sorted(per_root_file_counts.items())),
        },
        "rows": summary_rows,
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_summary_cache(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    *,
    rebuild_if_missing: bool = True,
    output_base: str | Path = POINTARENA_EXTRACT_DIR,
    roots: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    if summary_file.exists():
        return json.loads(summary_file.read_text(encoding="utf-8"))
    if not rebuild_if_missing:
        raise FileNotFoundError(f"Missing summary cache: {summary_file}")
    return build_summary_cache(summary_file, output_base=output_base, roots=roots)


def _ratio_weights(category_balance: dict[str, float]) -> dict[str, int]:
    normalized: dict[str, Fraction] = {}
    for key, value in category_balance.items():
        category = normalize_category(key)
        if category not in LATEST_VALID_CATEGORIES:
            raise ValueError(f"Unsupported category in category_balance: {key}")
        if value is None:
            continue
        numeric = float(value)
        if numeric <= 0:
            continue
        normalized[category] = Fraction(str(numeric)).limit_denominator(1000)
    if not normalized:
        raise ValueError("category_balance must contain at least one positive category weight")

    lcm = 1
    for fraction in normalized.values():
        lcm = math.lcm(lcm, fraction.denominator)

    weights: dict[str, int] = {}
    for category, fraction in normalized.items():
        weights[category] = int(fraction * lcm)

    gcd_value = 0
    for value in weights.values():
        gcd_value = value if gcd_value == 0 else math.gcd(gcd_value, value)
    if gcd_value > 1:
        for category in list(weights.keys()):
            weights[category] //= gcd_value
    return dict(sorted(weights.items()))


def _serialize_row_for_dataset(row: dict[str, Any], max_target_points: int) -> dict[str, Any]:
    item = dict(row)
    points = [(float(x), float(y)) for x, y in item.get("points", [])]
    if max_target_points > 0 and len(points) > max_target_points:
        rng = random.Random(item.get("id") or item.get("question") or "pointarena")
        indices = sorted(rng.sample(range(len(points)), max_target_points))
        points = [points[index] for index in indices]
    item["points"] = points
    item["metadata"] = dict(item.get("metadata", {}) or {})
    return item


class PointArenaRewrittenTrainingDataset(Dataset):
    """
    Training dataset backed by cached local PointArena rewrite outputs.

    Returned item schema follows ``src.train.point_dataset.UnifiedPointDataset``:
    - id
    - image
    - style
    - label
    - question
    - points  # absolute pixel coordinates
    - width
    - height
    - metadata
    """

    def __init__(
        self,
        summary_path: str | Path = DEFAULT_SUMMARY_PATH,
        *,
        output_base: str | Path = POINTARENA_EXTRACT_DIR,
        roots: Optional[Iterable[str]] = None,
        force_rebuild_summary: bool = False,
        rebuild_if_missing: bool = True,
        require_local_image: bool = True,
        verify_openable_image: bool = False,
        exclude_pointarena_eval: bool = False,
        point_mode: str = "all",
        max_target_points: int = 64,
        category_balance: Optional[dict[str, float]] = None,
        shuffle_selected: bool = False,
        selection_seed: int = 42,
    ) -> None:
        self.summary_path = Path(summary_path)
        self.output_base = Path(output_base)
        self.roots = list(roots) if roots is not None else None
        self.require_local_image = require_local_image
        self.verify_openable_image = bool(verify_openable_image)
        self.exclude_pointarena_eval = bool(exclude_pointarena_eval)
        self.point_mode = normalize_point_mode(point_mode)
        self.max_target_points = int(max_target_points)
        self.shuffle_selected = bool(shuffle_selected)
        self.selection_seed = int(selection_seed)
        self.rotation_round = 0
        self.category_balance = dict(category_balance or {})
        self.category_weights = _ratio_weights(self.category_balance) if self.category_balance else None
        self._image_validity_cache: dict[str, bool] = {}

        if force_rebuild_summary or not self.summary_path.exists():
            self.summary = build_summary_cache(self.summary_path, output_base=self.output_base, roots=self.roots)
        else:
            self.summary = load_summary_cache(
                self.summary_path,
                rebuild_if_missing=rebuild_if_missing,
                output_base=self.output_base,
                roots=self.roots,
            )

        self.summary_rows: list[dict[str, Any]] = list(self.summary.get("rows", []))
        self.eligible_rows: list[dict[str, Any]] = self._collect_eligible_rows()
        self.category_pools: dict[str, list[dict[str, Any]]] = self._build_category_pools()
        self.category_offsets: dict[str, int] = {category: 0 for category in self.category_pools}
        self.category_quotas: dict[str, int] = self._compute_category_quotas()
        self.rows: list[dict[str, Any]] = []
        self.discarded_rows: list[dict[str, Any]] = []
        self._refresh_selection()

    def _is_usable_local_image(self, image_path: str) -> bool:
        path = str(image_path or "").strip()
        if not path:
            return False
        cached = self._image_validity_cache.get(path)
        if cached is not None:
            return cached

        if self.verify_openable_image:
            ok = is_openable_local_image(path)
        else:
            ok = Path(path).exists()

        self._image_validity_cache[path] = bool(ok)
        return bool(ok)

    def _is_pointarena_eval_row(self, summary_row: dict[str, Any], row: dict[str, Any]) -> bool:
        if not self.exclude_pointarena_eval:
            return False

        sample_id = str(row.get("id") or summary_row.get("sample_id") or "").strip().lower()
        if sample_id.startswith("pointarena_"):
            return True

        source_file_candidates = [
            str(summary_row.get("source_file") or ""),
            str((row.get("metadata") or {}).get("source_file") or ""),
        ]
        for source_file in source_file_candidates:
            normalized = source_file.replace("\\", "/").lower()
            if normalized.endswith("/data/unified/val_pointarena.jsonl") or normalized.endswith("val_pointarena.jsonl"):
                return True

        selected_root_candidates = [
            str(summary_row.get("selected_root") or ""),
            str((row.get("metadata") or {}).get("selected_root") or ""),
        ]
        for root_name in selected_root_candidates:
            if str(root_name).strip().lower().startswith("pointarena_"):
                return True

        return False

    def _collect_eligible_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for summary_row in self.summary_rows:
            if summary_row.get("status") != "completed":
                continue
            if summary_row.get("trainable") is not True:
                continue
            row = summary_row.get("row")
            if not isinstance(row, dict):
                continue
            if self._is_pointarena_eval_row(summary_row, row):
                continue
            image_path = str(row.get("image") or "").strip()
            if self.require_local_image and not self._is_usable_local_image(image_path):
                continue
            points = row.get("points") or []
            if not should_keep_by_point_mode(len(points), self.point_mode):
                continue
            rows.append(_serialize_row_for_dataset(row, self.max_target_points))
        return rows

    def _build_category_pools(self) -> dict[str, list[dict[str, Any]]]:
        pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.eligible_rows:
            metadata = row.get("metadata", {}) or {}
            category = normalize_category(metadata.get("category"))
            if category not in LATEST_VALID_CATEGORIES:
                continue
            if self.category_weights is not None and category not in self.category_weights:
                continue
            pools[category].append(row)
        for category in list(pools.keys()):
            pools[category] = sorted(
                pools[category],
                key=lambda item: (
                    str((item.get("metadata", {}) or {}).get("selected_completed_at") or ""),
                    str(item.get("id") or ""),
                ),
            )
        return dict(sorted(pools.items()))

    def _compute_category_quotas(self) -> dict[str, int]:
        if not self.category_weights:
            return {category: len(rows) for category, rows in self.category_pools.items()}
        if not self.category_pools:
            return {category: 0 for category in self.category_weights}
        multiplier = None
        for category, weight in self.category_weights.items():
            available = len(self.category_pools.get(category, []))
            current = available // weight if weight > 0 else 0
            multiplier = current if multiplier is None else min(multiplier, current)
        multiplier = int(multiplier or 0)
        return {category: weight * multiplier for category, weight in self.category_weights.items()}

    def _selected_and_discarded_for_category(self, category: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pool = list(self.category_pools.get(category, []))
        quota = int(self.category_quotas.get(category, len(pool)))
        if quota <= 0 or not pool:
            return [], pool
        if quota >= len(pool):
            return pool, []
        offset = int(self.category_offsets.get(category, 0)) % len(pool)
        selected_indices = [((offset + i) % len(pool)) for i in range(quota)]
        selected_index_set = set(selected_indices)
        selected = [pool[index] for index in selected_indices]
        discarded = [pool[index] for index in range(len(pool)) if index not in selected_index_set]
        return selected, discarded

    def _refresh_selection(self) -> None:
        selected_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        if not self.category_weights:
            for category in sorted(self.category_pools.keys()):
                selected_rows.extend(self.category_pools[category])
        else:
            for category in self.category_weights:
                selected, discarded = self._selected_and_discarded_for_category(category)
                selected_rows.extend(selected)
                discarded_rows.extend(discarded)
        if self.shuffle_selected:
            rng = random.Random(self.selection_seed + self.rotation_round)
            rng.shuffle(selected_rows)
        self.rows = selected_rows
        self.discarded_rows = discarded_rows

    def rotate_balanced_samples(self, step: int | dict[str, int] | None = None) -> None:
        if not self.category_weights:
            return
        for category, pool in self.category_pools.items():
            pool_size = len(pool)
            quota = int(self.category_quotas.get(category, 0))
            if pool_size <= quota or quota <= 0:
                continue
            if isinstance(step, dict):
                shift = int(step.get(category, quota))
            elif step is None:
                shift = quota
            else:
                shift = int(step)
            self.category_offsets[category] = (int(self.category_offsets.get(category, 0)) + shift) % pool_size
        self.rotation_round += 1
        self._refresh_selection()

    def category_counts(self, *, active_only: bool = True) -> dict[str, int]:
        rows = self.rows if active_only else self.eligible_rows
        counts: Counter[str] = Counter()
        for row in rows:
            category = normalize_category(((row.get("metadata", {}) or {}).get("category")))
            if category:
                counts[category] += 1
        return dict(sorted(counts.items()))

    def balance_report(self) -> dict[str, Any]:
        return {
            "category_weights": dict(self.category_weights or {}),
            "category_quotas": dict(sorted(self.category_quotas.items())),
            "eligible_category_counts": {
                category: len(rows) for category, rows in sorted(self.category_pools.items())
            },
            "active_category_counts": self.category_counts(active_only=True),
            "discarded_count": len(self.discarded_rows),
            "rotation_round": self.rotation_round,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _compute_summary_counts(summary_rows: Sequence[dict[str, Any]], per_root_file_counts: dict[str, int]) -> dict[str, Any]:
    selected_status_counts: Counter[str] = Counter()
    selected_category_counts: Counter[str] = Counter()
    selected_root_counts: Counter[str] = Counter()
    trainable_completed = 0

    for row in summary_rows:
        status = str(row.get("status") or "unknown")
        selected_status_counts[status] += 1
        root_name = row.get("selected_root")
        if isinstance(root_name, str) and root_name.strip():
            selected_root_counts[root_name] += 1
        category = normalize_category(row.get("category"))
        if status == "completed" and category:
            selected_category_counts[category] += 1
            if row.get("trainable") is True and isinstance(row.get("row"), dict):
                trainable_completed += 1

    return {
        "trainable_completed_rows": int(trainable_completed),
        "counts": {
            "selected_status": dict(sorted(selected_status_counts.items())),
            "selected_category_completed": dict(sorted(selected_category_counts.items())),
            "selected_root": dict(sorted(selected_root_counts.items())),
            "files_per_root": dict(sorted(per_root_file_counts.items())),
        },
    }


def _summary_row_from_candidate(
    candidate: SampleCandidate | None,
    *,
    sample_id: str,
    history: list[dict[str, Any]],
    spatial_relation_to_filtered: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "history": history,
        "history_count": len(history),
    }
    if candidate is None:
        row.update(
            {
                "status": "no_usable_judgement",
                "category": None,
                "selected_root": None,
                "selected_path": None,
                "selected_version_kind": None,
                "selected_timestamp": None,
                "selected_filter_stage": None,
                "source": None,
                "source_file": None,
                "source_rank": None,
                "record_id": None,
                "trainable": False,
                "row": None,
            }
        )
        return row

    row.update(
        {
            "status": candidate.status,
            "category": candidate.category,
            "selected_root": candidate.root_name,
            "selected_path": str(candidate.sample_path.resolve()),
            "selected_version_kind": candidate.version_kind,
            "selected_timestamp": candidate.timestamp_iso,
            "selected_filter_stage": candidate.filter_stage,
            "source": candidate.source,
            "source_file": candidate.source_file,
            "source_rank": candidate.source_rank,
            "record_id": candidate.record_id,
            "trainable": False,
            "row": None,
        }
    )

    if candidate.status != "completed":
        return row

    if spatial_relation_to_filtered and normalize_category(candidate.category) == "Spatial Relation":
        row["status"] = "filtered_out"
        row["selected_filter_stage"] = "category_filter_spatial_relation_removed"
        return row

    structured_row = build_structured_training_row(
        candidate.sample,
        sample_id=sample_id,
        root_name=candidate.root_name,
        sample_path=candidate.sample_path,
        version_kind=candidate.version_kind,
    )
    row["trainable"] = structured_row is not None
    row["row"] = structured_row
    return row


def _scan_candidate_groups(root_paths: Sequence[Path]) -> tuple[dict[str, list[SampleCandidate]], dict[str, int], int]:
    grouped: dict[str, list[SampleCandidate]] = defaultdict(list)
    per_root_file_counts: dict[str, int] = {}
    scanned_files = 0

    for root_path in root_paths:
        sample_count = 0
        for sample_path in sorted(root_path.glob("*/*.json")):
            sample_count += 1
            scanned_files += 1
            candidate = parse_candidate(sample_path, root_path.name)
            if candidate is None:
                continue
            grouped[candidate.sample_id].append(candidate)
        per_root_file_counts[root_path.name] = sample_count

    return grouped, per_root_file_counts, scanned_files


def _sam_clean_sample_id(raw_id: Any) -> str:
    return f"sam_clean::{str(raw_id or '').strip()}"


def _build_sam_clean_structured_row(sample: dict[str, Any], *, sample_id: str, source_file: Path) -> dict[str, Any] | None:
    image_path = str(sample.get("image_path") or "").strip()
    if not image_path:
        return None
    width, height = load_image_size(image_path)
    if width <= 0 or height <= 0:
        return None

    target_xy = to_float_xy(sample.get("target"))
    anchor_xy = to_float_xy(sample.get("anchor"))
    if target_xy is None or anchor_xy is None:
        return None

    target_px = convert_norm01_to_pixels(float(target_xy[0]), float(target_xy[1]), width, height)
    anchor_tag = build_target([anchor_xy], single_point_only=False)
    base_question = str(sample.get("question") or "").strip()
    if not base_question:
        return None
    question = f"{base_question}\n{anchor_tag}".strip()
    meta = sample.get("meta", {}) or {}
    label = str(meta.get("target_label") or "target").strip() or "target"

    metadata = {
        "category": "Steerable",
        "source": "sam_clean",
        "source_file": str(source_file.resolve()),
        "source_rank": None,
        "record_id": sample_id,
        "selected_root": "sam_clean_mix10000_v2",
        "selected_sample_path": str(source_file.resolve()),
        "selected_version_kind": "sam_clean_jsonl",
        "selected_completed_at": "",
        "workflow": "sam_clean_steerable_d",
        "original_query": base_question,
        "rewritten_query": question,
        "ground_truth_label": "",
        "classification_allowed_categories": list(STEERABLE_D_ACTIVE_CATEGORIES),
        "rewrite_finish_reason": "sam_clean_passthrough",
        "anchor_d_format": anchor_tag,
        "task_type": str(sample.get("task_type") or ""),
        "anchor_norm01": [float(anchor_xy[0]), float(anchor_xy[1])],
        "target_norm01": [float(target_xy[0]), float(target_xy[1])],
    }
    metadata.update({str(k): v for k, v in meta.items() if str(k) not in metadata})

    return {
        "id": sample_id,
        "image": image_path,
        "style": "pointing",
        "label": label,
        "question": question,
        "points": [(float(target_px[0]), float(target_px[1]))],
        "width": int(width),
        "height": int(height),
        "metadata": metadata,
    }


def _build_sam_clean_summary_rows(sam_clean_jsonl: str | Path) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    jsonl_path = Path(sam_clean_jsonl)
    if not jsonl_path.exists():
        return [], {"sam_clean_mix10000_v2": 0}, 0

    rows: list[dict[str, Any]] = []
    scanned_files = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            scanned_files += 1
            try:
                sample = json.loads(text)
            except Exception:
                continue
            sample_id = _sam_clean_sample_id(sample.get("id"))
            structured_row = _build_sam_clean_structured_row(sample, sample_id=sample_id, source_file=jsonl_path)
            history = [
                {
                    "root": "sam_clean_mix10000_v2",
                    "path": str(jsonl_path.resolve()),
                    "status": "completed" if structured_row is not None else "no_usable_judgement",
                    "category": "Steerable" if structured_row is not None else None,
                    "version_kind": "sam_clean_jsonl",
                    "decision_rank": 3,
                    "is_judged": structured_row is not None,
                    "timestamp": "",
                    "filter_stage": None,
                }
            ]
            row = {
                "sample_id": sample_id,
                "history": history,
                "history_count": 1,
                "status": "completed" if structured_row is not None else "no_usable_judgement",
                "category": "Steerable" if structured_row is not None else None,
                "selected_root": "sam_clean_mix10000_v2" if structured_row is not None else None,
                "selected_path": str(jsonl_path.resolve()) if structured_row is not None else None,
                "selected_version_kind": "sam_clean_jsonl" if structured_row is not None else None,
                "selected_timestamp": "",
                "selected_filter_stage": None,
                "source": "sam_clean",
                "source_file": str(jsonl_path.resolve()),
                "source_rank": None,
                "record_id": sample_id,
                "trainable": structured_row is not None,
                "row": structured_row,
            }
            rows.append(row)
    return rows, {"sam_clean_mix10000_v2": scanned_files}, scanned_files


def build_steerable_d_summary_cache(
    summary_path: str | Path = DEFAULT_STEERABLE_D_SUMMARY_PATH,
    *,
    pointarena_output_base: str | Path = POINTARENA_EXTRACT_DIR,
    pointarena_roots: Optional[Iterable[str]] = None,
    clean3_roots: Optional[Iterable[str | Path]] = None,
    sam_clean_jsonl: str | Path = DEFAULT_SAM_CLEAN_JSONL,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    pointarena_output_root = Path(pointarena_output_base)
    pointarena_root_names = list(pointarena_roots) if pointarena_roots is not None else discover_requested_roots(pointarena_output_root)
    clean3_root_paths = [Path(p) for p in (list(clean3_roots) if clean3_roots is not None else list(DEFAULT_CLEAN3_ROOTS))]
    pointarena_root_paths = [pointarena_output_root / root_name for root_name in pointarena_root_names]

    grouped, per_root_file_counts, scanned_files = _scan_candidate_groups(pointarena_root_paths + clean3_root_paths)
    summary_rows: list[dict[str, Any]] = []

    for sample_id in sorted(grouped.keys()):
        candidates = grouped[sample_id]
        chosen = choose_latest_candidate(candidates)
        history = summarize_history(candidates)
        summary_rows.append(
            _summary_row_from_candidate(
                chosen,
                sample_id=sample_id,
                history=history,
                spatial_relation_to_filtered=True,
            )
        )

    sam_clean_rows, sam_clean_file_counts, sam_clean_scanned = _build_sam_clean_summary_rows(sam_clean_jsonl)
    summary_rows.extend(sam_clean_rows)
    scanned_files += int(sam_clean_scanned)
    per_root_file_counts.update(sam_clean_file_counts)
    summary_rows = sorted(summary_rows, key=lambda item: str(item.get("sample_id") or ""))

    stats = _compute_summary_counts(summary_rows, per_root_file_counts)
    payload = {
        "summary_version": SUMMARY_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_root": str(PROJECT_ROOT),
        "pointarena_output_base": str(pointarena_output_root.resolve()),
        "pointarena_roots": pointarena_root_names,
        "clean3_roots": [str(path.resolve()) for path in clean3_root_paths],
        "sam_clean_jsonl": str(Path(sam_clean_jsonl).resolve()),
        "scanned_files": int(scanned_files),
        "unique_sample_ids": len(summary_rows),
        "active_categories": list(STEERABLE_D_ACTIVE_CATEGORIES),
        "selection_policy": {
            "pointarena_and_clean3": "prefer latest judged full workflow over legacy; filter completed spatial-relation rows out of active set",
            "sam_clean": "treat every accepted sample as Steerable and append D-form anchor tag to the question",
        },
        "trainable_completed_rows": stats["trainable_completed_rows"],
        "counts": stats["counts"],
        "rows": summary_rows,
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_steerable_d_summary_cache(
    summary_path: str | Path = DEFAULT_STEERABLE_D_SUMMARY_PATH,
    *,
    rebuild_if_missing: bool = True,
    pointarena_output_base: str | Path = POINTARENA_EXTRACT_DIR,
    pointarena_roots: Optional[Iterable[str]] = None,
    clean3_roots: Optional[Iterable[str | Path]] = None,
    sam_clean_jsonl: str | Path = DEFAULT_SAM_CLEAN_JSONL,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    if summary_file.exists():
        return json.loads(summary_file.read_text(encoding="utf-8"))
    if not rebuild_if_missing:
        raise FileNotFoundError(f"Missing steerable+D summary cache: {summary_file}")
    return build_steerable_d_summary_cache(
        summary_path=summary_file,
        pointarena_output_base=pointarena_output_base,
        pointarena_roots=pointarena_roots,
        clean3_roots=clean3_roots,
        sam_clean_jsonl=sam_clean_jsonl,
    )


class PointArenaSteerableDTrainingDataset(PointArenaRewrittenTrainingDataset):
    def __init__(
        self,
        summary_path: str | Path = DEFAULT_STEERABLE_D_SUMMARY_PATH,
        *,
        pointarena_output_base: str | Path = POINTARENA_EXTRACT_DIR,
        pointarena_roots: Optional[Iterable[str]] = None,
        clean3_roots: Optional[Iterable[str | Path]] = None,
        sam_clean_jsonl: str | Path = DEFAULT_SAM_CLEAN_JSONL,
        force_rebuild_summary: bool = False,
        rebuild_if_missing: bool = True,
        require_local_image: bool = True,
        verify_openable_image: bool = False,
        exclude_pointarena_eval: bool = True,
        point_mode: str = "all",
        max_target_points: int = 64,
        category_balance: Optional[dict[str, float]] = None,
        shuffle_selected: bool = False,
        selection_seed: int = 42,
    ) -> None:
        self.pointarena_output_base = Path(pointarena_output_base)
        self.pointarena_roots = list(pointarena_roots) if pointarena_roots is not None else None
        self.clean3_roots = [Path(p) for p in (list(clean3_roots) if clean3_roots is not None else list(DEFAULT_CLEAN3_ROOTS))]
        self.sam_clean_jsonl = Path(sam_clean_jsonl)
        summary_file = Path(summary_path)
        if force_rebuild_summary or not summary_file.exists():
            build_steerable_d_summary_cache(
                summary_path=summary_file,
                pointarena_output_base=self.pointarena_output_base,
                pointarena_roots=self.pointarena_roots,
                clean3_roots=self.clean3_roots,
                sam_clean_jsonl=self.sam_clean_jsonl,
            )

        effective_balance = dict(category_balance or {})
        super().__init__(
            summary_path=summary_file,
            output_base=self.pointarena_output_base,
            roots=self.pointarena_roots,
            force_rebuild_summary=False,
            rebuild_if_missing=False if not rebuild_if_missing else True,
            require_local_image=require_local_image,
            verify_openable_image=verify_openable_image,
            exclude_pointarena_eval=exclude_pointarena_eval,
            point_mode=point_mode,
            max_target_points=max_target_points,
            category_balance=effective_balance,
            shuffle_selected=shuffle_selected,
            selection_seed=selection_seed,
        )
