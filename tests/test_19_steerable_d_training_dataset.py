from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.train.pointarena_rewritten_dataset_steerable_d import (
    PointArenaSteerableDTrainingDataset,
    build_steerable_d_summary_cache,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_image(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(path)


def _full_completed(
    *,
    record_id: str,
    category: str,
    rewritten_query: str,
    completed_at: str,
    image_path: str,
    width: int,
    height: int,
    points_abs: list[list[float]],
    source: str = "pixmo_points",
) -> dict:
    return {
        "processing": {
            "status": "completed",
            "completed_at": completed_at,
            "source": source,
            "source_file": f"/tmp/{source}.jsonl",
            "source_rank": 1,
            "workflow": "cleaning->multipoint->classification_or_forced_counting->rewrite_same_session",
        },
        "record": {
            "id": record_id,
            "source": source,
            "image_path": image_path,
            "query": "original query",
            "width": width,
            "height": height,
            "points_abs": points_abs,
        },
        "cleaning": {
            "keep": True,
            "core_target": "the target",
            "reason": "valid",
            "parsed": {
                "keep": True,
                "core_target": "the target",
                "reason": "valid",
            },
        },
        "multipoint": {
            "multi_point": False,
            "reason": "single point",
            "parsed": {
                "multi_point": False,
                "reason": "single point",
            },
        },
        "classification": {
            "category": category,
            "thought": "latest result",
            "allowed_categories": [
                "Reasoning",
                "Spatial Relation",
                "Affordance",
                "Counting",
                "Object Reference",
            ],
        },
        "rewrite": {
            "rewritten_query": rewritten_query,
            "finish_reason": "stop",
        },
        "session_messages": [1, 2, 3, 4, 5, 6],
    }


def _sam_clean_row(*, row_id: str, image_path: str, question: str, target_xy: list[float], anchor_xy: list[float]) -> dict:
    return {
        "id": row_id,
        "candidate_id": row_id + "::candidate",
        "image_id": "img_" + row_id,
        "image_path": image_path,
        "task_type": "move_until_axis_fixed",
        "question": question,
        "target": target_xy,
        "anchor": anchor_xy,
        "meta": {
            "target_label": "white stove",
            "direction_text": "right",
            "first_axis": "right",
        },
    }


def test_build_steerable_d_summary_maps_sam_clean_to_steerable_and_uses_d_anchor_format(tmp_path: Path) -> None:
    pointarena_base = tmp_path / "pointarena_extract"
    clean3_root = tmp_path / "clean_3types_local" / "qwen3_8b_three_types_add2000_nodup_reverse"
    sam_clean_jsonl = tmp_path / "make_steerable1" / "sam_clean" / "outputs" / "mix10000_v2" / "07_final" / "accepted_samples.jsonl"
    summary_path = tmp_path / "summary_steerable_d.json"

    pointarena_img = tmp_path / "images" / "pointarena_aff.jpg"
    clean3_img = tmp_path / "images" / "clean3_reason.jpg"
    sam_img = tmp_path / "images" / "sam_img.jpg"
    _write_image(pointarena_img, width=100, height=80)
    _write_image(clean3_img, width=120, height=90)
    _write_image(sam_img, width=200, height=100)

    _write_json(
        pointarena_base / "pixmo_points_len1_no_counting_10000_incremental" / "pixmo_points" / "aff.json",
        _full_completed(
            record_id="aff_1",
            category="Affordance",
            rewritten_query="Point to the handle.",
            completed_at="2026-04-20T00:00:00Z",
            image_path=str(pointarena_img),
            width=100,
            height=80,
            points_abs=[[10.0, 20.0]],
        ),
    )
    _write_json(
        pointarena_base / "pixmo_points_len1_no_counting_10000_incremental" / "pixmo_points" / "spatial.json",
        _full_completed(
            record_id="spatial_1",
            category="Spatial Relation",
            rewritten_query="Point to the cup on the left.",
            completed_at="2026-04-20T00:00:00Z",
            image_path=str(pointarena_img),
            width=100,
            height=80,
            points_abs=[[11.0, 21.0]],
        ),
    )
    _write_json(
        clean3_root / "pixmo_points" / "reason.json",
        _full_completed(
            record_id="reason_1",
            category="Reasoning",
            rewritten_query="Point to the tool used for cutting wood.",
            completed_at="2026-04-21T00:00:00Z",
            image_path=str(clean3_img),
            width=120,
            height=90,
            points_abs=[[30.0, 40.0]],
            source="pixmo_points",
        ),
    )
    _write_jsonl(
        sam_clean_jsonl,
        [
            _sam_clean_row(
                row_id="sam_1",
                image_path=str(sam_img),
                question="Use the blue point as your reference and point to the white stove to the right of it.",
                target_xy=[0.75, 0.2],
                anchor_xy=[0.25, 0.5],
            )
        ],
    )

    summary = build_steerable_d_summary_cache(
        summary_path=summary_path,
        pointarena_output_base=pointarena_base,
        pointarena_roots=["pixmo_points_len1_no_counting_10000_incremental"],
        clean3_roots=[clean3_root],
        sam_clean_jsonl=sam_clean_jsonl,
    )
    rows = {row["sample_id"]: row for row in summary["rows"]}

    assert rows["aff_1"]["status"] == "completed"
    assert rows["reason_1"]["status"] == "completed"
    assert rows["sam_clean::sam_1"]["status"] == "completed"
    assert rows["spatial_1"]["status"] == "filtered_out"

    ds = PointArenaSteerableDTrainingDataset(
        summary_path=summary_path,
        pointarena_output_base=pointarena_base,
        pointarena_roots=["pixmo_points_len1_no_counting_10000_incremental"],
        clean3_roots=[clean3_root],
        sam_clean_jsonl=sam_clean_jsonl,
        rebuild_if_missing=False,
        require_local_image=False,
    )
    assert ds.category_counts(active_only=False) == {
        "Affordance": 1,
        "Reasoning": 1,
        "Steerable": 1,
    }

    items = {item["id"]: item for item in ds}
    steerable = items["sam_clean::sam_1"]
    assert steerable["metadata"]["category"] == "Steerable"
    assert steerable["question"].startswith("Use the blue point as your reference")
    assert "<point>250,500</point>" in steerable["question"]
    assert steerable["points"] == [(149.25, 19.8)]


def test_steerable_d_dataset_keeps_five_way_balance_and_rotates_larger_classes(tmp_path: Path) -> None:
    pointarena_base = tmp_path / "pointarena_extract"
    clean3_root = tmp_path / "clean_3types_local" / "qwen3_8b_three_types_add2000_nodup_reverse"
    sam_clean_jsonl = tmp_path / "make_steerable1" / "sam_clean" / "outputs" / "mix10000_v2" / "07_final" / "accepted_samples.jsonl"
    summary_path = tmp_path / "summary_steerable_d.json"
    img_dir = tmp_path / "images"

    def make_img(name: str, w: int = 128, h: int = 96) -> str:
        path = img_dir / name
        _write_image(path, width=w, height=h)
        return str(path)

    aff_img = make_img("aff.jpg")
    count_img = make_img("count.jpg")
    obj_img = make_img("obj.jpg")
    reason_img = make_img("reason.jpg")
    steer_img = make_img("steer.jpg", w=200, h=100)

    for index in range(3):
        _write_json(
            pointarena_base / "pixmo_points_len1_no_counting_10000_incremental" / "pixmo_points" / f"aff_{index}.json",
            _full_completed(
                record_id=f"aff_{index}",
                category="Affordance",
                rewritten_query=f"Point to affordance {index}.",
                completed_at=f"2026-04-2{index}T00:00:00Z",
                image_path=aff_img,
                width=128,
                height=96,
                points_abs=[[10.0 + index, 20.0 + index]],
            ),
        )
    for index in range(2):
        _write_json(
            pointarena_base / "mix4_4000_clean_multipoint_counting_same_session_rulefilter" / "pixmo_points" / f"count_{index}.json",
            _full_completed(
                record_id=f"count_{index}",
                category="Counting",
                rewritten_query=f"Point to all counts {index}.",
                completed_at=f"2026-04-1{index}T00:00:00Z",
                image_path=count_img,
                width=128,
                height=96,
                points_abs=[[30.0 + index, 40.0 + index]],
            ),
        )
    for index in range(2):
        _write_json(
            clean3_root / "pixmo_points" / f"obj_{index}.json",
            _full_completed(
                record_id=f"obj_{index}",
                category="Object Reference",
                rewritten_query=f"Point to object {index}.",
                completed_at=f"2026-04-0{index}T00:00:00Z",
                image_path=obj_img,
                width=128,
                height=96,
                points_abs=[[50.0 + index, 60.0 + index]],
                source="pixmo_points",
            ),
        )
    for index in range(2):
        _write_json(
            clean3_root / "pixmo_points" / f"reason_{index}.json",
            _full_completed(
                record_id=f"reason_{index}",
                category="Reasoning",
                rewritten_query=f"Point to reason {index}.",
                completed_at=f"2026-04-0{index}T12:00:00Z",
                image_path=reason_img,
                width=128,
                height=96,
                points_abs=[[70.0 + index, 80.0 + index]],
                source="pixmo_points",
            ),
        )
    _write_jsonl(
        sam_clean_jsonl,
        [
            _sam_clean_row(
                row_id=f"sam_{index}",
                image_path=steer_img,
                question=f"Use the blue point as your reference and point to target {index}.",
                target_xy=[0.2 + 0.1 * index, 0.3],
                anchor_xy=[0.1 * (index + 1), 0.5],
            )
            for index in range(4)
        ],
    )

    ds = PointArenaSteerableDTrainingDataset(
        summary_path=summary_path,
        pointarena_output_base=pointarena_base,
        pointarena_roots=[
            "pixmo_points_len1_no_counting_10000_incremental",
            "mix4_4000_clean_multipoint_counting_same_session_rulefilter",
        ],
        clean3_roots=[clean3_root],
        sam_clean_jsonl=sam_clean_jsonl,
        force_rebuild_summary=True,
        require_local_image=False,
        category_balance={
            "Affordance": 1,
            "Counting": 1,
            "Object Reference": 1,
            "Reasoning": 1,
            "Steerable": 1,
        },
    )

    assert ds.category_counts() == {
        "Affordance": 2,
        "Counting": 2,
        "Object Reference": 2,
        "Reasoning": 2,
        "Steerable": 2,
    }

    before_ids = {item["id"] for item in ds if item["metadata"]["category"] == "Steerable"}
    ds.rotate_balanced_samples()
    after_ids = {item["id"] for item in ds if item["metadata"]["category"] == "Steerable"}

    assert ds.category_counts() == {
        "Affordance": 2,
        "Counting": 2,
        "Object Reference": 2,
        "Reasoning": 2,
        "Steerable": 2,
    }
    assert before_ids != after_ids
