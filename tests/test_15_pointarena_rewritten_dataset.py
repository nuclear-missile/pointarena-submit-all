from __future__ import annotations

import json
from pathlib import Path

from src.train.pointarena_rewritten_dataset import (
    PointArenaRewrittenTrainingDataset,
    build_summary_cache,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _legacy_completed(*, record_id: str, category: str, rewritten_query: str, points_abs: list[list[float]], width: int, height: int, completed_at: str) -> dict:
    return {
        "processing": {
            "status": "completed",
            "completed_at": completed_at,
            "source": "pixmo_points",
            "source_file": "/tmp/pixmo_points.jsonl",
            "source_rank": 1,
        },
        "record": {
            "id": record_id,
            "source": "pixmo_points",
            "image_path": f"/tmp/{record_id}.jpg",
            "query": "original query",
            "points_abs": points_abs,
            "width": width,
            "height": height,
        },
        "classification": {
            "category": category,
            "thought": "legacy result",
        },
        "rewrite": {
            "rewritten_query": rewritten_query,
        },
        "session_messages": [],
    }


def _full_completed(
    *,
    record_id: str,
    category: str,
    rewritten_query: str,
    completed_at: str,
    width: int | None = None,
    height: int | None = None,
    points_abs: list[list[float]] | None = None,
    target_points: list[list[float]] | None = None,
    image_size: list[int] | None = None,
) -> dict:
    record = {
        "id": record_id,
        "source": "pixmo_points",
        "image_path": f"/tmp/{record_id}.jpg",
        "query": "original query",
    }
    if width is not None:
        record["width"] = width
    if height is not None:
        record["height"] = height
    if points_abs is not None:
        record["points_abs"] = points_abs
    if target_points is not None:
        record["target"] = {
            "type": "point",
            "points": target_points,
            "point_format": "xy_norm_01",
            "image_size": image_size or [width or 0, height or 0],
        }

    return {
        "processing": {
            "status": "completed",
            "completed_at": completed_at,
            "source": "pixmo_points",
            "source_file": "/tmp/pixmo_points.jsonl",
            "source_rank": 1,
            "workflow": "cleaning->multipoint->classification_or_forced_counting->rewrite_same_session",
        },
        "record": record,
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


def _full_filtered(*, record_id: str, completed_at: str) -> dict:
    return {
        "processing": {
            "status": "filtered_out",
            "filter_stage": "cleaning",
            "completed_at": completed_at,
            "source": "pixmo_points",
            "source_file": "/tmp/pixmo_points.jsonl",
            "source_rank": 1,
            "workflow": "cleaning->multipoint->classification_or_forced_counting->rewrite_same_session",
        },
        "record": {
            "id": record_id,
            "source": "pixmo_points",
            "image_path": f"/tmp/{record_id}.jpg",
            "query": "old query",
            "points_abs": [[1.0, 1.0]],
            "width": 10,
            "height": 10,
        },
        "cleaning": {
            "keep": False,
            "core_target": "",
            "reason": "rejected by latest cleaning",
            "parsed": {
                "keep": False,
                "core_target": "",
                "reason": "rejected by latest cleaning",
            },
        },
        "multipoint": {},
        "classification": {},
        "rewrite": {},
        "session_messages": [1, 2],
    }


def test_summary_prefers_latest_full_workflow_and_uses_rewritten_query(tmp_path: Path) -> None:
    output_base = tmp_path / "pointarena_extract"
    summary_path = tmp_path / "summary.json"
    roots = [
        "mix4_first100_same_session",
        "mix4_pointsabs_len1_10000_rule_llm",
        "pointarena_full_clean_multipoint_counting_same_session",
    ]

    _write_json(
        output_base / "mix4_first100_same_session" / "pixmo_points" / "legacy_same_sample.json",
        _legacy_completed(
            record_id="dup_sample",
            category="Object Reference",
            rewritten_query="Point to the old object.",
            points_abs=[[12.0, 18.0]],
            width=100,
            height=50,
            completed_at="2026-03-20T00:00:00Z",
        ),
    )
    _write_json(
        output_base / "mix4_pointsabs_len1_10000_rule_llm" / "pixmo_points" / "filtered_same_sample.json",
        _full_filtered(record_id="dup_sample", completed_at="2026-03-21T00:00:00Z"),
    )
    _write_json(
        output_base / "mix4_first100_same_session" / "pixmo_points" / "legacy_only.json",
        _legacy_completed(
            record_id="legacy_only",
            category="Object Reference",
            rewritten_query="Point to the mug.",
            points_abs=[[14.0, 22.0]],
            width=120,
            height=80,
            completed_at="2026-03-20T00:00:00Z",
        ),
    )
    _write_json(
        output_base / "pointarena_full_clean_multipoint_counting_same_session" / "reasoning" / "full_pointarena.json",
        _full_completed(
            record_id="pointarena_norm",
            category="Reasoning",
            rewritten_query="Point to the mug handle.",
            completed_at="2026-03-22T00:00:00Z",
            target_points=[[0.5, 0.5]],
            image_size=[101, 51],
        ),
    )

    summary = build_summary_cache(summary_path, output_base=output_base, roots=roots)
    rows = {row["sample_id"]: row for row in summary["rows"]}

    assert rows["dup_sample"]["status"] == "filtered_out"
    assert rows["legacy_only"]["status"] == "completed"
    assert rows["pointarena_norm"]["status"] == "completed"

    ds = PointArenaRewrittenTrainingDataset(
        summary_path=summary_path,
        output_base=output_base,
        roots=roots,
        rebuild_if_missing=False,
        require_local_image=False,
    )
    assert len(ds) == 2

    items = {item["id"]: item for item in ds}
    assert items["legacy_only"]["question"] == "Point to the mug."
    assert items["legacy_only"]["points"] == [(14.0, 22.0)]
    assert items["pointarena_norm"]["question"] == "Point to the mug handle."
    assert items["pointarena_norm"]["points"] == [(50.0, 25.0)]


def test_balanced_selection_and_rotation(tmp_path: Path) -> None:
    output_base = tmp_path / "pointarena_extract"
    summary_path = tmp_path / "summary.json"
    roots = ["mix4_pointsabs_len1_10000_rule_llm"]

    for index in range(4):
        _write_json(
            output_base / roots[0] / "pixmo_points" / f"obj_{index}.json",
            _full_completed(
                record_id=f"obj_{index}",
                category="Object Reference",
                rewritten_query=f"Point to object {index}.",
                completed_at=f"2026-03-2{index}T00:00:00Z",
                width=100,
                height=100,
                points_abs=[[float(index), float(index + 1)]],
            ),
        )
    for index in range(2):
        _write_json(
            output_base / roots[0] / "pixmo_points" / f"reason_{index}.json",
            _full_completed(
                record_id=f"reason_{index}",
                category="Reasoning",
                rewritten_query=f"Point to reason {index}.",
                completed_at=f"2026-03-1{index}T00:00:00Z",
                width=100,
                height=100,
                points_abs=[[float(index + 10), float(index + 11)]],
            ),
        )

    ds = PointArenaRewrittenTrainingDataset(
        summary_path=summary_path,
        output_base=output_base,
        roots=roots,
        force_rebuild_summary=True,
        require_local_image=False,
        category_balance={"Object Reference": 1, "Reasoning": 1},
    )

    assert len(ds) == 4
    assert ds.category_counts() == {"Object Reference": 2, "Reasoning": 2}
    assert len(ds.discarded_rows) == 2

    before_ids = {item["id"] for item in ds if item["metadata"]["category"] == "Object Reference"}
    ds.rotate_balanced_samples()
    after_ids = {item["id"] for item in ds if item["metadata"]["category"] == "Object Reference"}

    assert ds.category_counts() == {"Object Reference": 2, "Reasoning": 2}
    assert before_ids != after_ids
