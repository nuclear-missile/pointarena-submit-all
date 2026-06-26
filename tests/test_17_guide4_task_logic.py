from __future__ import annotations

from make_steerable1.guide4.scripts.task_logic import (
    angle_delta_deg,
    axis_constrained_nearest_meta,
    direction_matches,
    other_object_anchor_candidates,
)


def test_angle_delta_deg_wraps_around_zero() -> None:
    assert angle_delta_deg(0.0, 0.0) == 0.0
    assert angle_delta_deg(0.0, 350.0) == 10.0
    assert angle_delta_deg(355.0, 5.0) == 10.0


def test_direction_matches_respects_tolerance() -> None:
    anchor = [20.0, 50.0]
    target = [80.0, 50.0]
    assert direction_matches(anchor, target, "right", tolerance_deg=10.0)
    assert not direction_matches(anchor, target, "up", tolerance_deg=10.0)


def test_other_object_anchor_candidates_keep_only_other_classes_with_matching_direction() -> None:
    target = {"object_id": "target", "same_class_key": "cup", "center_xy": [80.0, 50.0]}
    objects = [
        target,
        {"object_id": "same_class", "same_class_key": "cup", "center_xy": [30.0, 50.0]},
        {"object_id": "other_ok", "same_class_key": "book", "center_xy": [20.0, 50.0]},
        {"object_id": "other_wrong_dir", "same_class_key": "plate", "center_xy": [80.0, 10.0]},
        {"object_id": "other_duplicate_a", "same_class_key": "pen", "center_xy": [10.0, 50.0]},
        {"object_id": "other_duplicate_b", "same_class_key": "pen", "center_xy": [15.0, 50.0]},
    ]

    candidates = other_object_anchor_candidates(
        target_object_id=target["object_id"],
        target_same_class_key=target["same_class_key"],
        target_xy=target["center_xy"],
        objects=objects,
        direction="right",
        tolerance_deg=10.0,
        require_unique_class=True,
    )

    assert [item["object_id"] for item in candidates] == ["other_ok"]


def test_axis_constrained_nearest_rejects_closer_same_class_in_direction_sector() -> None:
    anchor = [100.0, 100.0]
    target = {"object_id": "target", "center_xy": [50.0, 100.0]}
    same = [
        target,
        {"object_id": "closer_left", "center_xy": [70.0, 100.0]},
        {"object_id": "far_down", "center_xy": [100.0, 180.0]},
    ]

    meta = axis_constrained_nearest_meta(
        target_object_id=target["object_id"],
        target_xy=target["center_xy"],
        same_label_objects=same,
        anchor_xy=anchor,
        direction="left",
        target_tolerance_deg=10.0,
        competitor_sector_deg=45.0,
    )

    assert not meta["pass"]
    assert meta["blocking_object_ids"] == ["closer_left"]


def test_axis_constrained_nearest_accepts_when_closer_objects_are_outside_direction_sector() -> None:
    anchor = [100.0, 100.0]
    target = {"object_id": "target", "center_xy": [50.0, 100.0]}
    same = [
        target,
        {"object_id": "closer_down", "center_xy": [100.0, 70.0]},
        {"object_id": "farther_left", "center_xy": [20.0, 100.0]},
    ]

    meta = axis_constrained_nearest_meta(
        target_object_id=target["object_id"],
        target_xy=target["center_xy"],
        same_label_objects=same,
        anchor_xy=anchor,
        direction="left",
        target_tolerance_deg=10.0,
        competitor_sector_deg=45.0,
    )

    assert meta["pass"]
    assert meta["blocking_object_ids"] == []
    assert meta["target_distance_px"] == 50.0
