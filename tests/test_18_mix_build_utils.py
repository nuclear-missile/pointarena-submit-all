from __future__ import annotations

import pandas as pd

from make_steerable1.sam_clean.mix_build_utils import (
    assign_global_image_orders,
    compute_task_targets,
    select_one_candidate_per_image,
)


def make_candidate(
    candidate_id: str,
    image_id: str,
    task_type: str,
    image_order: int,
    object_order: int,
    task_rank: int,
):
    return {
        "candidate_id": candidate_id,
        "image_id": image_id,
        "task_type": task_type,
        "image_order": image_order,
        "object_order": object_order,
        "task_rank": task_rank,
        "direction_text": "",
    }


def test_compute_task_targets_for_10000():
    assert compute_task_targets(10000) == {
        "move_until_axis_fixed": 8500,
        "nearest_among_objects": 500,
        "nearest_among_objects_axis_constrained": 500,
        "move_then_axis_limited_fixed": 250,
        "move_until_axis_from_other_object": 250,
    }


def test_select_one_candidate_per_image_keeps_one_row_per_image_and_prioritizes_scarce_tasks():
    candidate_df = pd.DataFrame(
        [
            make_candidate("c1", "img_1", "move_until_axis_fixed", 0, 0, 0),
            make_candidate("c2", "img_1", "nearest_among_objects", 0, 0, 3),
            make_candidate("c3", "img_2", "move_until_axis_fixed", 1, 0, 0),
            make_candidate("c4", "img_2", "move_then_axis_limited_fixed", 1, 0, 2),
            make_candidate("c5", "img_3", "move_until_axis_fixed", 2, 0, 0),
            make_candidate("c6", "img_4", "nearest_among_objects_axis_constrained", 3, 0, 4),
        ]
    )
    remaining = {
        "move_until_axis_fixed": 10,
        "nearest_among_objects": 1,
        "nearest_among_objects_axis_constrained": 1,
        "move_then_axis_limited_fixed": 1,
        "move_until_axis_from_other_object": 0,
    }

    selected = select_one_candidate_per_image(candidate_df, remaining)

    assert len(selected) == 4
    assert selected["image_id"].nunique() == 4
    assert selected.groupby("image_id").size().max() == 1
    assert set(selected["task_type"]) == {
        "nearest_among_objects",
        "nearest_among_objects_axis_constrained",
        "move_then_axis_limited_fixed",
        "move_until_axis_fixed",
    }
    assert selected.loc[selected["image_id"] == "img_1", "task_type"].item() == "nearest_among_objects"
    assert selected.loc[selected["image_id"] == "img_2", "task_type"].item() == "move_then_axis_limited_fixed"


def test_assign_global_image_orders_only_maps_selected_images():
    object_df = pd.DataFrame(
        [
            {"object_id": "o1", "image_id": "img_keep_1", "image_order": 0, "object_order": 0},
            {"object_id": "o2", "image_id": "img_keep_2", "image_order": 1, "object_order": 0},
            {"object_id": "o3", "image_id": "img_drop", "image_order": 2, "object_order": 0},
        ]
    )
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": "c1", "image_id": "img_keep_1", "image_order": 0, "object_order": 0, "task_rank": 0},
            {"candidate_id": "c2", "image_id": "img_keep_2", "image_order": 1, "object_order": 0, "task_rank": 1},
        ]
    )

    kept_object_df, kept_candidate_df, selected_image_ids, next_image_order = assign_global_image_orders(
        object_df,
        candidate_df,
        image_order_map={},
        next_image_order=0,
    )

    assert selected_image_ids == ["img_keep_1", "img_keep_2"]
    assert next_image_order == 2
    assert kept_object_df["image_id"].tolist() == ["img_keep_1", "img_keep_2"]
    assert kept_object_df["image_order"].tolist() == [0, 1]
    assert kept_candidate_df["image_order"].tolist() == [0, 1]
