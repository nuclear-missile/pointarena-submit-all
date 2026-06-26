from __future__ import annotations

import math
from collections import Counter
from typing import Any


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


def angle_delta_deg(a: float, b: float) -> float:
    diff = abs((float(a) - float(b)) % 360.0)
    return round(min(diff, 360.0 - diff), 6)


def direction_angle_deg(anchor_xy: list[float], target_xy: list[float]) -> float:
    dx = float(target_xy[0]) - float(anchor_xy[0])
    dy = float(target_xy[1]) - float(anchor_xy[1])
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def direction_matches(anchor_xy: list[float], target_xy: list[float], direction: str, tolerance_deg: float = 10.0) -> bool:
    base = DIRECTION_ANGLES[direction]
    actual = direction_angle_deg(anchor_xy, target_xy)
    return angle_delta_deg(actual, base) <= float(tolerance_deg)


def euclidean_distance(a_xy: list[float], b_xy: list[float]) -> float:
    return float(math.hypot(float(a_xy[0]) - float(b_xy[0]), float(a_xy[1]) - float(b_xy[1])))


def other_object_anchor_candidates(
    target_object_id: str,
    target_same_class_key: str,
    target_xy: list[float],
    objects: list[dict[str, Any]],
    direction: str,
    tolerance_deg: float = 10.0,
    require_unique_class: bool = True,
) -> list[dict[str, Any]]:
    counts = Counter(str(obj.get("same_class_key") or "") for obj in objects)
    candidates: list[dict[str, Any]] = []
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        same_class_key = str(obj.get("same_class_key") or "")
        center_xy = obj.get("center_xy") or []
        if object_id == target_object_id:
            continue
        if same_class_key == target_same_class_key:
            continue
        if require_unique_class and counts.get(same_class_key, 0) != 1:
            continue
        if not center_xy or len(center_xy) != 2:
            continue
        if not direction_matches(center_xy, target_xy, direction, tolerance_deg=tolerance_deg):
            continue
        candidates.append(
            {
                **obj,
                "distance_to_target_px": round(euclidean_distance(center_xy, target_xy), 3),
                "direction_delta_deg": round(
                    angle_delta_deg(direction_angle_deg(center_xy, target_xy), DIRECTION_ANGLES[direction]),
                    3,
                ),
            }
        )
    candidates.sort(key=lambda item: (float(item["distance_to_target_px"]), str(item.get("object_id") or "")))
    return candidates


def axis_constrained_nearest_meta(
    target_object_id: str,
    target_xy: list[float],
    same_label_objects: list[dict[str, Any]],
    anchor_xy: list[float],
    direction: str,
    target_tolerance_deg: float = 10.0,
    competitor_sector_deg: float = 45.0,
) -> dict[str, Any]:
    target_distance = euclidean_distance(anchor_xy, target_xy)
    target_angle = direction_angle_deg(anchor_xy, target_xy)
    target_delta = angle_delta_deg(target_angle, DIRECTION_ANGLES[direction])
    blocking: list[dict[str, Any]] = []
    sector_objects: list[dict[str, Any]] = []

    for other in same_label_objects:
        object_id = str(other.get("object_id") or "")
        center_xy = other.get("center_xy") or []
        if object_id == target_object_id or not center_xy or len(center_xy) != 2:
            continue
        other_distance = euclidean_distance(anchor_xy, center_xy)
        other_angle = direction_angle_deg(anchor_xy, center_xy)
        other_delta = angle_delta_deg(other_angle, DIRECTION_ANGLES[direction])
        if other_delta <= float(competitor_sector_deg):
            sector_objects.append(
                {
                    "object_id": object_id,
                    "distance_px": round(other_distance, 3),
                    "direction_delta_deg": round(other_delta, 3),
                }
            )
            if other_distance < target_distance:
                blocking.append(
                    {
                        "object_id": object_id,
                        "distance_px": round(other_distance, 3),
                        "direction_delta_deg": round(other_delta, 3),
                    }
                )

    blocking.sort(key=lambda item: (float(item["distance_px"]), str(item["object_id"])))
    sector_objects.sort(key=lambda item: (float(item["distance_px"]), str(item["object_id"])))
    passed = target_delta <= float(target_tolerance_deg) and not blocking
    return {
        "pass": passed,
        "target_distance_px": round(target_distance, 3),
        "target_direction_delta_deg": round(target_delta, 3),
        "target_within_direction_tolerance": target_delta <= float(target_tolerance_deg),
        "direction_sector_deg": float(competitor_sector_deg),
        "blocking_object_ids": [item["object_id"] for item in blocking],
        "blocking_objects": blocking,
        "sector_same_class_objects": sector_objects,
    }
