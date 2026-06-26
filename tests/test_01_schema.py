from __future__ import annotations

from src.data.schema import validate_record


def test_schema_validation_ok():
    row = {
        "id": "x",
        "source": "pixmo_points",
        "task_type": "object_reference",
        "image_path": "/tmp/a.jpg",
        "query": "Point to the cup",
        "target": {
            "type": "point",
            "points": [[12.0, 34.0]],
            "point_format": "xy_abs_pixels",
            "image_size": [100, 80],
        },
        "meta": {"split": "train"},
    }
    vr = validate_record(row)
    assert vr.ok
