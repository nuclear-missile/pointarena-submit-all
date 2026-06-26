import json

from scripts.paths import ROOT


def test_clean_outputs_exist():
    root = ROOT / "data" / "clean"
    for name in ["train.jsonl", "val.jsonl", "train_stats.json"]:
        assert (root / name).exists(), f"missing {name}"


def test_train_first_row_schema():
    root = ROOT / "data" / "clean" / "train.jsonl"
    with root.open("r", encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["dataset"] in ["pixmo_points", "robopoint", "where2place", "refspatial"]
    assert isinstance(row["points"], list)
