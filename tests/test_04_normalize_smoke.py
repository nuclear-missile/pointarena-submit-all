import json
from pathlib import Path

from scripts.paths import ROOT


def _has_rows(p: Path):
    assert p.exists(), f"missing {p}"
    with p.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    assert first, f"empty file: {p}"
    obj = json.loads(first)
    assert "uid" in obj and "query" in obj and "dataset" in obj


def test_interim_files_exist_and_nonempty():
    root = ROOT / "data" / "interim"
    for name in ["pixmo_points.jsonl", "where2place.jsonl", "robopoint.jsonl", "refspatial.jsonl"]:
        _has_rows(root / name)
