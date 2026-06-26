from __future__ import annotations

from pathlib import Path

from src.data._common import iter_jsonl


def test_pixmo_output_exists_and_valid():
    p = Path("data/unified/train_pixmo.jsonl")
    assert p.exists(), "run scripts/01_prepare_data.sh first"

    first = next(iter_jsonl(p))
    assert first["source"] == "pixmo_points"
    assert first["target"]["type"] == "point"
    assert first["query"].strip()

    x, y = first["target"]["points"][0]
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
