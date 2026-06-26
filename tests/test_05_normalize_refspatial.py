from __future__ import annotations

from pathlib import Path

from src.data._common import iter_jsonl


def test_refspatial_output_exists_and_valid():
    p = Path("data/unified/train_refspatial.jsonl")
    assert p.exists(), "run scripts/01_prepare_data.sh first"

    first = next(iter_jsonl(p))
    assert first["source"] == "refspatial"
    assert first["task_type"] in {"spatial_relation", "reasoning", "free_space_reference"}
