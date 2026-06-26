from __future__ import annotations

import json
from pathlib import Path


def test_context_stats_file_exists_and_valid():
    p = Path("data/unified/stats/max_length_stats.json")
    assert p.exists(), "run scripts/01_prepare_data.sh first"

    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["num_samples"] > 0
    assert obj["chosen_context_len"] > 0

    max_total = obj["total_len"]["max_total_len"]
    chosen = obj["chosen_context_len"]
    if not obj.get("used_truncation_rule", False):
        assert chosen >= max_total
