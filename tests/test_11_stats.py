import json

from scripts.paths import ROOT


def test_stats_keys():
    p = ROOT / "data" / "clean" / "train_stats.json"
    stats = json.load(open(p, "r", encoding="utf-8"))
    for k in ["dataset_counts", "task_type_counts", "api_reject_rate", "dedup_rate"]:
        assert k in stats
