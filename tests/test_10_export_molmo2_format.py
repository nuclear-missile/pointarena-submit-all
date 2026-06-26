import json

from scripts.paths import ROOT


def test_export_format():
    p = ROOT / "outputs" / "molmo2_train.jsonl"
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert "messages" in row
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
