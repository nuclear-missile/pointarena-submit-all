from __future__ import annotations

from pathlib import Path

from src.data._common import iter_jsonl


def test_train_all_and_val_exist_and_nonempty():
    train = Path("data/unified/train_all.jsonl")
    val = Path("data/unified/val_pointarena.jsonl")
    assert train.exists()
    assert val.exists()

    first_train = next(iter_jsonl(train))
    first_val = next(iter_jsonl(val))

    assert first_train["id"]
    assert first_val["source"] == "pointarena_val"
    assert first_val["meta"].get("split") == "val"


def test_train_ids_unique_on_sample():
    train = Path("data/unified/train_all.jsonl")
    ids = set()
    for i, row in enumerate(iter_jsonl(train)):
        rid = row["id"]
        assert rid not in ids
        ids.add(rid)
        if i >= 5000:
            break
