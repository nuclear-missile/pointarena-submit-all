from __future__ import annotations

from pathlib import Path

from scripts.train.coordinate_qa_dataset import CoordinateQADataset


def test_coordinate_qa_dataset_smoke() -> None:
    path = Path('artifacts/clean_full/merged_coords_local.jsonl')
    assert path.exists(), 'merged_coords_local.jsonl missing; run prepare_full_coordinate_qa.py first'

    ds = CoordinateQADataset(default_path=str(path), require_local_image=True, return_pil=False)
    assert len(ds) > 0

    item = ds[0]
    assert 'uid' in item and item['uid']
    assert 'dataset' in item and item['dataset']
    assert isinstance(item.get('points_norm', []), list)
