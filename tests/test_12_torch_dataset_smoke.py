from scripts.train.pointarena_torch_dataset import PointArenaPointingDataset


def test_dataset_smoke_return_pil():
    ds = PointArenaPointingDataset(split="train", return_pil=True, require_local_image=True)
    assert len(ds) > 0
    item = ds[0]
    assert "image" in item and hasattr(item["image"], "size")
    assert isinstance(item["points_norm"], list)
    assert isinstance(item["points_xy"], list)
