import numpy as np

from scripts.clean.mask_to_points import mask_to_points


def test_mask_to_points_rectangle():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 3:7] = 1
    pts = mask_to_points(mask)
    assert "medoid" in pts
    assert "random_inside" in pts


def test_mask_to_points_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    pts = mask_to_points(mask)
    assert pts == []
