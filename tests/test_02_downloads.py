from pathlib import Path

from scripts.paths import ROOT


def test_downloaded_roots_exist():
    root = ROOT / "data" / "raw"
    names = ["pixmo_points", "robopoint", "where2place", "refspatial"]
    for n in names:
        p = root / n
        assert p.exists(), f"missing dataset dir: {p}"


def test_hf_datasets_saved():
    root = ROOT / "data" / "raw"
    for n in ["pixmo_points", "where2place", "refspatial", "robopoint"]:
        p = root / n / "hf_saved"
        assert p.exists(), f"missing hf_saved: {p}"
