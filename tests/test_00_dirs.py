from pathlib import Path

from scripts.paths import ROOT


def test_required_dirs_exist():
    required = [
        ROOT / "data" / "raw" / "pixmo_points",
        ROOT / "data" / "raw" / "robopoint",
        ROOT / "data" / "raw" / "where2place",
        ROOT / "data" / "raw" / "refspatial",
        ROOT / "data" / "clean",
        ROOT / "scripts" / "download",
        ROOT / "scripts" / "normalize",
        ROOT / "scripts" / "clean",
        ROOT / "scripts" / "api_filter",
        ROOT / "scripts" / "validate",
        ROOT / "tests",
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    assert not missing, f"missing dirs: {missing}"
