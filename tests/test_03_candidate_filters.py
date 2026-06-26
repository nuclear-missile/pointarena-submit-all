from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _count_jsonl_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def test_candidate_filter_previews_and_report() -> None:
    proc = subprocess.run(
        [
            "python",
            "scripts/03_build_candidate_filters.py",
            "--preview-size",
            "50",
            "--filter-d-threshold",
            "5",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = ROOT / "artifacts" / "robopoint_probe" / "candidate_filters"
    previews = [
        out_dir / "filter_a_preview.jsonl",
        out_dir / "filter_b_preview.jsonl",
        out_dir / "filter_c_preview.jsonl",
        out_dir / "filter_d_preview.jsonl",
    ]

    for p in previews:
        assert p.exists(), f"missing preview: {p}"
        assert _count_jsonl_lines(p) >= 20

    text = (out_dir / "filter_comparison.md").read_text(encoding="utf-8").lower()
    assert "precision proxy" in text
    assert "recall proxy" in text
    assert "recommended filter" in text
