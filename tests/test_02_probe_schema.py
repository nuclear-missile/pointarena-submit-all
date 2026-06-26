from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_probe_outputs_exist_and_report_has_required_terms() -> None:
    proc = subprocess.run(
        [
            "python",
            "scripts/02_probe_schema.py",
            "--head",
            "200",
            "--sample",
            "200",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = ROOT / "artifacts" / "robopoint_probe"
    field_stats = out_dir / "field_stats.json"
    report = out_dir / "probe_report.md"
    sample_rows = out_dir / "sample_rows.jsonl"

    assert field_stats.exists()
    payload = json.loads(field_stats.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and len(payload) > 0

    text = report.read_text(encoding="utf-8").lower()
    assert "object reference" in text
    assert "free space reference" in text
    assert "explicit label" in text
    assert "fallback signal" in text

    lines = [ln for ln in sample_rows.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 50
