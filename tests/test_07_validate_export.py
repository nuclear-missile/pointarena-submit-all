from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_export_writes_reports() -> None:
    proc = subprocess.run(["python", "scripts/07_validate_export.py"], cwd=ROOT, check=False, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr

    out_dir = ROOT / "artifacts" / "robopoint_clean" / "validation"
    summary = out_dir / "summary.json"
    report = out_dir / "validation_report.md"
    assert summary.exists()
    assert report.exists()

    payload = json.loads(summary.read_text(encoding="utf-8"))
    total = int(payload.get("total_samples", 0))
    if total > 0:
        assert payload.get("missing_image", 0) == 0
        assert payload.get("invalid_task_type", 0) == 0
