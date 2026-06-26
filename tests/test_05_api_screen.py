from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_api_screen_dry_run_outputs_contract() -> None:
    proc = subprocess.run(
        ["python", "scripts/05_api_screen.py", "--mode", "dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    out = ROOT / "artifacts" / "robopoint_clean" / "api_screen_decisions.jsonl"
    assert out.exists()

    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) > 0

    row = json.loads(lines[0])
    for k in ["keep", "predicted_task", "confidence"]:
        assert k in row
