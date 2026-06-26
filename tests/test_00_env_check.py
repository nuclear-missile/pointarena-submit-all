from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_env_check_runs_and_writes_report() -> None:
    proc = subprocess.run(["python", "scripts/00_env_check.py"], cwd=ROOT, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    out = ROOT / "artifacts" / "logs" / "env_check.json"
    assert out.exists(), "env_check.json not found"

    payload = json.loads(out.read_text(encoding="utf-8"))
    for k in ["cwd", "resource_dir", "python_version", "writable"]:
        assert k in payload
