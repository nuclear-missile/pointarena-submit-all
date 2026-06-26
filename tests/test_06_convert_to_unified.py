from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_convert_unified_output_exists_and_schema_is_valid_when_non_empty() -> None:
    proc = subprocess.run(["python", "scripts/06_convert_to_unified.py"], cwd=ROOT, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    out = ROOT / "artifacts" / "robopoint_clean" / "robopoint_obj_free_unified.jsonl"
    summary = ROOT / "artifacts" / "robopoint_clean" / "convert_summary.json"
    assert out.exists()
    assert summary.exists()

    rows = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if rows:
        row = json.loads(rows[0])
        assert row["task_type"] in {"object_ref", "free_space_ref"}
        assert Path(row["image"]).is_absolute()
        assert isinstance(row["target_points"], list) and len(row["target_points"]) > 0
    else:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload.get("dropped_no_points", 0) >= 0
