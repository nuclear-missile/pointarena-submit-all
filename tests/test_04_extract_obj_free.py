from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def test_extract_candidates_outputs_exist_and_have_required_fields() -> None:
    proc = subprocess.run(["python", "scripts/04_extract_obj_free.py"], cwd=ROOT, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    out_dir = ROOT / "artifacts" / "robopoint_clean"
    obj = out_dir / "obj_ref_candidates.jsonl"
    free = out_dir / "free_space_candidates.jsonl"
    union = out_dir / "obj_free_union_candidates.jsonl"

    assert obj.exists()
    assert free.exists()
    assert union.exists()

    obj_rows = _read_jsonl(obj)
    free_rows = _read_jsonl(free)
    union_rows = _read_jsonl(union)

    assert len(union_rows) > 0
    assert len(obj_rows) + len(free_rows) == len(union_rows)

    for row in union_rows[:10]:
        assert "task_guess" in row
        assert "image_abs" in row
        assert "user_text" in row
