from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_export_for_pointarena_parsing(tmp_path: Path):
    inp = tmp_path / "gen.jsonl"
    out = tmp_path / "pred.jsonl"
    rows = [
        {"id": "a", "image_size": [100, 80], "model_output": "<point>12.7,34.2</point>"},
        {"id": "b", "image_size": [100, 80], "model_output": "bad output"},
    ]
    inp.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    subprocess.run(["python", "-m", "src.eval.export_for_pointarena", "--inputs", str(inp), "--output", str(out)], check=True)

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines[0]["pred_point"] == [13, 34]
    assert lines[1]["pred_point"] == [50, 40]
