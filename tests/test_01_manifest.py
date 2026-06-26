from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_exists_and_has_local_files() -> None:
    proc = subprocess.run(["bash", "scripts/01_download_robopoint.sh"], cwd=ROOT, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    manifest = ROOT / "artifacts" / "robopoint_probe" / "MANIFEST.json"
    assert manifest.exists(), "MANIFEST.json not generated"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    assert isinstance(files, list)
    assert len(files) > 0

    has_json_like = any(str(f.get("path", "")).endswith((".json", ".jsonl", ".arrow", ".parquet")) for f in files)
    has_image = any(str(f.get("path", "")).lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for f in files)

    assert has_json_like, "no JSON/arrow/parquet-like data file found in manifest"
    assert has_image, "no image file found in manifest"

    for f in files[:200]:
        p = str(f.get("abs_path", ""))
        assert not p.startswith("http://") and not p.startswith("https://")
