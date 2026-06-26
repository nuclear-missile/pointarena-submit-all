from __future__ import annotations

import json
from pathlib import Path


def test_resource_probe_exists_and_fields():
    p = Path("reports/resource_probe.json")
    assert p.exists(), "run scripts/00_probe_resources.sh first"
    obj = json.loads(p.read_text(encoding="utf-8"))

    assert obj.get("resource_dir")
    assert "molmo_models" in obj
    models = obj["molmo_models"]
    assert any(k.startswith("molmo2") for k in models.keys())

    pe = obj.get("pointarena_eval", {})
    assert pe.get("script")
    assert pe.get("dataset_path")

    # ensure detected paths remain inside resource dir
    resource = Path(obj["resource_dir"]).resolve()
    script = Path(pe["script"]).resolve()
    assert script.is_relative_to(resource)
