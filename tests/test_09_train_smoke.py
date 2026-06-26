from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image


def test_train_smoke_mock(tmp_path: Path):
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    img = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img)

    row = {
        "id": "x",
        "source": "toy",
        "task_type": "object_reference",
        "image_path": str(img),
        "query": "Point to center",
        "target": {
            "type": "point",
            "points": [[32, 32]],
            "point_format": "xy_abs_pixels",
            "image_size": [64, 64],
        },
        "meta": {"split": "train"},
    }
    train_jsonl.write_text("\n".join([json.dumps(row) for _ in range(8)]) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps({**row, "source": "pointarena_val", "meta": {"split": "val", "category": "spatial", "gt_mask_path": ""}}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "run"
    cmd = [
        "python",
        "-m",
        "src.train.run_lora_train",
        "--model_path",
        "/mnt/data/lv_qi/xing/pointarena/models/allenai/Molmo2-8B",
        "--train_jsonl",
        str(train_jsonl),
        "--val_jsonl",
        str(val_jsonl),
        "--output_dir",
        str(out_dir),
        "--context_len",
        "256",
        "--max_steps",
        "2",
        "--save_every_steps",
        "1",
        "--eval_every_steps",
        "1",
        "--mock_train",
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "checkpoint-1" / "train_state.json").exists()
    assert (out_dir / "leaderboard.json").exists()
