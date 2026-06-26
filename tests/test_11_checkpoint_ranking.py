from __future__ import annotations

import json
from pathlib import Path

from src.train.checkpoint_manager import update_leaderboard


def test_checkpoint_ranking_and_topm(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    for step, overall, reason, steer in [
        (200, 60.0, 40.0, 30.0),
        (400, 61.0, 35.0, 30.0),
        (600, 61.0, 45.0, 10.0),
        (800, 50.0, 50.0, 50.0),
    ]:
        ckpt = out / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_model.safetensors").write_text("x", encoding="utf-8")
        (ckpt / f"metrics_step_{step}.json").write_text("{}", encoding="utf-8")
        (ckpt / f"pointarena_report_step_{step}.md").write_text("# report", encoding="utf-8")
        update_leaderboard(
            out,
            {
                "step": step,
                "overall": overall,
                "subtasks": {
                    "reasoning": reason,
                    "steerability": steer,
                },
                "checkpoint_dir": str(ckpt),
            },
            keep_top_m=2,
        )

    board = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert len(board) == 2

    # top should be step 600: same overall as 400 but better reasoning
    assert int(board[0]["step"]) == 600

    # dropped checkpoint should have weight file removed but report preserved
    dropped = out / "checkpoint-800"
    assert (dropped / "pointarena_report_step_800.md").exists()
