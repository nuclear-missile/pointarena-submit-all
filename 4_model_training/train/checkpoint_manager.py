from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def rank_key(item: dict[str, Any]) -> tuple:
    subtasks = item.get("subtasks", {}) or {}
    overall = float(item.get("overall", 0.0))
    reasoning = float(subtasks.get("reasoning", 0.0))
    steerability = float(subtasks.get("steerability", 0.0))
    step = int(item.get("step", 10**9))
    return (-overall, -reasoning, -steerability, step)


def _load_board(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_board(path: Path, board: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_large_files(ckpt_dir: Path) -> None:
    patterns = [
        "*.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ]
    for pat in patterns:
        for p in ckpt_dir.glob(pat):
            if p.name.startswith("metrics_step_") or p.name.startswith("pointarena_report_step_"):
                continue
            try:
                p.unlink()
            except Exception:
                pass


def update_leaderboard(output_dir: str | Path, result: dict[str, Any], keep_top_m: int = 3) -> list[dict[str, Any]]:
    out = Path(output_dir)
    board_path = out / "leaderboard.json"
    board = _load_board(board_path)

    checkpoint_dir = str(result.get("checkpoint_dir", ""))
    board = [x for x in board if str(x.get("checkpoint_dir", "")) != checkpoint_dir]
    board.append(result)
    board.sort(key=rank_key)

    kept = board[:keep_top_m]
    dropped = board[keep_top_m:]

    # delete only heavy weights for dropped checkpoints, keep reports
    for row in dropped:
        ckpt = Path(str(row.get("checkpoint_dir", "")))
        if ckpt.exists() and ckpt.is_dir():
            _delete_large_files(ckpt)

    _save_board(board_path, kept)
    return kept
