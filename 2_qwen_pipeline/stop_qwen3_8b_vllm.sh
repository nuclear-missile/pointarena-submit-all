#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="$ROOT_DIR/clean_3types_local/pids/qwen3_8b_vllm.pid"

if [[ ! -f "$PID_PATH" ]]; then
  echo "[INFO] no pid file: $PID_PATH"
  exit 0
fi

PID="$(cat "$PID_PATH" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  echo "[INFO] empty pid file"
  rm -f "$PID_PATH"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "[OK] stopped pid=$PID"
else
  echo "[INFO] pid not running: $PID"
fi

rm -f "$PID_PATH"
