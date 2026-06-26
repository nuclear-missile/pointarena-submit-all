#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT_DIR/clean_3types_local"
LOG_DIR="$WORK_DIR/logs"
PID_DIR="$WORK_DIR/pids"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-8B}"
ENV_NAME="${CONDA_ENV_NAME:-base}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8018}"
MIN_FREE_MB="${MIN_FREE_MB:-10000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"

mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERR] missing model dir: $MODEL_DIR"
  exit 1
fi

GPU_ID="${GPU_ID:-}"
if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$(
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
      | awk -F',' -v min_free="$MIN_FREE_MB" '
          {
            idx=$1 + 0;
            used=$2 + 0;
            free=$3 + 0;
            if (free >= min_free) {
              if (best_idx == "" || used < best_used || (used == best_used && free > best_free)) {
                best_idx=idx;
                best_used=used;
                best_free=free;
              }
            }
          }
          END {
            if (best_idx == "") {
              exit 1;
            }
            print best_idx;
          }
      '
  )"
fi

if [[ -z "$GPU_ID" ]]; then
  echo "[ERR] no GPU has free memory >= ${MIN_FREE_MB} MB"
  exit 1
fi

LOG_PATH="$LOG_DIR/qwen3_8b_vllm.log"
PID_PATH="$PID_DIR/qwen3_8b_vllm.pid"

if [[ -f "$PID_PATH" ]]; then
  OLD_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[INFO] server already running with pid=$OLD_PID"
    echo "[INFO] host=$VLLM_HOST port=$VLLM_PORT gpu=$GPU_ID"
    exit 0
  fi
fi

set +u
source /mnt/data/lv_qi/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
set -u

echo "[INFO] launching model=$MODEL_DIR on gpu=$GPU_ID host=$VLLM_HOST port=$VLLM_PORT"
CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  >"$LOG_PATH" 2>&1 &

PID="$!"
echo "$PID" > "$PID_PATH"
echo "[OK] started pid=$PID"
echo "[OK] log=$LOG_PATH"
