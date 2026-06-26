#!/usr/bin/env bash
# Build 20k Counting data — local GPU 7
# Continues from pixmo_points rank 8192 (forward frontier after counting_continuation)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT_DIR/clean_counting_local"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-8B}"

OUTPUT_ROOT="counting_20k_forward"
ENDPOINT="http://127.0.0.1:8018"
MODEL="Qwen3-8B"
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Include ALL existing counting data roots to track cumulative count
COUNT_ROOTS=(
  "$WORK_DIR/../pointarena_extract/mix4_counting_to_10000_continuation"
  "$WORK_DIR/../pointarena_extract/mix4_4000_clean_multipoint_counting_same_session_rulefilter"
  "$WORK_DIR/../pointarena_extract/pointarena_full_clean_multipoint_counting_same_session"
)
COUNT_ROOTS_ARG=""
for r in "${COUNT_ROOTS[@]}"; do
  COUNT_ROOTS_ARG="$COUNT_ROOTS_ARG,$r"
done
COUNT_ROOTS_ARG="${COUNT_ROOTS_ARG#,}"

TARGET_TOTAL=20000
WORKERS=24
BATCH_MAX_NEW_JOBS=2000

echo "============================================"
echo "Building 20k Counting Data"
echo "Output: $OUTPUT_ROOT"
echo "Target: $TARGET_TOTAL Counting samples"
echo "GPU: $GPU_ID (for vLLM server)"
echo "============================================"

cd "$WORK_DIR"

python3 run_mix4_streaming_local.py \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --output-root "$OUTPUT_ROOT" \
    --resume-from-output-root "$WORK_DIR/../pointarena_extract/mix4_counting_to_10000_continuation" \
    --count-roots "$COUNT_ROOTS_ARG" \
    --scan-direction forward \
    --exclude-image-keys-json "$WORK_DIR/exclusion_image_keys.json" \
    --target-category Counting \
    --target-total "$TARGET_TOTAL" \
    --max-new-jobs "$BATCH_MAX_NEW_JOBS" \
    --workers "$WORKERS" \
    --timeout 120 \
    --sample-retries 4 \
    --overwrite

echo "[DONE] Counting data construction complete"
