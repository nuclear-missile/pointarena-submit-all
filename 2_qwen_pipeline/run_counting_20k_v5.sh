#!/usr/bin/env bash
# Build 20k Counting data using clean_3types_local pipeline with --counting-enabled
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT_DIR/clean_3types_local"   # Use ORIGINAL clean_3types_local
OUTPUT_DIR="$ROOT_DIR/clean_counting_local/counting_20k_forward"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3-8B}"

ENDPOINT="http://127.0.0.1:8018"
MODEL="Qwen3-8B"
TARGET_TOTAL=20000
WORKERS=24
BATCH=2000

COUNT_ROOTS="\
$ROOT_DIR/pointarena_extract/mix4_counting_to_10000_continuation,\
$ROOT_DIR/pointarena_extract/mix4_4000_clean_multipoint_counting_same_session_rulefilter,\
$ROOT_DIR/pointarena_extract/pointarena_full_clean_multipoint_counting_same_session"

RESUME_ROOT="$ROOT_DIR/pointarena_extract/mix4_counting_to_10000_continuation"
EXCLUDE_KEYS="$WORK_DIR/exclusion_image_keys.json"

echo "============================================"
echo "Building 20k Counting Data (v5)"
echo "Using clean_3types_local pipeline"
echo "Target: $TARGET_TOTAL Counting samples"
echo "Output: $OUTPUT_DIR"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

cd "$WORK_DIR"

python3 run_mix4_streaming_local.py \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --output-root "$OUTPUT_DIR" \
    --resume-from-output-root "$RESUME_ROOT" \
    --count-roots "$COUNT_ROOTS" \
    --scan-direction forward \
    --exclude-image-keys-json "$EXCLUDE_KEYS" \
    --target-category Counting \
    --target-total "$TARGET_TOTAL" \
    --max-new-jobs "$BATCH" \
    --workers "$WORKERS" \
    --timeout 120 \
    --sample-retries 4 \
    --counting-enabled \
    --overwrite

echo "[DONE] Counting data construction complete"
