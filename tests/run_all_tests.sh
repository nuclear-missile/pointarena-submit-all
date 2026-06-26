#!/usr/bin/env bash
# Run all tests for PointArena submission
# Usage: cd tests/ && bash run_all_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./env.sh

pytest -q \
  test_00_detect_resources.py \
  test_01_schema.py \
  test_02_normalize_pixmo.py \
  test_03_normalize_robopoint.py \
  test_04_normalize_where2place.py \
  test_05_normalize_refspatial.py \
  test_06_build_unified_dataset.py \
  test_07_prompt_and_parse.py \
  test_08_context_length.py \
  test_09_train_smoke.py \
  test_10_eval_adapter.py \
  test_11_checkpoint_ranking.py
