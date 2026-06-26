#!/usr/bin/env bash
# Environment setup for tests
# Usage: source env.sh
# This script is sourced by run_all_tests.sh

# Add parent directory (submit_all/) to PYTHONPATH for src.* imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# Optional: activate conda environment if available
# conda activate pointarena 2>/dev/null || true

echo "[env.sh] PYTHONPATH=${PYTHONPATH}"
echo "[env.sh] Running tests from: ${SCRIPT_DIR}"
