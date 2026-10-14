#!/usr/bin/env bash
# DSpark defaults for the shared configurable fast launcher.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dspark}"
export DRAFT_MODEL="${DRAFT_MODEL:-${DRAFT:-$SPEC_MAIN/output/dspark_qwen3_4b_openperfectblend_bs16/dspark_qwen3_4b_openperfectblend/checkpoints/0}}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-dspark}"
export NPU="${NPU:-6}"
export PORT="${PORT:-8213}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"
export DRAFT_ENFORCE_EAGER="${DRAFT_ENFORCE_EAGER:-omit}"
# Environment paths and all other settings are editable in run_server_dflash.sh
# or can be supplied here as environment overrides.
# The known DSpark path is epoch 1. For epoch-matched comparisons, explicitly
# set DRAFT_MODEL to your corresponding trained DSpark checkpoint.
exec bash "$SCRIPT_DIR/run_server_dflash.sh"
