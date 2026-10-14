#!/usr/bin/env bash
# DFlash-Prefix defaults for the same fast launcher used by vanilla and DSpark.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"
export DRAFT_MODEL="${DRAFT_MODEL:-${DRAFT:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-prefix-dflash}"
export NPU="${NPU:-6}"
export PORT="${PORT:-8213}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"
export DRAFT_ENFORCE_EAGER="${DRAFT_ENFORCE_EAGER:-omit}"
exec bash "$SCRIPT_DIR/run_server_dflash.sh"
