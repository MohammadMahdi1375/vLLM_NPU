#!/usr/bin/env bash
# One editable launcher for Qwen3 + DFlash, DFlash-Prefix, or DSpark.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"

# ============================================================
# Paths and method -- edit these or override with environment variables.
# DFlash AND DFlash-Prefix: dflash. DSpark: dspark.
# auto reads the method from the checkpoint; mismatches are rejected.
# ============================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-/home/n84449292/m84379596/conda/vllm-ascend-0202}}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"  # dflash | dspark | auto
DRAFT_MODEL="${DRAFT_MODEL:-${DRAFT:-$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4}}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"  # 7 for block-8 DFlash/Prefix; 15 for block-16
# Prefix:  DRAFT_MODEL="$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4"
# DSpark:  SPECULATIVE_METHOD=dspark
#          DRAFT_MODEL="$SPEC_MAIN/output/dspark_qwen3_4b_openperfectblend_bs16/dspark_qwen3_4b_openperfectblend/checkpoints/0"

# ============================================================
# Execution and server settings
# ============================================================
NPU="${NPU:-6}"
PORT="${PORT:-8213}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-$SPECULATIVE_METHOD}"
TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"  # graph | eager
DRAFT_ENFORCE_EAGER="${DRAFT_ENFORCE_EAGER:-omit}" # omit: no speculative enforce_eager field; 1: true; 0: false
MODE="${MODE:-fast}"                         # prefix selector only: fast | validate | reference
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.96}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-128}"         # KV-cache page, not draft block size
GRAPH_METRICS="${GRAPH_METRICS:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"                 # 1 prints configuration; no CANN/NPU/server
RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/evaluator_server_$(date +%Y%m%d_%H%M%S)_$$}"

# ============================================================
# Ascend environment
# ============================================================
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || { echo "CHECK_ONLY must be 0 or 1" >&2; exit 1; }
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
if [[ "$CHECK_ONLY" == 0 ]]; then
    source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
    source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
fi
# Use CANN's platform selection; do not hard-code an A3 SOC on A2.
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"; fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"

# All method/execution choices are passed explicitly. The helper checks config,
# writes the exact vLLM command + log + evaluator record, then starts vLLM.
# In graph mode there is no top-level --enforce-eager. omit also leaves the
# speculative field absent. DSpark's current Ascend drafter itself stays eager.
exec "$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/dflash_prefix_compiled/target_graph_server.py" \
    --spec-main "$SPEC_MAIN" --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --speculative-method "$SPECULATIVE_METHOD" --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \
    --target-execution "$TARGET_EXECUTION" --draft-enforce-eager "$DRAFT_ENFORCE_EAGER" \
    --mode "$MODE" --npu "$NPU" --port "$PORT" --served-model-name "$SERVED_MODEL_NAME" \
    --seed "$SEED" --kv-block-size "$KV_BLOCK_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" --max-batched-tokens "$MAX_BATCHED_TOKENS" \
    --memory-utilization "$MEMORY_UTILIZATION" --graph-metrics "$GRAPH_METRICS" \
    --run-dir "$RUN_DIR" --check-only "$CHECK_ONLY" --register-evaluator 1
