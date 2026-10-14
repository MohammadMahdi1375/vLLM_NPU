#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Qwen3-4B + DSpark on Ascend (checkpoint 0)
# Baseline serving setup matched to run_server_dflash.sh.
# ============================================================

ENV_ROOT="/home/n84449292/m84379596/conda/vllm-ascend-0202"
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CANN_ROOT="/home/n84449292/m84379596/CANN/9.1.0"
PYTHON_BIN="$ENV_ROOT/bin/python"

# Target/verifier: original Qwen3-4B, never the draft checkpoint.
TARGET_MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

# Trained DSpark drafter: block_size=16; Markov + confidence heads.
DRAFT_MODEL="$SPEC_MAIN/output/dspark_qwen3_4b_openperfectblend_bs16/dspark_qwen3_4b_openperfectblend/checkpoints/0"

# Ascend fused attention currently supports <=16 verification tokens,
# including the target bonus token: K + 1 <= 16.
NUM_SPECULATIVE_TOKENS=15

# Simple sanity checks for accidentally wrong paths.
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$TARGET_MODEL/config.json" ]] || { echo "Missing target config: $TARGET_MODEL/config.json" >&2; exit 1; }
[[ -f "$DRAFT_MODEL/config.json" ]] || { echo "Missing DSpark config: $DRAFT_MODEL/config.json" >&2; exit 1; }

# ============================================================
# Environment (same as working DFlash server)
# ============================================================

unset PYTHONPATH
unset ASCEND_LAUNCH_BLOCKING

# No DFlash-Prefix selector or NPU-graph special handling for DSpark.
unset DFLASH_PREFIX_INFERENCE DFLASH_PREFIX_NPU_GRAPH
unset DFLASH_PREFIX_TOKEN_CACHE DFLASH_PREFIX_PROFILE_STEPS
unset DFLASH_PREFIX_SCORE_CHECK DFLASH_PREFIX_VALIDATE_STEPS

source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"

export SOC_VERSION=ascend910_9372
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi

export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=false

export HCCL_CONNECT_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# ============================================================
# Start DSpark server
# ============================================================

cd "$SPEC_MAIN/speculators"

exec "$PYTHON_BIN" \
    -m vllm.entrypoints.cli.main \
    serve "$TARGET_MODEL" \
    --host 127.0.0.1 \
    --port 8212 \
    --served-model-name qwen3-4b-dspark \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --seed 42 \
    --generation-config vllm \
    --max-num-seqs 1 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.96 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --api-server-count 1 \
    --renderer-num-workers 1 \
    --additional-config '{"enable_reduce_sample": false}' \
    --enable-per-request-metrics \
    --per-request-spec-decode-metrics summary \
    --speculative-config "{
        \"method\": \"dspark\",
        \"model\": \"$DRAFT_MODEL\",
        \"num_speculative_tokens\": $NUM_SPECULATIVE_TOKENS,
        \"draft_sample_method\": \"greedy\",
        \"enable_adaptive_verification\": false,
        \"enforce_eager\": true
    }"
