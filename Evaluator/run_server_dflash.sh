#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Qwen3-4B + DFlash-Prefix on Ascend -- edit settings directly
# Uses the already-installed prefix inference / NPU graph patches.
# ============================================================

ENV_ROOT="/home/n84449292/m84379596/conda/vllm-dflash2-main"
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CANN_ROOT="/home/n84449292/m84379596/CANN/9.1.0"
PYTHON_BIN="$ENV_ROOT/bin/python"

# TARGET must be the original target model, not a draft checkpoint.
TARGET_MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

# Active draft: prefix block 8, matching the example you supplied.
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs8/tau_v2_scratch_20260928_082808_3142334/checkpoints/0"
# NUM_SPECULATIVE_TOKENS=7

# Prefix block 16: uncomment BOTH lines to use your measured block-16 run.
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0"
DRAFT_MODEL="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/0"
NUM_SPECULATIVE_TOKENS=15

# Vanilla alternatives: set DRAFT_MODEL and match the proposal count (7 or 15).
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs8/checkpoints/0"
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/0"
# For vanilla, also set DFLASH_PREFIX_INFERENCE=reference and DFLASH_PREFIX_NPU_GRAPH=0 below.

# ============================================================
# Environment
# ============================================================

unset PYTHONPATH
unset ASCEND_LAUNCH_BLOCKING

source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"

export SOC_VERSION=ascend910_9372
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi

export ASCEND_RT_VISIBLE_DEVICES=12
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
# Prefix execution: the working optimized path
# ============================================================

export DFLASH_PREFIX_INFERENCE=fast        # fast | validate | reference
export DFLASH_PREFIX_NPU_GRAPH=1           # 1: graph walk; 0: eager walk
export DFLASH_PREFIX_TOKEN_CACHE=0         # keep off: that cache changed token choices
export DFLASH_PREFIX_PROFILE_STEPS=0       # no profiler during speed measurements
export DFLASH_PREFIX_SCORE_CHECK=strict
export DFLASH_PREFIX_VALIDATE_STEPS=32     # initial checks, completed during warmup

# Global --enforce-eager below remains intentional: the selector has its own graph.
# Current selector graph supports a fixed batch of 1; keep --max-num-seqs 1.
# Warm up before timing and confirm the server prints:
# PREFIX_PARITY_PASS ... mode=fast npu_graph=True; steady-state fast path enabled

# ============================================================
# Start server -- all vLLM arguments are explicitly editable
# ============================================================

cd "$SPEC_MAIN/speculators"

exec "$PYTHON_BIN" \
    -m vllm.entrypoints.cli.main \
    serve "$TARGET_MODEL" \
    --host 127.0.0.1 \
    --port 8212 \
    --served-model-name qwen3-4b-dflash \
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
        \"method\": \"dflash\",
        \"model\": \"$DRAFT_MODEL\",
        \"num_speculative_tokens\": $NUM_SPECULATIVE_TOKENS,
        \"draft_sample_method\": \"greedy\",
        \"disable_padded_drafter_batch\": false,
        \"enable_adaptive_verification\": false,
        \"enforce_eager\": true
    }"
