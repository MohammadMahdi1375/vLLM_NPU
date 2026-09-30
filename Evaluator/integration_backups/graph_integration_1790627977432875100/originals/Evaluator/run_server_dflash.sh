#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Qwen3-4B + DFlash on Ascend
# ============================================================

ENV_ROOT="/home/n84449292/m84379596/conda/vllm-dflash2-main"
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CANN_ROOT="/home/n84449292/m84379596/CANN/9.1.0"
PYTHON_BIN="$ENV_ROOT/bin/python"

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

export ASCEND_RT_VISIBLE_DEVICES=9

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
# Start server
# ============================================================

cd "$SPEC_MAIN/speculators"

exec "$PYTHON_BIN" \
    -m vllm.entrypoints.cli.main \
    serve \
    "/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_prefix_qwen3_4b_openperfectblend_bs8/tau_v2_scratch_20260928_082808_3142334/checkpoints/0" \
    --host 127.0.0.1 \
    --port 8209 \
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
    --speculative-config '{
        "method": "dflash",
        "model": "/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_qwen3_4b_openperfectblend_bs8/checkpoints/0",
        "num_speculative_tokens": 7,
        "draft_sample_method": "greedy",
        "disable_padded_drafter_batch": false,
        "enable_adaptive_verification": false,
        "enforce_eager": true
    }'


# /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/0
# /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0
