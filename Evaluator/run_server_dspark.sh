#!/usr/bin/env bash
# Qwen3-4B + dspark; standalone vLLM server.
# Usage: DEVICES=0 bash run_server_dspark.sh
set -eo pipefail

ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"

TARGET="${TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dspark_qwen3_4b_5epoch_fullvocab_markov/checkpoints/0}"
PORT="${PORT:-8111}"
HOST="${HOST:-127.0.0.1}"
DEVICES="${DEVICES:-4}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-4b-dspark}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# The current ReTrace native implementation uses TP=1 and the V1 runner.
# Use the same settings for the two official checkpoints for comparison.
if [[ "${TP_SIZE:-1}" != 1 || ! "$DEVICES" =~ ^[0-9]+$ ]]; then
  echo "Use one logical NPU: DEVICES=0 TP_SIZE=1 bash run_server_dspark.sh" >&2
  exit 1
fi
for checkpoint in "$TARGET" "$DRAFT"; do
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing checkpoint config: $checkpoint/config.json" >&2
    exit 1
  fi
done
test -x "$PYTHON_BIN" || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"

export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Select the Conda C++ runtime needed by this environment's SQLite/ICU.
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

# JSON encoding handles checkpoint paths without shell quoting assumptions.
SPEC_CONFIG=$("$PYTHON_BIN" - "$DRAFT" <<'PY'
import json
import sys
print(json.dumps({
    "method": "dspark",
    "model": sys.argv[1],
    "num_speculative_tokens": 8,
    "draft_sample_method": "greedy",
    "disable_padded_drafter_batch": False,
    "enable_adaptive_verification": False,
    "enforce_eager": True,
}))
PY
)

echo "Model: $SERVED_MODEL | NPU: $DEVICES | port: $PORT"
echo "Target: $TARGET"
echo "Draft: $DRAFT"

# Avoid importing the outer vllm checkout as a namespace package.
cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$TARGET" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --dtype bfloat16 \
  --seed 42 \
  --generation-config vllm \
  --max-num-seqs 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --block-size 128 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --api-server-count 1 \
  --renderer-num-workers 1 \
  --additional-config '{"enable_reduce_sample": false}' \
  --enable-per-request-metrics \
  --per-request-spec-decode-metrics summary \
  --speculative-config "$SPEC_CONFIG" \
  "$@"
