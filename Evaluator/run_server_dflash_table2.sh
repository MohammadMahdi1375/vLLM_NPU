#!/usr/bin/env bash
# Independent server: no evaluator.py dependency. Use one idle NPU.
set -eo pipefail
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
TARGET="${TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:-/home/n84449292/m84379596/Huggingface/Qwen3-4B-DFlash-b16}"
DEVICES="${DEVICES:-0}"
PORT="${PORT:-8200}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-4b-dflash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
if [[ ! "$DEVICES" =~ ^[0-9]+$ || "${TP_SIZE:-1}" != 1 ]]; then
  echo 'This comparison uses one logical NPU, TP=1 and batch=1.' >&2
  exit 1
fi
for checkpoint in "$TARGET" "$DRAFT"; do
  test -f "$checkpoint/config.json" || { echo "Missing $checkpoint/config.json" >&2; exit 1; }
done
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export RETRACE_DFLASH_MATCHED_SAMPLING=1
cd "$SPEC_MAIN/speculators"
SPEC_CONFIG=$("$PYTHON_BIN" - "$DRAFT" <<'PY'
import json, sys, hashlib
from pathlib import Path
draft = Path(sys.argv[1])
c = json.loads((draft / 'config.json').read_text())
if c.get('block_size') != 16:
    raise SystemExit('Expected DFlash block_size=16 (anchor + 15 proposals)')
print(json.dumps({'method': 'dflash', 'model': str(draft),
    'num_speculative_tokens': 15, 'draft_sample_method': 'probabilistic',
    'disable_padded_drafter_batch': False, 'enable_adaptive_verification': False,
    'enforce_eager': True}))
print('Checkpoint config SHA256: ' + hashlib.sha256((draft/'config.json').read_bytes()).hexdigest(), file=sys.stderr)
print('ReTrace enabled: ' + str(c.get('retrace_enabled', False)), file=sys.stderr)
PY
)
echo "Target=$TARGET"
echo "Draft=$DRAFT"
echo "NPU=$DEVICES port=$PORT; sampling=T0 greedy, T>0 temperature/top-k/top-p proposals with exact q export"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$TARGET" \
  --host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 \
  --seed 42 --generation-config vllm --max-num-seqs 1 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --block-size 128 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --no-enable-chunked-prefill --api-server-count 1 --renderer-num-workers 1 \
  --additional-config '{"enable_reduce_sample": false}' \
  --enable-per-request-metrics --per-request-spec-decode-metrics summary \
  --speculative-config "$SPEC_CONFIG" "$@"
