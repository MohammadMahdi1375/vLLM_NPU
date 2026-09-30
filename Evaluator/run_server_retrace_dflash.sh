#!/usr/bin/env bash
# Standalone Qwen3-4B + ReTrace/DFlash-b16 server. No evaluator dependency.
# Start: DEVICES=0 bash run_server_retrace_dflash.sh
# Choose another trained checkpoint: DRAFT=/absolute/checkpoints/7 DEVICES=0 bash ...
# Evaluation requests must set temperature=0; evaluator_retrace_t0.py does this.
################################################################################
# cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/Evaluator
# DEVICES=0 bash run_server_retrace_dflash.sh
################################################################################
set -eo pipefail

ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
TARGET="${TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
# This is the completed DFlash+ReTrace checkpoint you reported. Override DRAFT
# to evaluate a newer paper-recipe checkpoint; do not use the original DFlash
# checkpoint here, because it does not contain the ReTrace checkpoint identity.
DRAFT="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/retrace_dflash_qwen3_4b_8epoch_fullvocab/checkpoints/7"
PORT="${PORT:-8201}"
HOST="${HOST:-127.0.0.1}"
DEVICES="${DEVICES:-0}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-4b-retrace-dflash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [[ "${TP_SIZE:-1}" != 1 || ! "$DEVICES" =~ ^[0-9]+$ ]]; then
  echo 'Use one logical NPU: DEVICES=0 TP_SIZE=1 bash run_server_retrace_dflash.sh' >&2
  exit 1
fi
for checkpoint in "$TARGET" "$DRAFT"; do
  test -f "$checkpoint/config.json" || { echo "Missing $checkpoint/config.json" >&2; exit 1; }
done
test -x "$PYTHON_BIN" || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
# Greedy proposals at T=0 do not require the opt-in T>0 proposal sampling hook.
export RETRACE_DFLASH_MATCHED_SAMPLING=0
# Enable /reset_prefix_cache for the paper's post-warmup cache reset on this
# local benchmark server. For a network-facing server, leave dev mode disabled.
if [[ "$HOST" == 127.0.0.1 || "$HOST" == localhost || "$HOST" == ::1 ]]; then
  export VLLM_SERVER_DEV_MODE=1
else
  export VLLM_SERVER_DEV_MODE=0
fi

SPEC_CONFIG=$("$PYTHON_BIN" - "$TARGET" "$DRAFT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
target = json.loads((Path(sys.argv[1]) / 'config.json').read_text())
path = Path(sys.argv[2]).resolve()
raw = (path / 'config.json').read_bytes()
c = json.loads(raw)
if c.get('block_size') != 16:
    raise SystemExit('Expected DFlash block_size=16: one anchor + 15 draft proposals.')
if 'ReTraceDraftModel' not in c.get('architectures', []):
    raise SystemExit('DRAFT must be a trained DFlash-based ReTraceDraftModel checkpoint.')
if c.get('retrace_enabled') is not True:
    raise SystemExit('The checkpoint has ReTrace conditioning disabled.')
if c.get('sample_from_anchor', False):
    raise SystemExit('Expected a verified anchor, not sample_from_anchor=True.')
if float(c.get('retrace_beta_max', 1.0)) != 1.0:
    raise SystemExit('Expected retrace_beta_max=1.0 for the paper recipe.')
if c.get('draft_vocab_size') != target.get('vocab_size'):
    raise SystemExit('The native ReTrace proposer requires the full target vocabulary.')
print('Checkpoint config SHA256: ' + hashlib.sha256(raw).hexdigest(), file=sys.stderr)
print('ReTrace enabled; block=16; proposals=15; target vocabulary=' + str(target['vocab_size']), file=sys.stderr)
# In the installed native integration, the ReTraceDraftModel architecture
# selects ReTrace while the speculative method remains "dflash".
print(json.dumps({
    'method': 'dflash', 'model': str(path), 'num_speculative_tokens': 15,
    'draft_sample_method': 'greedy', 'disable_padded_drafter_batch': False,
    'enable_adaptive_verification': False, 'enforce_eager': True,
}))
PY
)

echo "Model: $SERVED_MODEL | logical NPU: $DEVICES | port: $PORT"
echo "Target: $TARGET"
echo "Draft: $DRAFT"
echo 'T0 evaluation: block 16, 15 proposals, TP=1, one request at a time.'
# Avoid importing the outer vllm checkout as a namespace package.
cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$TARGET" \
  --host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 \
  --seed 42 --generation-config vllm --max-num-seqs 1 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --block-size 128 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --no-enable-chunked-prefill --api-server-count 1 --renderer-num-workers 1 \
  --additional-config '{"enable_reduce_sample": false}' \
  --enable-per-request-metrics --per-request-spec-decode-metrics summary \
  --speculative-config "$SPEC_CONFIG" "$@"
