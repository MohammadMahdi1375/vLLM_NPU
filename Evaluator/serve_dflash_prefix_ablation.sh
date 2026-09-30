#!/usr/bin/env bash
set -eo pipefail

# Run ONE mode at a time on the same NPU and port:
#   bash serve_dflash_prefix_ablation.sh prefix_off
#   bash serve_dflash_prefix_ablation.sh prefix_on
#   bash serve_dflash_prefix_ablation.sh vanilla
# Stop the previous evaluation server before starting the next mode.
MODE="${1:-prefix_off}"
case "$MODE" in
    prefix_off|prefix_on|vanilla) ;;
    *) echo "Mode must be prefix_off, prefix_on, or vanilla" >&2; exit 2 ;;
esac

ENV_ROOT="/home/n84449292/m84379596/conda/vllm-dflash2-main"
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CANN_ROOT="/home/n84449292/m84379596/CANN/9.1.0"
PYTHON_BIN="$ENV_ROOT/bin/python"

# Epoch 1 is checkpoints/0 in this trainer.
VANILLA_DRAFT="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/0"
PREFIX_DRAFT="$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/checkpoints/0"

# Environment matches the evaluation launcher supplied in this conversation.
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

export ASCEND_RT_VISIBLE_DEVICES=10
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

# The adapter reads prefix_disable_selector from config.json. A fresh view
# reuses the exact weight files through symlinks; the source checkpoint is intact.
# Setting PREFIX_DISABLE_SELECTOR in the environment alone would have no effect
# in a direct vLLM launch. Both prefix modes explicitly use the torch backend.
RUN_DIR="$(mktemp -d "$SPEC_MAIN/output/prefix_ablation_${MODE}_XXXXXX")"
DRAFT_PATH="$("$PYTHON_BIN" - "$MODE" "$VANILLA_DRAFT" "$PREFIX_DRAFT" "$RUN_DIR" <<'PY_ABLATION'
import hashlib
import json
import os
from pathlib import Path
import sys

mode = sys.argv[1]
if mode not in {"vanilla", "prefix_on", "prefix_off"}:
    raise SystemExit("Invalid ablation mode")
source = Path(sys.argv[2] if mode == "vanilla" else sys.argv[3]).resolve()
run = Path(sys.argv[4]).resolve()
config_path = source / "config.json"
original_bytes = config_path.read_bytes()
cfg = json.loads(original_bytes)
expected_type = "dflash" if mode == "vanilla" else "dflash_prefix"
if cfg.get("speculators_model_type") != expected_type:
    raise SystemExit(f"Expected {expected_type} checkpoint: {source}")
if cfg.get("block_size") != 16 or cfg.get("sample_from_anchor", False):
    raise SystemExit("This comparison requires block_size=16, sample_from_anchor=False")
if cfg.get("draft_vocab_size") != 151936:
    raise SystemExit("This comparison requires the full Qwen3-4B vocabulary")
if not list(source.glob("*.safetensors")) and not (source / "model.safetensors.index.json").is_file():
    raise SystemExit(f"Checkpoint has no safetensors weights: {source}")

draft = source
if mode != "vanilla":
    if "DFlashPrefixDraftModel" not in cfg.get("architectures", []):
        raise SystemExit("Prefix checkpoint has the wrong architecture tag")
    cfg["prefix_disable_selector"] = mode == "prefix_off"
    cfg["prefix_walk_backend"] = "torch"
    draft = run / "draft_view"
    draft.mkdir(exist_ok=False)
    (draft / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for item in source.iterdir():
        if item.name != "config.json" and (item.is_file() or item.is_dir()):
            (draft / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())

manifest = {
    "mode": mode,
    "source_checkpoint": str(source),
    "source_config_sha256": hashlib.sha256(original_bytes).hexdigest(),
    "draft_path": str(draft),
    "selector_disabled": mode == "prefix_off" if mode != "vanilla" else None,
    "walk_backend": "torch" if mode != "vanilla" else None,
    "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "checkpoint_config": cfg,
}
(run / "ablation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(draft)
PY_ABLATION
)"
cp -- "$0" "$RUN_DIR/launcher.sh"
echo "Mode: $MODE; NPU: $ASCEND_RT_VISIBLE_DEVICES; port: 8101"
echo "Draft: $DRAFT_PATH"
echo "Run configuration: $RUN_DIR/ablation.json"

cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" \
    -m vllm.entrypoints.cli.main \
    serve \
    "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
    --host 127.0.0.1 \
    --port 8102 \
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
        \"model\": \"$DRAFT_PATH\",
        \"num_speculative_tokens\": 15,
        \"draft_sample_method\": \"greedy\",
        \"disable_padded_drafter_batch\": false,
        \"enable_adaptive_verification\": false,
        \"enforce_eager\": true
    }"
