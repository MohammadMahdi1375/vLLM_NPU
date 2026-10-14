#!/usr/bin/env bash
# Qwen3 + DFlash, DFlash-Prefix, or DSpark on this project's Ascend v1 runner.
# Self-contained launcher: settings and the vLLM command are visible here.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"

# ============================================================
# Edit these settings, or override them on the command line.
# Both vanilla DFlash and DFlash-Prefix use method=dflash.
# DSpark uses method=dspark and a DSpark checkpoint.
# ============================================================
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-/home/n84449292/m84379596/conda/vllm-ascend-0202}}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"

SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"  # dflash | dspark
DRAFT_MODEL="${DRAFT_MODEL:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"  # 7 for your block-8 DFlash; 15 for block-16
# Vanilla example:
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4"
# DSpark example (also set SPECULATIVE_METHOD=dspark):
# DRAFT_MODEL="$SPEC_MAIN/output/dspark_qwen3_4b_openperfectblend_bs16/dspark_qwen3_4b_openperfectblend/checkpoints/0"

NPU="${NPU:-6}"
PORT="${PORT:-8213}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-$SPECULATIVE_METHOD}"
TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"  # graph | eager
MODE="${MODE:-fast}"                         # prefix: fast | validate | reference
CHECK_ONLY="${CHECK_ONLY:-0}"                 # 1: print command; no CANN or NPU use
SERVER_LOG="${SERVER_LOG:-$SPEC_MAIN/output/serve_${SPECULATIVE_METHOD}_${PORT}_$(date +%Y%m%d_%H%M%S)_$$.log}"

# ============================================================
# Check the method/checkpoint match without importing Torch.
# Detect which prefix selector was trained; never change weights/config.
# ============================================================
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || { echo "CHECK_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$MODE" == fast || "$MODE" == validate || "$MODE" == reference ]] || { echo "Invalid MODE: $MODE" >&2; exit 1; }
metadata="$("$PYTHON_BIN" - "$TARGET_MODEL" "$DRAFT_MODEL" "$SPECULATIVE_METHOD" "$NUM_SPECULATIVE_TOKENS" <<'PY_CONFIG'
import json, sys
from pathlib import Path
target, draft = (Path(p).expanduser().resolve(strict=True) for p in sys.argv[1:3])
method, proposals = sys.argv[3], int(sys.argv[4])
if str(proposals) != sys.argv[4]:
    raise SystemExit("NUM_SPECULATIVE_TOKENS must be a decimal integer without leading zeros")
t, d = (json.loads((p / "config.json").read_text()) for p in (target, draft))
if target == draft or "speculators_model_type" in t:
    raise SystemExit("TARGET_MODEL must be the original target, never the drafter")
if t.get("text_config", t).get("model_type") != "qwen3":
    raise SystemExit("This launcher is for your Qwen3 text target")
family, block = d.get("speculators_model_type"), d.get("block_size")
expected = {"dflash": "dflash", "dflash_prefix": "dflash", "dspark": "dspark"}.get(family)
if expected is None or method != expected:
    raise SystemExit(f"Checkpoint family={family!r} requires method={expected!r}, got {method!r}")
if type(block) is not int or not 1 <= proposals <= 15:
    raise SystemExit("Use a valid checkpoint block_size and 1..15 proposals for this fixed-batch launcher")
if family != "dspark" and proposals != block - 1:
    raise SystemExit(f"DFlash/Prefix block_size={block} requires {block - 1} proposals")
if family == "dspark" and proposals > block - (not d.get("sample_from_anchor", False)):
    raise SystemExit("Requested DSpark proposals exceed the trained positions")
kind = d.get("prefix_selector_kind") if family == "dflash_prefix" else family
if family == "dflash_prefix":
    if kind not in ("local_prefix_v2", "compiled_prefix_v3") or d.get("prefix_top_k") != 16:
        raise SystemExit("Expected your top-16 local_prefix_v2 or compiled_prefix_v3 checkpoint")
    if d.get("prefix_disable_selector", False) or d.get("prefix_inference_gate_scale", 1) != 1:
        raise SystemExit("Use a checkpoint with its selector enabled at the trained scale")
    if d.get("prefix_walk_backend", "torch") != "torch":
        raise SystemExit("The attention prefix selector requires prefix_walk_backend=torch")
print(kind)
print(json.dumps(str(draft)))  # JSON-escaped path, including its quotes
PY_CONFIG
)"
KIND="${metadata%%$'\n'*}"
DRAFT_JSON="${metadata#*$'\n'}"
VERIFY_TOKENS=$((NUM_SPECULATIVE_TOKENS + 1))

# ============================================================
# Environment
# ============================================================
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
for variable in ${!DFLASH_PREFIX_@}; do unset "$variable"; done
if [[ "$CHECK_ONLY" == 0 ]]; then
    source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
    source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
fi
# Keep CANN's A2 platform selection; do not hard-code an A3 SOC_VERSION.
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=0 VLLM_NVTX_SCOPES_FOR_PROFILING=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"

# Prefix-specific acceleration. Vanilla and DSpark do not use these selectors.
export DFLASH_PREFIX_INFERENCE=reference DFLASH_PREFIX_NPU_GRAPH=0
if [[ "$KIND" == local_prefix_v2 || "$KIND" == compiled_prefix_v3 ]]; then
    export DFLASH_PREFIX_INFERENCE="$MODE"
    export DFLASH_PREFIX_TOKEN_CACHE=0 DFLASH_PREFIX_SCORE_CHECK=strict DFLASH_PREFIX_VALIDATE_STEPS=32
    export DFLASH_PREFIX_PROFILE_STEPS=0 DFLASH_PREFIX_SELECTOR_PROFILE_STEPS=0 DFLASH_PREFIX_FUSION_PROFILE_STEPS=0
    export DFLASH_PREFIX_HISTORY_GROUP_SIZE=1 DFLASH_PREFIX_FUSION=off
    if [[ "$MODE" != reference ]]; then
        export DFLASH_PREFIX_NPU_GRAPH=1 DFLASH_PREFIX_GRAPH_SCOPE=selector
        if [[ "$KIND" == local_prefix_v2 ]]; then export DFLASH_PREFIX_FUSION=local_unpaired; fi
    fi
fi

# ============================================================
# Target graph: same settings as the measured DFlash fast path.
# "enforce_eager": true INSIDE speculative-config applies to the drafter.
# A top-level --enforce-eager disables the TARGET graph.
# DSpark target-graph performance/parity must be measured separately.
# ============================================================
TARGET_FLAGS=()
case "$TARGET_EXECUTION" in
    graph)
        COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[$VERIFY_TOKENS],\"max_cudagraph_capture_size\":$VERIFY_TOKENS}"
        ;;
    eager)
        COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"NONE","custom_ops":["all"]}'
        TARGET_FLAGS+=(--enforce-eager)
        ;;
    *) echo "TARGET_EXECUTION must be graph or eager" >&2; exit 1 ;;
esac

cd "$SPEC_MAIN/speculators"
COMMAND=(
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$TARGET_MODEL"
    --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 --seed 42
    --generation-config vllm --max-num-seqs 1
    --max-model-len 32768 --max-num-batched-tokens 32768
    --block-size 128 --gpu-memory-utilization 0.96
    --no-async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill
    --api-server-count 1 --renderer-num-workers 1
    --compilation-config "$COMPILATION_CONFIG"
    --additional-config '{"enable_reduce_sample":false,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false,"enable_super_kernel":false,"fuse_norm_quant":false,"fuse_qknorm_rope":false,"fuse_muls_add":false}}'
    --enable-per-request-metrics --per-request-spec-decode-metrics summary
    --speculative-config "{
        \"method\": \"$SPECULATIVE_METHOD\",
        \"model\": $DRAFT_JSON,
        \"num_speculative_tokens\": $NUM_SPECULATIVE_TOKENS,
        \"draft_sample_method\": \"greedy\",
        \"disable_padded_drafter_batch\": false,
        \"enable_adaptive_verification\": false,
        \"enforce_eager\": true
    }"
    "${TARGET_FLAGS[@]}"
)

if [[ "$CHECK_ONLY" == 0 ]]; then
    mkdir -p -- "$(dirname -- "$SERVER_LOG")"
    exec > >(tee "$SERVER_LOG") 2>&1
fi
printf 'Method: %s; checkpoint kind: %s; proposals: %s; target: %s; prefix mode: %s\n' \
    "$SPECULATIVE_METHOD" "$KIND" "$NUM_SPECULATIVE_TOKENS" "$TARGET_EXECUTION" "$DFLASH_PREFIX_INFERENCE"
printf 'Target: %s\nDraft: %s\nServer log: %s\n' "$TARGET_MODEL" "$DRAFT_MODEL" "$SERVER_LOG"
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Configuration checked. No server started."
    exit 0
fi
# Evaluator warmup still works. This simple launcher does not register launch.json.
# For prefix fast mode, confirm the parity-pass/steady-state marker in this log.
exec "${COMMAND[@]}"
