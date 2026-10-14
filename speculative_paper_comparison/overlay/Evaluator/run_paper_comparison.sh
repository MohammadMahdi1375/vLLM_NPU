#!/usr/bin/env bash
# Validate all three implementations, then benchmark on one idle NPU in rotation.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-/home/n84449292/m84379596/conda/vllm-ascend-0202}}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DFLASH_DRAFT="${DFLASH_DRAFT:-$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4}"
PREFIX_DRAFT="${PREFIX_DRAFT:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"
# Required: point to your DSpark checkpoint with the intended training budget.
: "${DSPARK_DRAFT:?Set DSPARK_DRAFT to your trained DSpark checkpoint directory}"
MANIFEST="${MANIFEST:-$SPEC_MAIN/Evaluator/retrace_eval_prompts_all_v2.json}"
RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/paper_comparison_$(date +%Y%m%d_%H%M%S)_$$}"
DATASETS="${DATASETS:-gsm8k}"
read -r -a DATASET_LIST <<< "$DATASETS"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ "${CHECK_ONLY:-0}" == 0 || "${CHECK_ONLY:-0}" == 1 ]] || { echo "CHECK_ONLY must be 0 or 1" >&2; exit 2; }
EXTRA=()
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then EXTRA+=(--check-only); fi
if [[ -n "${VALIDATION_DIR:-}" ]]; then EXTRA+=(--validation-dir "$VALIDATION_DIR"); fi
if [[ -n "${TRAINING_RECORDS:-}" ]]; then EXTRA+=(--training-records "$TRAINING_RECORDS"); fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/compare_speculative.py" \
    --spec-main "$SPEC_MAIN" --target "$TARGET_MODEL" \
    --dflash "$DFLASH_DRAFT" --prefix-dflash "$PREFIX_DRAFT" --dspark "$DSPARK_DRAFT" \
    --manifest "$MANIFEST" --output "$RUN_DIR" --datasets "${DATASET_LIST[@]}" \
    --backbone-policy "${BACKBONE_POLICY:-matched_eager}" \
    --phase "${PHASE:-all}" --repeats "${REPEATS:-3}" \
    --num-prompts "${NUM_PROMPTS:-128}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-15}" \
    --npu "${NPU:-6}" --port "${PORT:-8213}" --seed "${SEED:-42}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" --max-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
    --memory-utilization "${MEMORY_UTILIZATION:-0.96}" --kv-block-size "${KV_BLOCK_SIZE:-128}" \
    "${EXTRA[@]}" "$@"
