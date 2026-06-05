#!/bin/bash
set -euo pipefail
cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# ============ Configuration ============
MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
OUTPUT_DIR="./output/dflash_qwen3_8b_perfectblend_10k_online"
VLLM_PORT=8000
MAX_SAMPLES=10000
SEQ_LENGTH=2048
EPOCHS=5
LR=3e-4

# DFlash params (unchanged from the official example)
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=8192
TARGET_LAYER_IDS="2 18 33"

# NPU split: server and trainer must be disjoint
VLLM_NPUS="0,1"               # 2 NPUs serve hidden states
TRAIN_NPUS="2,3,4,5,6,7"      # 6 NPUs train
NUM_TRAIN_NPUS=6
VLLM_DP=2                     # = number of VLLM_NPUS
# =======================================

# Step 1: prepare data
echo "=== Step 1: prepare_data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: launch vLLM hidden-states server (its own NPUs)
echo "=== Step 2: launch vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP" --port "$VLLM_PORT" &
VLLM_PID=$!
cleanup() { echo "Stopping vLLM..."; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Waiting for server..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 2; done
echo "Server ready."

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_LOGS=""
# Step 3: train against the LIVE server — hidden states generated on the fly
echo "=== Step 3: train (online) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate \
    --on-generate delete \
    --log-freq 10

echo "Done. Checkpoints in $OUTPUT_DIR/checkpoints/"