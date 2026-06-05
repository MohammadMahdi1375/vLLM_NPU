#!/bin/bash
set -eo pipefail
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
set -u
cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# ============ Configuration ============
MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
OUTPUT_DIR="./output/dflash_online_colocated_10k"
VLLM_PORT=8000
MAX_SAMPLES=10000
SEQ_LENGTH=2048
EPOCHS=5
LR=3e-4

# DFlash params
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=8192
TARGET_LAYER_IDS="2 18 33"

# Co-location: server AND trainer share ALL 8 NPUs
ALL_NPUS="0,1,2,3,4,5,6,7"
VLLM_DP=8
NUM_TRAIN_NPUS=8
VLLM_GMU=0.4                  # server memory cap; leaves room for the trainer on each card
# =======================================

# Environment
export HCCL_CONNECT_TIMEOUT=1800
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

# Step 1: prepare data
echo "=== Step 1: prepare_data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --overwrite

# Step 2: launch vLLM hidden-states server on ALL 8 NPUs (memory capped)
echo "=== Step 2: launch vLLM server (8 NPUs, gmu=$VLLM_GMU) ==="
ASCEND_RT_VISIBLE_DEVICES="$ALL_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --max-model-len 4096 \
       --gpu-memory-utilization "$VLLM_GMU" &
VLLM_PID=$!
cleanup() { echo "Stopping vLLM..."; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Waiting for server..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 2; done
echo "Server ready."

# Step 3: train online on ALL 8 NPUs; hidden states generated then DELETED (zero storage)
echo "=== Step 3: train (online, co-located on 8 NPUs) ==="
ASCEND_RT_VISIBLE_DEVICES="$ALL_NPUS" torchrun \
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
    --request-timeout 180 \
    --max-retries 8 \
    --log-freq 10

echo "Done. Checkpoints in $OUTPUT_DIR/checkpoints/"