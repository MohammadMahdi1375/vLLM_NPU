#!/bin/bash
# ============================================================================
# DFlash CO-LOCATED, 2-NODE — target TP=16 + draft DP=16 across 16 NPUs.
# ONE torchrun job spanning both nodes; every rank hosts a TP=16 target shard
# AND trains a draft replica (DP = world_size = 16). In-process, one HCCL group.
#
# RUN ON BOTH NODES:
#     parent 80.5.5.108:   bash dflash_colo_2node.sh 0
#     child  80.5.5.109:   bash dflash_colo_2node.sh 1
#
# Key facts for the 2-node co-located case:
#  * Hidden states stay on LOCAL /dev/shm — each rank reads only its own slice
#    on its own node (the TP gather/scatter writes per-rank files). No shared FS
#    needed for hidden states (unlike the separate workflow).
#  * The tokenized dataset DOES go on a shared FS so both nodes read identical
#    Multipack shards. prepare_data runs once (on node 0); node 1 waits for it.
#  * Inter-node HCCL must be up between the NPUs (RDMA fabric). torchrun handles
#    the TCP rendezvous via MASTER_ADDR/PORT; HCCL uses the device network.
#  * Requires the vendor in_the_same_node_as patch (env LOCAL_WORLD_SIZE based):
#    with LOCAL_WORLD_SIZE=8 it computes node = global_rank // 8, so it
#    generalizes to 2 nodes. This path was only tested single-node — watch the
#    engine-init logs (MessageQueue / _node_count) the first time.
# ============================================================================
set -eo pipefail

NODE_RANK="${1:?usage: bash dflash_colo_2node.sh <node_rank 0|1>   (0=parent, 1=child)}"

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

# ===================== CONFIG (identical on both nodes) =====================
PARENT_IP="80.5.5.108"        # = MASTER_ADDR (node 0)
CHILD_IP="80.5.5.109"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=8              # 8 NPUs per node
TARGET_TP_SIZE=16            # target sharded across all 16 cards
LOCAL_NPUS="0,1,2,3,4,5,6,7"  # the 8 local cards each node exposes

MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"   # swap to DeepSeek-V4-Flash for the real target
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
# Tokenized data on the SHARED FS (same absolute path on both nodes):
SHARED_OUT="/share/canada_group_folder/n84449292/dflash_colo_2node"
# Hidden states on LOCAL /dev/shm (per-node, fast):
SHARED_STORAGE_PATH="/dev/shm/hidden_states"

MAX_SAMPLES=10000
SEQ_LENGTH=3072
EPOCHS=1
LR=6e-4
SEED=42
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"
# VOCAB: full 151936 (matches SpecForge). Requires the model.py full-vocab patch;
# we OMIT --draft-vocab-size so the trainer uses the full verifier vocab.
# ============================================================================

export no_proxy="localhost,127.0.0.1,::1,${PARENT_IP},${CHILD_IP}"
export NO_PROXY="$no_proxy"
export DFLASH_TP_GATHER=1                      # REQUIRED for TP>1 co-location
export HCCL_CONNECT_TIMEOUT=1800
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
# If the cross-node rendezvous can't find the route, set these to the host NIC
# that carries the 80.5.5.0/24 network on each node (find via `ip -o addr`):
# export GLOO_SOCKET_IFNAME=eth0
# export HCCL_SOCKET_IFNAME=eth0

cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# clear stragglers + stale per-node RAM state
pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "EngineCore"       2>/dev/null || true
rm -rf "$SHARED_STORAGE_PATH"
sleep 2

# ---- data: node 0 tokenizes to the shared FS; node 1 waits for it ----
if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$SHARED_OUT"
    touch "$SHARED_OUT/.write_test" && rm -f "$SHARED_OUT/.write_test" \
      || { echo "ERROR: $SHARED_OUT not writable on node 0"; exit 1; }
    echo "=== [node 0] Step 1: prepare_data (-> shared FS) ==="
    python scripts/prepare_data.py \
        --model "$MODEL" --data "$DATASET" --output "$SHARED_OUT" \
        --max-samples "$MAX_SAMPLES" --seq-length "$SEQ_LENGTH" --overwrite
    touch "$SHARED_OUT/.prepare_done"
else
    echo "=== [node $NODE_RANK] waiting for node 0 to finish prepare_data on shared FS ==="
    until [ -f "$SHARED_OUT/.prepare_done" ]; do sleep 3; done
    echo "    data ready."
fi

echo "=== Step 2: train (in-process co-located, TP=$TARGET_TP_SIZE, DP=$((NNODES*NPROC_PER_NODE)), node_rank=$NODE_RANK) ==="
ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$PARENT_IP" \
    --master_port "$MASTER_PORT" \
    --nproc_per_node "$NPROC_PER_NODE" \
    scripts/train.py \
    --in-process-target \
    --target-tp-size "$TARGET_TP_SIZE" \
    --shared-storage-path "$SHARED_STORAGE_PATH" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$SHARED_OUT" \
    --save-path "$SHARED_OUT/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --mask-token-id 151669 \
    --noise-std 0.0 \
    --scheduler-type cosine \
    --logger tensorboard \
    --run-name dflash_colo_2node \
    --log-dir ./logs/colo_2node \
    --on-missing generate \
    --on-generate delete \
    --log-freq 10 \
    --seed "$SEED"

echo "[node $NODE_RANK] done. Checkpoints (shared): $SHARED_OUT/checkpoints/"
