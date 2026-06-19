#!/bin/bash
# ============================================================================
# DFlash SEPARATE (online) trainer for Qwen3-4B — vLLM hidden-states server +
# FSDP trainer on disjoint NPUs (server NPU 0, DP=1; trainer NPUs 1-7, DP=7).
#
# Adapted from the Qwen3-8B version. Intended differences vs that script:
#   - verifier model  : Qwen3-8B -> Qwen3-4B
#   - dataset         : 10k subset -> open_perfectblend_full.jsonl (10k samples)
#   - NPU split       : 4 server / 4 trainer -> 1 server / 7 trainer
#
# DRAFTER DESIGN (matches z-lab/Qwen3-4B-DFlash-b16/config.json):
# The DFlash trainer derives the WHOLE draft architecture from the *verifier*
# config in create_transformer_layer_config() (scripts/train.py); there are NO
# CLI flags for hidden_size / intermediate_size / heads / head_dim. Pointing the
# verifier at Qwen3-4B and keeping the depth/block flags below reproduces the
# published config exactly:
#     hidden_size=2560, intermediate_size=9728, num_attention_heads=32,
#     num_key_value_heads=8, head_dim=128 (explicit, not hidden/heads),
#     num_hidden_layers=5 (=NUM_LAYERS), num_target_layers=36 (verifier depth),
#     block_size=16, target_layer_ids=[1,9,17,25,33], hidden_act=silu,
#     mask_token_id=151669, vocab_size=151936 (full), rope_theta=1e6,
#     rms_norm_eps=1e-6, all layer_types=full_attention.
# => Do NOT add architecture flags; the drafter is correct via the verifier.
#
# NOTE: the curl health URL and --vllm-endpoint are PLAIN URLs (no markdown
# "[...](...)" wrapping, which would break curl and the trainer).
# ============================================================================
set -eo pipefail
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
unset DFLASH_TP_GATHER                          # online mode: gather/scatter MUST be OFF
set -u
cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# ============ Configuration ============
MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
DATASET="/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl"
OUTPUT_DIR="./output/dflash_separate_qwen3_4b"
VLLM_PORT=8000
MAX_SAMPLES=10000
SEQ_LENGTH=3072            # per-sample max length used by prepare_data

# ---- "Batch size" lever ----------------------------------------------------
# This trainer has NO --batch-size flag. It packs samples by a TOKEN BUDGET per
# rank (--total-seq-len) via MultipackDistributedBatchSamplerV2. So the number of
# sequences per rank per step = TOTAL_SEQ_LEN / avg_sample_len  (variable).
# To leave room for ~TARGET_BATCH samples per rank, budget = TARGET_BATCH * the
# per-sample cap. With short samples you'll fit MORE than TARGET_BATCH; this sets
# headroom, not a hard count. Must be >= SEQ_LENGTH or long samples get clipped.
TARGET_BATCH=4                                  # desired ~sequences per rank per step
TOTAL_SEQ_LEN=$(( SEQ_LENGTH * TARGET_BATCH ))  # = 12288 tokens/rank budget
EPOCHS=1
LR=6e-4
SEED=42

SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"

# VOCAB: Qwen3-4B verifier full vocab is 151936. The z-lab Qwen3-4B-DFlash-b16
# checkpoint uses the FULL vocab (no draft<->target mapping). The DFlash code
# raises if draft_vocab_size == verifier vocab WITH mappings ("mappings not
# needed"), so FULL vocab must be requested by OMITTING --draft-vocab-size.
# This block does that automatically: it passes --draft-vocab-size only when
# the value is < verifier vocab; at == it omits the flag so the trainer falls
# back to the full vocab (the published config's behaviour).
VERIFIER_VOCAB=151936
DRAFT_VOCAB_SIZE=151936
VOCAB_FLAG=""
if [ "$DRAFT_VOCAB_SIZE" -lt "$VERIFIER_VOCAB" ]; then
    VOCAB_FLAG="--draft-vocab-size $DRAFT_VOCAB_SIZE"
fi

VLLM_NPUS="0"               # 1 NPU serves hidden states (DP=1)
TRAIN_NPUS="1,2,3,4,5,6,7"  # 7 NPUs train (draft DP=7)
NUM_TRAIN_NPUS=7
VLLM_DP=1
# =======================================================================

echo "=== Step 0a: drafter config preview (what the trainer injects) ==="
# Mirrors the SPECULATORS saved schema: the qwen3 dims live nested under
# "transformer_layer_config" (built by create_transformer_layer_config), and the
# DFlash wrapper fields (built by from_training_args). NOTE: this is NOT the same
# layout as z-lab's published config.json, which flattens the qwen3 fields, uses
# architectures=["DFlashDraftModel"], dflash_config{}, auto_map, and an extra
# num_target_layers key. The numbers match; only the wrapper schema differs.
# The *authoritative* saved config is dumped at the end. Never aborts the run.
DRAFT_VOCAB_FOR_PREVIEW="$DRAFT_VOCAB_SIZE"
MODEL="$MODEL" NUM_LAYERS="$NUM_LAYERS" BLOCK_SIZE="$BLOCK_SIZE" \
MAX_ANCHORS="$MAX_ANCHORS" TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
DRAFT_VOCAB="$DRAFT_VOCAB_FOR_PREVIEW" python - <<'PY' || echo "[preview] skipped (non-fatal)"
import json, os, sys
try:
    model = os.environ["MODEL"]
    v = json.load(open(os.path.join(model, "config.json")))
    v = v.get("text_config", v)  # multimodal verifiers nest under text_config

    hidden = v["hidden_size"]
    n_heads = v["num_attention_heads"]
    n_kv = v["num_key_value_heads"]
    head_dim = v.get("head_dim")
    # Same fallback rule as create_transformer_layer_config()
    if head_dim and hidden % n_heads != 0 and hidden % head_dim == 0:
        n_heads = hidden // head_dim
        if n_heads % n_kv != 0:
            n_kv = n_heads

    n_layers = int(os.environ["NUM_LAYERS"])
    target_ids = [int(x) for x in os.environ["TARGET_LAYER_IDS"].split()]
    draft_vocab = int(os.environ["DRAFT_VOCAB"])
    full_vocab = v["vocab_size"]
    block_size = int(os.environ["BLOCK_SIZE"])

    drafter = {
        # ---- DFlash wrapper (from_training_args) ----
        "speculators_model_type": "dflash",
        "architectures": ["DFlashSpeculator"],
        "block_size": block_size,
        "max_anchors": int(os.environ["MAX_ANCHORS"]),
        "draft_vocab_size": full_vocab if draft_vocab >= full_vocab else draft_vocab,
        "mask_token_id": 151669,
        "aux_hidden_state_layer_ids": target_ids,
        "sliding_window_non_causal": False,
        "proposal_speculative_tokens": block_size - 1,   # = 15, set from block_size
        # ---- draft backbone (create_transformer_layer_config) ----
        "transformer_layer_config": {
            "model_type": v.get("model_type", "qwen3"),
            "vocab_size": full_vocab,
            "hidden_size": hidden,
            "intermediate_size": v["intermediate_size"],
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": v.get("max_position_embeddings"),
            "rms_norm_eps": v.get("rms_norm_eps"),
            "rope_theta": v.get("rope_theta"),
            "tie_word_embeddings": False,
            "layer_types": ["full_attention"] * n_layers,
        },
        # ---- informational only (NOT keys the speculators trainer writes) ----
        "_info": {
            "vocab_mapping_used": draft_vocab < full_vocab,
            "verifier_num_layers": v["num_hidden_layers"],  # z-lab calls this num_target_layers
            "note": "z-lab config.json flattens transformer_layer_config and adds "
                    "num_target_layers / dflash_config / auto_map; dims are identical.",
        },
    }
    print(json.dumps(drafter, indent=2))
except Exception as e:  # never block training on a preview failure
    print(f"[preview] could not derive drafter config: {e}", file=sys.stderr)
PY

echo "=== Step 0b: training hyperparameters (effective values) ==="
# These are the ACTUAL knobs in effect. Several are fixed in the trainer code and
# are NOT exposed as flags (noted below) — included so the log is self-describing.
TOKENS_PER_STEP=$(( TOTAL_SEQ_LEN * NUM_TRAIN_NPUS ))
cat <<EOF
optimizer              : adamw (betas=(0.9,0.999), eps=1e-8  [torch defaults, not exposed])
learning_rate          : $LR
weight_decay           : 0.01            [default]
lr_scheduler           : cosine, num_cycles=0.5, warmup = total_steps // 100 (1%, since --scheduler-warmup-steps unset)
epochs                 : $EPOCHS
seed                   : $SEED
data_parallel (ranks)  : $NUM_TRAIN_NPUS  (FSDP, one packed micro-batch per rank)
batch policy           : token-budgeted multipack, NOT a fixed sample count
  per-sample max len   : $SEQ_LENGTH tokens (prepare_data --seq-length)
  batch_max_length     : $TOTAL_SEQ_LEN tokens PER RANK (= --total-seq-len) -> headroom for ~$TARGET_BATCH x $SEQ_LENGTH-token samples
  ~tokens / opt step   : ~$TOKENS_PER_STEP  ($TOTAL_SEQ_LEN x $NUM_TRAIN_NPUS ranks)
  seqs / rank / step   : ~$TARGET_BATCH if samples are near max len; MORE if shorter (variable)
gradient_accumulation  : 1  (no accumulation; step every micro-batch)  [fixed in trainer]
grad_clip_max_norm     : 1.0  [fixed in trainer.py, not exposed]
noise_std              : 0.0  (default is 0.05)
loss_fn                : kl_div            [default]
ttt_steps              : 3                 [default]
step_weight_beta       : 0.6              [default]
draft_attn_impl        : simple_flex_attention   [default]
hidden_states_dtype    : bfloat16          [default]
EOF

echo "=== Step 1: prepare_data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --overwrite

# Drop stale vocab mappings from any prior run so they REGENERATE at the
# DRAFT_VOCAB_SIZE set above (a cached mapping would otherwise size-mismatch).
rm -f "$OUTPUT_DIR"/d2t.npy "$OUTPUT_DIR"/t2d.npy

echo "=== Step 2: launch vLLM server (NPUs $VLLM_NPUS, DP=$VLLM_DP) ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --max-model-len 4096 \
       --gpu-memory-utilization 0.85 &

VLLM_PID=$!
cleanup() { echo "Stopping vLLM..."; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Waiting for server..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 2; done
echo "Server ready."

export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1

echo "=== Step 3: train (online, NPUs $TRAIN_NPUS, draft DP=$NUM_TRAIN_NPUS) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    $VOCAB_FLAG \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$TOTAL_SEQ_LEN" \
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
    --run-name dflash_separate_qwen3_4b \
    --log-dir ./logs/separate_qwen3_4b \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 180 \
    --max-retries 8 \
    --log-freq 10 \
    --seed "$SEED"

echo "=== Step 4: final drafter config (authoritative, from saved checkpoint) ==="
# save_pretrained() writes the COMPLETE DFlashSpeculatorConfig (incl. the nested
# transformer_layer_config) to checkpoints/<epoch>/config.json. Dump the newest.
FINAL_CFG=$(ls -t "$OUTPUT_DIR"/checkpoints/*/config.json 2>/dev/null | head -1)
if [ -n "${FINAL_CFG:-}" ]; then
    echo "--- $FINAL_CFG ---"
    python -m json.tool "$FINAL_CFG" 2>/dev/null || cat "$FINAL_CFG"
else
    echo "[final] no saved config.json found under $OUTPUT_DIR/checkpoints/"
fi

echo "Done. Checkpoints: $OUTPUT_DIR/checkpoints/  |  TB logs: ./logs/separate_qwen3_4b"