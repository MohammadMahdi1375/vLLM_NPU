#!/bin/bash
set -eo pipefail

cd /home/n84449292/m84379596/DFlash/vLLM_NPU/vllm

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "multiproc_executor" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

export PYTHONPATH=/home/n84449292/m84379596/DFlash/vLLM_NPU/vllm:/home/n84449292/m84379596/DFlash/vLLM_NPU/vllm-ascend:$PYTHONPATH
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=INFO

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800

export MASTER_ADDR=80.5.5.108
export MASTER_PORT=29501

export GLOO_SOCKET_IFNAME="$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')"
export HCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export NCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export GLOO_USE_IPV6=0

export no_proxy="localhost,127.0.0.1,80.5.5.108,80.5.5.109"
export NO_PROXY="$no_proxy"

echo "===== NODE 1 ENV ====="
echo "PWD=$(pwd)"
echo "IFACE=$GLOO_SOCKET_IFNAME"
ip -o -4 addr show dev "$GLOO_SOCKET_IFNAME"
python - <<'PY'
import vllm
print("vllm file:", getattr(vllm, "__file__", None))
from vllm import SamplingParams
print("SamplingParams OK:", SamplingParams)
PY

MODEL=/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16
LOG=dsv4_vllm_node109_raw.log

vllm serve "$MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 80.5.5.108 \
  --master-port 29501 \
  --headless \
  --enable-expert-parallel \
  --max-num-seqs 1 \
  --max-model-len 2048 \
  2>&1 | tee "$LOG"