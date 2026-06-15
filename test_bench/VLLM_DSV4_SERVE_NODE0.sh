cd /home/n84449292/m84379596/DFlash/vLLM_NPU/vllm

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "multiproc_executor" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

export PYTHONPATH=/home/n84449292/m84379596/DFlash/vLLM_NPU/vllm:/home/n84449292/m84379596/DFlash/vLLM_NPU/vllm-ascend:$PYTHONPATH
export VLLM_USE_V1=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export DSV4_VLLM_SERVE_PATCH=1
export DFLASH_DISABLE_QLI=1
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MASTER_ADDR=80.5.5.108
export MASTER_PORT=29501
export HCCL_CONNECT_TIMEOUT=1800

export GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')
export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export GLOO_USE_IPV6=0

MODEL=/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16

vllm serve "$MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 80.5.5.108 \
  --master-port 29501 \
  --enable-expert-parallel \
  --max-num-seqs 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 30000