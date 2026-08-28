# Ascend DFlash2 Working Stack

## vLLM

Source commit:

`b389ac29465b33f9e9c534df221ea3c129e9793f`

Contains upstream DFlash2 support from vLLM PR #52816.

## vLLM-Ascend

Base working commit:

`8e61107606222a0cbb21d6c64fbb30c216669a29`

Includes local compatibility fixes in:

- `vllm_ascend/models/qwen3_dflash2.py`
- `vllm_ascend/ops/triton/mamba/precopy.py`

Target platform:

- Ascend A3
- `SOC_VERSION=ascend910_9372`
- CANN 9.1.0

## Speculators

Source commit:

`7a58fc56217632d8d179b665734fa2269e8d9ffa`

Speculators version generated locally:

`0.8.0.dev207`

Includes experimental DFlash2 training and checkpoint support.
