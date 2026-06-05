# vLLM_NPU

Source-built, editable **vLLM + vLLM-Ascend + Speculators** stack for **Huawei Ascend A2 (910B)** NPUs, configured for **DFlash speculative decoding** on the CANN 9.0.0 line.

This repository bundles the following components so they can be cloned and installed together, with **no wheel dependency** (useful for multi-node clusters where prebuilt wheels aren't available):

| Component | Path | Role |
|-----------|------|------|
| `vllm` | `vllm/` | vLLM core (built with `VLLM_TARGET_DEVICE=empty`) |
| `vllm-ascend` | `vllm-ascend/` | Ascend NPU plugin + custom AscendC ops (incl. DFlash sparse attention) |
| `speculators` | `speculators/` | Speculative-decoding / DFlash draft training & config |
| `test_bench` | `test_bench/` | Inference benchmarking & analysis scripts |

All three install **editable** (`pip install -e`), so source edits take effect immediately — intended for development and multi-node training.

---

## 1. Prerequisites

**Hardware:** Ascend 910B (A2) NPU(s), aarch64 (Kunpeng) host.

**Software (install these first):**
- **CANN 9.0.0** toolkit + `910b` ops/kernels + `nnal` (ATB). **`install.sh` handles this
  automatically**: if CANN is already sourced (or present at `$CANN_HOME`) it's skipped,
  otherwise it's downloaded from the Huawei OBS mirror and installed user-space. If you
  prefer to source it yourself first:
  ```bash
  source <CANN_HOME>/ascend-toolkit/set_env.sh
  source <CANN_HOME>/nnal/atb/set_env.sh      # ATB — required for LLM scenarios
  ```
- **conda** (miniconda/anaconda) on `PATH`
- **An Ascend NPU node with a compatible driver** (`npu-smi info` works)

That's it for hard prerequisites — **`install.sh` provisions the rest itself**: it creates
the conda env (Python 3.11), installs the pinned Ascend stack
(**torch 2.10.0 + torch-npu 2.10.0 + triton-ascend 3.2.1 + numpy 1.26.4**), installs GNU
`patch`, and installs CANN 9.0.0 if it's missing or the wrong version.

> The vLLM and vLLM-Ascend versions must match (this repo pins the `0.20.x` line:
> vllm `0.20.2`, vllm-ascend `0.20.2rc2`). DFlash on Ascend requires the `0.20.x`
> line on CANN 9.0.0.

---

## 2. Install

```bash
git clone https://github.com/<YOUR_USERNAME>/vLLM_NPU.git
cd vLLM_NPU

# make sure CANN is sourced and the conda env is active, then:
bash install.sh
```

`install.sh` is end-to-end: it creates/activates the conda env, installs the Ascend
torch stack, version-checks and installs CANN 9.0.0 if needed, then builds the three
components (editable, from source) and verifies. Re-running it is safe (each step skips
if already satisfied).

### Configuration (all optional — override via env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENV_NAME` | `vllm-dflash` | conda env name to create/use |
| `ENV_PREFIX` | *(unset)* | use a **prefix** env at this path instead of a named env |
| `PYTHON_VERSION` | `3.11` | Python for a freshly created env |
| `CANN_HOME` | `~/CANN/CANN9.0.0` | where CANN is / gets installed |
| `CANN_FORCE_REINSTALL` | `0` | set `1` to force a CANN reinstall |
| `SKIP_TORCH_STACK` | `0` | set `1` if you manage torch/torch-npu/triton yourself |
| `PIP_EXTRA_INDEX_URL` | *(unset)* | extra pip index for `torch-npu` / `triton-ascend` |
| `TORCH_INDEX_URL` | *(unset)* | torch index (x86 only: `https://download.pytorch.org/whl/cpu`) |

Examples:

```bash
# Use an existing prefix env and your existing CANN location:
ENV_PREFIX=/home/me/conda/vllm-dflash \
CANN_HOME=/home/me/CANN/CANN9.0.0 \
bash install.sh

# Behind a proxy (never hard-code credentials in the repo):
export https_proxy="http://USER:PASS@HOST:PORT"; export http_proxy="$https_proxy"
bash install.sh
```

> CANN is installed **user-space** (`--install-path`); it does not touch the system NPU
> driver/firmware. If the toolkit prints a driver/firmware mismatch warning, that firmware
> update is a separate, root-level step. If `triton-ascend` can't be found by pip, set
> `PIP_EXTRA_INDEX_URL` to the Ascend pip index and re-run.

### Manual install (equivalent to install.sh)

```bash
# numpy pin (do not skip — an unconstrained resolve breaks triton-ascend)
pip install "numpy==1.26.4"

# 1) vllm — no device backend compiled
cd vllm        && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps && cd ..

# 2) speculators
cd speculators && pip install -e . --no-build-isolation --no-deps && cd ..

# 3) vllm-ascend — compiles the custom AscendC ops (needs CANN sourced + patch)
cd vllm-ascend && rm -rf csrc/build build dist *.egg-info \
               && pip install -e . --no-build-isolation --no-deps && cd ..
```

### Verify

```bash
pip list --editable | grep -iE "vllm|speculators"
python -c "import vllm, vllm_ascend, speculators; print('OK')"
```

A clean `OK` with **no** `Failed to register custom ops` warning means the DFlash
ops compiled in successfully.

---

## 3. How to use

> Paths below (`/home/...`, `/share/canada_group_folder/...`) are cluster-specific
> **examples** — adjust them to your checkpoints and repo location. `CANN_HOME` defaults
> to the example path but is overridable.

### Inference (DFlash speculative decoding)

**Step 1 — set up the runtime environment** (CANN toolkit + ATB libs). The ATB library
path must match the C++ ABI that torch was built with, hence the `ABI` probe:

```bash
export CANN_HOME=${CANN_HOME:-/home/n84449292/m84379596/CANN/CANN9.0.0}
set +u
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
set -u
ABI=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
export LD_LIBRARY_PATH="$CANN_HOME/nnal/atb/9.0.0/atb/cxx_abi_${ABI}/lib:${LD_LIBRARY_PATH:-}"
```

(optional) confirm the ATB runtime actually loads:

```bash
python3 -c "import ctypes; ctypes.CDLL('libatb.so'); print('libatb OK')"
```

**Step 2 — start the DFlash vLLM server** (Qwen3-8B verifier + DFlash draft, TP=8):

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU

TARGET=/share/canada_group_folder/ckpt/Qwen3-8B
DRAFT=/share/canada_group_folder/ckpt/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/071541888480df12d8a1ef7acbaabed88b0a8bd4/

VLLM_USE_V1=1 \
vllm serve "$TARGET" \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --max-num-seqs 64 \
    --max-model-len 4096 \
    --speculative-config '{"model":"'"$DRAFT"'","num_speculative_tokens":16,"draft_tensor_parallel_size":8}' \
    --host 0.0.0.0 \
    --port 30000
```

**Step 3 — benchmark the running server** (from a second shell). `no_proxy` keeps the
client talking to `localhost` directly instead of through the cluster HTTP proxy:

```bash
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

python test_bench/bench_dflash_vllm.py \
    --base-url http://localhost:30000 \
    --model "$TARGET" \
    --dataset gsm8k --num-prompts 128 --max-new-tokens 2048 --concurrency 1
```

### Single-node training on Qwen

> **TODO — to be finalized.** DFlash draft-model training is driven from `speculators/`
> (with SpecForge). The runtime environment is the same as **Step 1** above (source CANN +
> set the ATB `LD_LIBRARY_PATH`). Replace the placeholder below with the confirmed
> single-node launch command:
>
> ```bash
> # placeholder — replace with the actual training entrypoint/config
> torchrun --standalone --nproc_per_node=8 <training_script>.py \
>     --model Qwen3-8B --config <train_config> ...
> ```

### Multi-node training on Qwen

> **TODO — to be finalized.** Run **Step 1** on every node, then launch with the
> rendezvous flags (master address/port, per-node rank, node count). Placeholder:
>
> ```bash
> # on each node — NODE_RANK=0 is the master
> torchrun \
>     --nnodes="$NNODES" --node_rank="$NODE_RANK" \
>     --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
>     --nproc_per_node=8 <training_script>.py \
>     --model Qwen3-8B --config <train_config> ...
> ```

---

## 4. Build notes (why the flags matter)

- **`--no-deps` everywhere.** Pip's dependency resolver otherwise drags `numpy`
  back up to 2.x (breaking triton-ascend) and may disturb the pinned
  torch/torch-npu/triton-ascend set. Always install with `--no-deps`.
- **`--no-build-isolation`.** Builds against the active environment's torch-npu
  instead of a throwaway isolated env.
- **`patch` is required.** The vllm-ascend op build compiles a bundled protobuf
  (`protobuf-25.1`) and applies a patch to it. Without GNU `patch` the build fails
  with `Error 127` (command not found) deep inside the op packaging step.
- **protobuf source ships in this repo.** `vllm-ascend/csrc/third_party/protobuf/
  protobuf-all-25.1.tar.gz` is included so the op build doesn't need to download it
  (the build's internal downloader does not use an HTTP proxy). The abseil source
  tarball is likewise included under `vllm-ascend/csrc/third_party/pkg/`.
- **`COMPILE_CUSTOM_KERNELS` must stay enabled (default).** Setting it to `0` skips
  the AscendC op build — which would disable DFlash's sparse-attention op on the NPU.

---

## 5. Notes & limitations

- Build artifacts (`csrc/build/`, compiled `*.so`, `_cann_ops_custom/` contents,
  protobuf/abseil extraction dirs) are git-ignored and regenerated per machine.
- The op build is specific to the host arch + CANN version, so each node compiles
  vllm-ascend locally; everything here is source, so there is no wheel to ship.
- DFlash serve/training flags are version-specific to the `0.20.x` Ascend backend;
  see the vLLM-Ascend docs for the current `--speculative-config` form.

---

## 6. Attribution & license

This repository redistributes source from three Apache-2.0 projects. Their original
`LICENSE` files are retained in each subdirectory. All credit for the upstream code
belongs to their respective authors:

- **vLLM** — https://github.com/vllm-project/vllm
- **vLLM-Ascend** — https://github.com/vllm-project/vllm-ascend
- **Speculators** — https://github.com/neuralmagic/speculators

Bundled third-party source: **protobuf** (BSD-3-Clause) and **abseil-cpp**
(Apache-2.0), included as release tarballs for offline builds.

This bundle is provided as-is, under the terms of the respective upstream licenses.
