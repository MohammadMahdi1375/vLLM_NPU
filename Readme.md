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
- **Python 3.11** (a conda env is recommended)
- **PyTorch 2.10.0** + **torch-npu 2.10.0** + **triton-ascend 3.2.1**
- **numpy == 1.26.4**  (triton-ascend requires this exact version)
- **patch** (GNU patch — used to patch the bundled protobuf during the op build):
  ```bash
  conda install -y -c conda-forge patch
  ```

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

`install.sh` installs the three components **in order** with the correct flags, pins numpy, and verifies the result. It also auto-installs CANN 9.0.0 if it isn't already present.

**CANN auto-install / proxy.** If CANN isn't found, `install.sh` downloads it from the
Huawei OBS mirror into `$CANN_HOME` (default `~/CANN/CANN9.0.0`). Override the location
with `CANN_HOME=...`, and if your node needs a proxy to reach the internet, export it
**before** running (do not hard-code credentials anywhere in the repo):

```bash
export https_proxy="http://USER:PASS@HOST:PORT"; export http_proxy="$https_proxy"
CANN_HOME=/opt/cann/9.0.0 bash install.sh      # example with a custom CANN path
```

> The downloaded CANN is installed **user-space** (`--install-path`); it does not touch
> the system NPU driver/firmware. CANN 9.0.0 expects a recent driver — if the toolkit
> prints a driver/firmware mismatch warning, that firmware update is a separate, root-level step.

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

## 3. Build notes (why the flags matter)

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

## 4. Notes & limitations

- Build artifacts (`csrc/build/`, compiled `*.so`, `_cann_ops_custom/` contents,
  protobuf/abseil extraction dirs) are git-ignored and regenerated per machine.
- The op build is specific to the host arch + CANN version, so each node compiles
  vllm-ascend locally; everything here is source, so there is no wheel to ship.
- DFlash serve/training flags are version-specific to the `0.20.x` Ascend backend;
  see the vLLM-Ascend docs for the current `--speculative-config` form.

---

## 5. Attribution & license

This repository redistributes source from three Apache-2.0 projects. Their original
`LICENSE` files are retained in each subdirectory. All credit for the upstream code
belongs to their respective authors:

- **vLLM** — https://github.com/vllm-project/vllm
- **vLLM-Ascend** — https://github.com/vllm-project/vllm-ascend
- **Speculators** — https://github.com/neuralmagic/speculators

Bundled third-party source: **protobuf** (BSD-3-Clause) and **abseil-cpp**
(Apache-2.0), included as release tarballs for offline builds.

This bundle is provided as-is, under the terms of the respective upstream licenses.
