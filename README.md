# Ascend NPU DFlash Qwen3-4B Training Setup

This document describes the working source-based setup for training a DFlash draft model with:

- `vllm`
- `vllm-ascend`
- `speculators`

The intended parent workspace is:

```bash
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
```

The parent repository branch is:

```bash
spec_main
```

The parent repository contains three Git submodules:

```text
vLLM_NPU_spec_main/
├── vllm/         # source checkout of vLLM
├── vllm-ascend/  # source checkout of vllm-ascend
└── speculators/  # source checkout of speculators
```

The training script used for Qwen3-4B DFlash is:

```bash
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/examples/ascend_npu_dflash/dflash_qwen3_4b.sh
```

---

## 1. Known-good versions

This setup was validated with the following stack.

| Component | Version / branch | Notes |
|---|---:|---|
| OS | Linux | Ascend A2 node |
| Python | 3.11 | Conda env |
| CANN | 9.0.0 | Required by `torch-npu` and `vllm-ascend` |
| NNAL / ATB | 9.0.0 | Required for `libatb.so` |
| PyTorch | 2.10.0+cpu | Used together with `torch-npu` |
| torch-npu | 2.10.0 | Provides NPU backend |
| vLLM | v0.20.2 | Source install from submodule |
| vllm-ascend | v0.20.2rc1 | Source install from submodule |
| triton-ascend | 3.2.1 | Install last |
| NumPy | 1.26.4 | Required by `triton-ascend==3.2.1` |
| Speculators | `spec_main` fork branch | Must include PR #589 / `--draft-attn-impl` |
| DFlash attention backend | `sdpa` | Required on Ascend; do not use flex attention |

The vLLM Ascend documentation lists Linux, Python >=3.10 and <3.13, Ascend NPU hardware, CANN 9.0.0, `torch==2.10.0`, `torch-npu==2.10.0`, and NNAL 9.0.0 as requirements for the current Ascend stack. The vLLM Ascend repository also notes that vLLM Ascend is the community-maintained Ascend plugin for vLLM and that its release branches are paired with corresponding vLLM versions.

References:

- vLLM Ascend installation docs: https://docs.vllm.ai/projects/ascend/en/latest/installation.html
- vLLM Ascend repository: https://github.com/vllm-project/vllm-ascend
- Speculators documentation: https://docs.vllm.ai/en/latest/features/speculative_decoding/speculators/
- Speculators PR #589: https://github.com/vllm-project/speculators/pull/589

---

## 2. System dependencies

Install system build tools.

### Ubuntu / Debian

```bash
sudo apt-get update -y
sudo apt-get install -y gcc g++ cmake libnuma-dev git curl wget jq
```

### RHEL / openEuler

```bash
sudo yum install -y gcc gcc-c++ cmake numactl-devel git curl wget jq
```

---

## 3. Clone the parent repository with submodules

Clone your parent repo branch:

```bash
mkdir -p /home/n84449292/m84379596/DFlash
cd /home/n84449292/m84379596/DFlash

git clone --branch spec_main --recurse-submodules \
  git@github-real:MohammadMahdi1375/vLLM_NPU.git \
  vLLM_NPU_spec_main
```

If already cloned without submodules:

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
git switch spec_main
git submodule update --init --recursive
```

Expected layout:

```bash
ls /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
```

should include:

```text
vllm
vllm-ascend
speculators
```

---

## 4. Recommended Git remote structure

Each submodule should have:

```text
origin   = your fork, used for pushing
upstream = official vllm-project repo, used for pulling official updates
```

Expected remotes:

### Parent repo

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
git remote -v
```

Expected:

```text
origin  git@github-real:MohammadMahdi1375/vLLM_NPU.git
```

### vLLM submodule

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm
git remote -v
```

Expected:

```text
origin    git@github-real:MohammadMahdi1375/vllm.git
upstream  https://github.com/vllm-project/vllm.git
```

### vllm-ascend submodule

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm-ascend
git remote -v
```

Expected:

```text
origin    git@github-real:MohammadMahdi1375/vllm-ascend.git
upstream  https://github.com/vllm-project/vllm-ascend.git
```

### speculators submodule

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators
git remote -v
```

Expected:

```text
origin    git@github-real:MohammadMahdi1375/speculators.git
upstream  https://github.com/vllm-project/speculators.git
```

---

## 5. Create a clean conda environment

Do not reuse the older `vllm-dflash` environment for this stack.

Create a fresh env:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y \
  -p /home/n84449292/m84379596/conda/vllm-ascend-0202 \
  python=3.11

conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

python -m pip install --upgrade pip setuptools wheel
```

---

## 6. Source CANN and NNAL

Use the CANN/NNAL installation already available on the node:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
```

Verify NPU driver visibility:

```bash
npu-smi info
```

---

## 7. Install vLLM from source

### Important

On `aarch64`, do not run plain `pip install vllm`. It may fall back to a CUDA-oriented source build and fail with `CUDA_HOME is not set`.

Use:

```bash
VLLM_TARGET_DEVICE=empty
```

because the NPU kernels come from `vllm-ascend`, not from vLLM.

Install from the submodule:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/vllm"

git fetch upstream --tags
git checkout spec_main 2>/dev/null || git switch -c spec_main

VLLM_TARGET_DEVICE=empty python -m pip install -e .
```

If the submodule is pinned to `v0.20.2`, verify:

```bash
git describe --tags --always
```

Expected base:

```text
v0.20.2
```

---

## 8. Install vllm-ascend from source

Use source install because modifications to `.py` files in `vllm-ascend/` should affect training.

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/vllm-ascend"

git fetch upstream --tags
git checkout spec_main 2>/dev/null || git switch -c spec_main
git submodule update --init --recursive
```

Install build dependencies into the current env. This avoids isolated-build failures where `triton-ascend` is not visible to the build env.

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  attrs "cmake>=3.26" decorator einops googleapis-common-protos numpy packaging pip \
  pybind11 pyyaml scipy pandas pandas-stubs psutil "setuptools>=64" "setuptools-scm>=8" \
  transformers==5.5.3 torch==2.10.0 torch-npu==2.10.0 torchvision wheel msgpack quart \
  numba "xgrammar>=0.1.30" "fastapi<0.124.0" "compressed_tensors>=0.11.0" \
  arctic-inference==0.1.1 triton-ascend==3.2.1
```

Then install `vllm-ascend` editable:

```bash
python -m pip install -e . --no-build-isolation
```

If you only changed Python files later, no rebuild is usually needed. If you changed compiled custom ops/C++/NPU code, rerun:

```bash
cd "$SPEC_MAIN/vllm-ascend"
python -m pip install -e . --no-build-isolation
```

---

## 9. Backfill CANN Python dependencies

A fresh conda env may miss packages required by CANN Python modules.

```bash
python -m pip install \
  decorator "scipy>=1.7.3" ml-dtypes tornado absl-py attrs psutil pyyaml
```

Optional profiling tools:

```bash
python -m pip install matplotlib "pandas~=2.2" openpyxl
```

---

## 10. Install triton-ascend last

Install `triton-ascend` after the rest of the stack:

```bash
python -m pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Then pin NumPy to the version required by `triton-ascend==3.2.1`:

```bash
python -m pip install --force-reinstall numpy==1.26.4
```

Verify:

```bash
python - <<'PY'
import numpy
import scipy

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
PY
```

Expected:

```text
numpy: 1.26.4
```

---

## 11. Install speculators from source

Install the `speculators` submodule in editable mode:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/speculators"

python -m pip install -e . --no-deps
```

Install runtime dependencies used by the trainer:

```bash
python -m pip install \
  click \
  "datasets>=4.0.0,<=5.0.0" \
  huggingface-hub \
  "loguru>=0.7.2,<=0.7.3" \
  openai \
  protobuf \
  psutil \
  "pydantic>=2.0.0" \
  pydantic-settings \
  rich \
  safetensors \
  tensorboard \
  "tqdm>=4.66.3,<=4.68.3" \
  "typer>=0.12.0"
```

TensorBoard is required because the current training script uses:

```bash
--logger tensorboard
```

---

## 12. Create conda activation hook

This makes every activation repeatable and prevents old repo paths from leaking into the environment.

```bash
ENV_DIR=/home/n84449292/m84379596/conda/vllm-ascend-0202

mkdir -p "$ENV_DIR/etc/conda/activate.d"

cat > "$ENV_DIR/etc/conda/activate.d/vllm_ascend_0202.sh" <<'HOOK'
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

# Clear old workspace paths first.
unset PYTHONPATH

# Source CANN/NNAL after clearing PYTHONPATH so their Python packages, including acl, are added.
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

# Prepend source repos while preserving CANN Python paths.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

export VLLM_USE_V1=1
HOOK
```

Reactivate:

```bash
conda deactivate
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202
```

---

## 13. Verify imports and NPU availability

Run:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

python - <<'PY'
import inspect
import acl
from acl.rt import memcpy
import torch
import torch_npu
import vllm
import vllm_ascend
import speculators
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

print("acl OK")
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
print("speculators:", speculators.__file__)
print("load_and_preprocess_dataset:", inspect.signature(load_and_preprocess_dataset))
PY
```

Expected paths:

```text
vllm:        /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm/vllm/__init__.py
vllm_ascend: /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm-ascend/vllm_ascend/__init__.py
speculators: /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/src/speculators/__init__.py
NPU available: True
NPU count: 8
```

Also verify the vLLM CLI import:

```bash
python -m vllm.entrypoints.cli.main --help >/tmp/vllm_cli_help.log && echo "vLLM CLI OK"
```

Expected:

```text
vLLM CLI OK
```

---

## 14. Training script requirements

The Qwen3-4B script should be here:

```bash
$SPEC_MAIN/speculators/examples/ascend_npu_dflash/dflash_qwen3_4b.sh
```

Important requirements inside the script:

### Correct source path setup

The top of the script must preserve CANN Python paths:

```bash
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

unset PYTHONPATH

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
```

Do not do this:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
unset PYTHONPATH
```

because it removes the CANN Python paths and causes:

```text
ModuleNotFoundError: No module named 'acl'
```

### Use SDPA attention on Ascend

The trainer command must include:

```bash
--draft-attn-impl sdpa \
```

Do not use the default flex attention backend on Ascend. Without this, PyTorch raises:

```text
ValueError: FlexAttention is only supported on CUDA, CPU or HPU devices. Found input tensors on npu device.
```

### TensorBoard

If the script uses:

```bash
--logger tensorboard
```

then the env must include:

```bash
python -m pip install tensorboard
```

---

## 15. Current Qwen3-4B training configuration

The working script uses:

```bash
MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
DATASET="/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl"
OUTPUT_DIR="./output/dflash_separate_qwen3_4b"

VLLM_NPUS="0"
TRAIN_NPUS="1,2,3,4,5,6,7"
NUM_TRAIN_NPUS=7
VLLM_DP=1

PREP_SEQ_LEN=3072
VLLM_MAX_MODEL_LEN=$((PREP_SEQ_LEN + 256))
TOTAL_SEQ_LEN=3072

EPOCHS=1
LR=6e-4
SEED=42

SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"

VERIFIER_VOCAB=151936
DRAFT_VOCAB_SIZE=151936
MASK_TOKEN_ID=151669
```

The critical training argument is:

```bash
--draft-attn-impl sdpa
```

---

## 16. Run training

First clean stale processes:

```bash
pkill -9 -f "vllm serve|EngineCore|APIServer|launch_vllm.py|torchrun|scripts/train.py" 2>/dev/null || true
```

Then run:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators

mkdir -p logs

bash examples/ascend_npu_dflash/dflash_qwen3_4b.sh \
  2>&1 | tee logs/qwen3_4b_dflash_0202.log
```

Expected healthy signs:

```text
Server ready after ...s.
POST /v1/completions HTTP/1.1" 200 OK
Epoch 0 ...
```

---

## 17. Common errors and fixes

### Error: `load_and_preprocess_dataset() got an unexpected keyword argument 'trust_remote_code'`

Cause: Python imported an older `speculators` package.

Fix:

```bash
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
```

Then verify:

```bash
python - <<'PY'
import inspect
import speculators
from speculators.data_generation.preprocessing import load_and_preprocess_dataset
print(speculators.__file__)
print(inspect.signature(load_and_preprocess_dataset))
PY
```

The signature must include:

```text
trust_remote_code
```

---

### Error: `Glm47MoeModelToolParser has no attribute _extract_tool_call_regions`

Cause: incompatible `vllm` and `vllm-ascend` commits.

Fix: use the compatible pair:

```text
vllm        v0.20.2
vllm-ascend v0.20.2rc1
```

and make sure imports point to the submodules inside `vLLM_NPU_spec_main`.

---

### Error: `ModuleNotFoundError: No module named 'acl'`

Cause: CANN Python paths were removed from `PYTHONPATH`.

Fix: clear `PYTHONPATH` before sourcing CANN, not after:

```bash
unset PYTHONPATH
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
```

---

### Error: `FlexAttention is only supported on CUDA, CPU or HPU devices. Found input tensors on npu device.`

Cause: DFlash used the default flex attention backend.

Fix: add:

```bash
--draft-attn-impl sdpa \
```

to the `scripts/train.py` arguments.

---

### Error: `ModuleNotFoundError: No module named 'tensorboard'`

Cause: training script uses TensorBoard logger but package is missing.

Fix:

```bash
python -m pip install tensorboard
```

---

### Warning/conflict: `triton-ascend 3.2.1 requires numpy==1.26.4`

Fix:

```bash
python -m pip install --force-reinstall numpy==1.26.4
```

---

## 18. Git workflow for submodules

The parent repo only stores pointers to submodule commits. It does not directly store the full contents of `vllm`, `vllm-ascend`, or `speculators`.

When modifying a submodule, always:

1. Commit inside the submodule.
2. Push the submodule branch to your fork.
3. Commit the updated submodule pointer in the parent repo.
4. Push the parent repo.

### Example: modifying speculators

```bash
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/speculators"

git status
git add examples/ascend_npu_dflash/dflash_qwen3_4b.sh
git commit -m "Update Qwen3-4B Ascend DFlash training script"
git push origin spec_main
```

Then update parent pointer:

```bash
cd "$SPEC_MAIN"

git status
git add speculators
git commit -m "Update speculators submodule pointer"
git push origin spec_main
```

### Example: modifying vLLM

```bash
cd "$SPEC_MAIN/vllm"

git status
git add <modified-files>
git commit -m "Apply vLLM change"
git push origin spec_main

cd "$SPEC_MAIN"
git add vllm
git commit -m "Update vLLM submodule pointer"
git push origin spec_main
```

### Example: modifying vllm-ascend

```bash
cd "$SPEC_MAIN/vllm-ascend"

git status
git add <modified-files>
git commit -m "Apply vllm-ascend change"
git push origin spec_main

cd "$SPEC_MAIN"
git add vllm-ascend
git commit -m "Update vllm-ascend submodule pointer"
git push origin spec_main
```

---

## 19. Pulling future official changes

Each submodule has:

```text
origin   = your fork
upstream = official vllm-project repo
```

### Pull official changes into vLLM

```bash
cd "$SPEC_MAIN/vllm"

git fetch upstream --tags
git switch spec_main
git merge upstream/main
git push origin spec_main

cd "$SPEC_MAIN"
git add vllm
git commit -m "Update vLLM submodule from upstream"
git push origin spec_main
```

### Pull official changes into vllm-ascend

```bash
cd "$SPEC_MAIN/vllm-ascend"

git fetch upstream --tags
git switch spec_main
git merge upstream/main
git push origin spec_main

cd "$SPEC_MAIN"
git add vllm-ascend
git commit -m "Update vllm-ascend submodule from upstream"
git push origin spec_main
```

### Pull official changes into speculators

```bash
cd "$SPEC_MAIN/speculators"

git fetch upstream main
git switch spec_main
git merge upstream/main
git push origin spec_main

cd "$SPEC_MAIN"
git add speculators
git commit -m "Update speculators submodule from upstream"
git push origin spec_main
```

---

## 20. Checkout or test a specific official PR

Example for `speculators` PR:

```bash
cd "$SPEC_MAIN/speculators"

git fetch upstream pull/589/head:pr-589
git switch pr-589
```

To merge it into your branch:

```bash
git switch spec_main
git merge pr-589
git push origin spec_main

cd "$SPEC_MAIN"
git add speculators
git commit -m "Merge speculators PR 589"
git push origin spec_main
```

The same pattern works for `vllm` and `vllm-ascend`.

---

## 21. Final health checklist

Before training, run:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

python - <<'PY'
import acl
import torch
import torch_npu
import vllm
import vllm_ascend
import speculators

print("acl OK")
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
print("speculators:", speculators.__file__)
PY
```

Expected:

```text
acl OK
torch: 2.10.0+cpu
torch_npu: 2.10.0
NPU available: True
NPU count: 8
vllm: /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm/vllm/__init__.py
vllm_ascend: /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm-ascend/vllm_ascend/__init__.py
speculators: /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/src/speculators/__init__.py
```

If the paths do not match, fix `PYTHONPATH` before running training.

---

## 22. Minimal rerun command

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

pkill -9 -f "vllm serve|EngineCore|APIServer|launch_vllm.py|torchrun|scripts/train.py" 2>/dev/null || true

cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators

bash examples/ascend_npu_dflash/dflash_qwen3_4b.sh \
  2>&1 | tee logs/qwen3_4b_dflash_0202.log
```
