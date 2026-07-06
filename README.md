# Ascend NPU DFlash Qwen3-4B Training Setup

Comprehensive installation, environment, Git workflow, and training guide for running DFlash draft-model training on Huawei Ascend NPUs using:

- [`vllm`](https://github.com/vllm-project/vllm)
- [`vllm-ascend`](https://github.com/vllm-project/vllm-ascend)
- [`speculators`](https://github.com/vllm-project/speculators)
- Huawei CANN 9.0.0
- NNAL / ATB 9.0.0
- PyTorch + `torch-npu` 2.10.0

This setup is intended for source-based development, not only package usage. The goal is to make changes in `vllm`, `vllm-ascend`, or `speculators` and have those changes immediately reflected in training.

---

## Table of contents

1. [What this repository contains](#1-what-this-repository-contains)
2. [Validated stack](#2-validated-stack)
3. [Target directory layout](#3-target-directory-layout)
4. [High-level installation order](#4-high-level-installation-order)
5. [System dependencies](#5-system-dependencies)
6. [CANN, ops, NNAL, and ATB installation](#6-cann-ops-nnal-and-atb-installation)
7. [GitHub SSH/proxy setup](#7-github-sshproxy-setup)
8. [Clone parent repository and submodules](#8-clone-parent-repository-and-submodules)
9. [Submodule remote structure](#9-submodule-remote-structure)
10. [Create conda environment](#10-create-conda-environment)
11. [Source CANN/ATB and set paths](#11-source-cannatb-and-set-paths)
12. [Install vLLM from source](#12-install-vllm-from-source)
13. [Install vllm-ascend from source](#13-install-vllm-ascend-from-source)
14. [Install triton-ascend and pin NumPy](#14-install-triton-ascend-and-pin-numpy)
15. [Install speculators from source](#15-install-speculators-from-source)
16. [Create conda activation hook](#16-create-conda-activation-hook)
17. [Final health checks](#17-final-health-checks)
18. [Training script requirements](#18-training-script-requirements)
19. [Current Qwen3-4B DFlash training configuration](#19-current-qwen3-4b-dflash-training-configuration)
20. [Run training](#20-run-training)
21. [Development workflow with submodules](#21-development-workflow-with-submodules)
22. [Pulling upstream changes](#22-pulling-upstream-changes)
23. [Testing or merging official PRs](#23-testing-or-merging-official-prs)
24. [Troubleshooting](#24-troubleshooting)
25. [Minimal rerun checklist](#25-minimal-rerun-checklist)

---

## 1. What this repository contains

The parent repository is a coordination repository that keeps compatible source checkouts of:

```text
vLLM_NPU_spec_main/
├── vllm/          # source checkout of vLLM
├── vllm-ascend/   # source checkout of vLLM Ascend plugin
└── speculators/   # source checkout of vLLM speculators
```

The intended parent workspace is:

```bash
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
```

The parent repository branch is:

```bash
spec_main
```

The main Qwen3-4B DFlash training script is expected at:

```bash
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/examples/ascend_npu_dflash/dflash_qwen3_4b.sh
```

The design is:

- the parent repo stores the exact submodule pointers;
- each submodule is a real Git repo;
- each submodule uses the user's fork as `origin`;
- each submodule can also track the official `vllm-project` repo as `upstream`.

---

## 2. Validated stack

This setup was validated with the following stack.

| Component | Version / branch | Notes |
|---|---:|---|
| OS | Linux | Ascend A2 node |
| Architecture | `aarch64` | Important for vLLM source build behavior |
| Python | 3.11 | Conda environment |
| CANN | 9.0.0 | Required by `torch-npu` and `vllm-ascend` |
| Ascend ops | 910B ops 9.0.0 | Installed into CANN root |
| NNAL / ATB | 9.0.0 | Required for `libatb.so` |
| PyTorch | 2.10.0+cpu | Used together with `torch-npu` |
| torch-npu | 2.10.0 | Provides NPU backend |
| vLLM | v0.20.2 compatible source | Source install from submodule |
| vllm-ascend | v0.20.2rc compatible source | Source install from submodule |
| triton-ascend | 3.2.1 | Install after core dependencies |
| NumPy | 1.26.4 | Required by `triton-ascend==3.2.1` |
| Speculators | `spec_main` fork branch | Must include support for `--draft-attn-impl` |
| DFlash attention backend | `sdpa` | Required on Ascend; do not use flex attention |

References:

- vLLM Ascend installation docs: https://docs.vllm.ai/projects/ascend/en/latest/installation.html
- vLLM Ascend repository: https://github.com/vllm-project/vllm-ascend
- Speculators documentation: https://docs.vllm.ai/en/latest/features/speculative_decoding/speculators/
- Speculators PR #589: https://github.com/vllm-project/speculators/pull/589

---

## 3. Target directory layout

Recommended final layout:

```text
/home/n84449292/m84379596/
├── CANN/
│   └── CANN9.0.0/
│       ├── ascend-toolkit/
│       │   └── set_env.sh
│       ├── cann-9.0.0/
│       │   └── set_env.sh
│       └── nnal/
│           └── atb/
│               └── set_env.sh
├── CANN_9.0.0/
│   ├── Ascend-cann-toolkit_9.0.0_linux-aarch64.run
│   ├── Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
│   └── Ascend-cann-nnal_9.0.0_linux-aarch64.run
├── conda/
│   └── vllm-ascend-0202/
└── DFlash/
    └── vLLM_NPU_spec_main/
        ├── vllm/
        ├── vllm-ascend/
        └── speculators/
```

The important source commands are:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
```

If your toolkit installer prints a different CANN environment path, this also may exist:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/cann-9.0.0/set_env.sh
```

For this training stack, use ATB for large-model scenarios. Do not source `asdsip` together with ATB.

---

## 4. High-level installation order

Use this order:

1. Install CANN toolkit.
2. Install Ascend 910B ops.
3. Install NNAL / ATB.
4. Configure GitHub SSH over port 443 and proxy if needed.
5. Clone parent repo and submodules.
6. Create a clean Python 3.11 conda environment.
7. Source CANN and ATB.
8. Install `vllm` from source with `VLLM_TARGET_DEVICE=empty`.
9. Install `vllm-ascend` from source with `--no-build-isolation`.
10. Install `triton-ascend==3.2.1` and pin `numpy==1.26.4`.
11. Install `speculators` from source.
12. Create a conda activation hook.
13. Run import and NPU health checks.
14. Run DFlash training.

---

## 5. System dependencies

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

Check compiler tools:

```bash
gcc --version
g++ --version
cmake --version
git --version
```

---

## 6. CANN, ops, NNAL, and ATB installation

Assume the `.run` installers are in:

```bash
/home/n84449292/m84379596/CANN_9.0.0
```

Set the target install root:

```bash
cd /home/n84449292/m84379596/CANN_9.0.0

export CANN_ROOT=/home/n84449292/m84379596/CANN/CANN9.0.0

mkdir -p "$CANN_ROOT"
chmod +x *.run
```

Install in this order.

### 6.1 Install CANN toolkit

```bash
./Ascend-cann-toolkit_9.0.0_linux-aarch64.run \
  --install \
  --install-path="$CANN_ROOT"
```

When prompted:

```text
Do you accept the EULA to install CANN?[Y/N]
```

type:

```text
Y
```

### 6.2 Install 910B ops

```bash
./Ascend-cann-910b-ops_9.0.0_linux-aarch64.run \
  --install \
  --install-path="$CANN_ROOT"
```

Again accept the EULA with `Y`.

### 6.3 Install NNAL / ATB

```bash
./Ascend-cann-nnal_9.0.0_linux-aarch64.run \
  --install \
  --install-path="$CANN_ROOT"
```

Again accept the EULA with `Y`.

The successful NNAL installation should mention:

```text
Ascend-cann-atb_9.0.0_linux-aarch64.run install success
Ascend-cann-SIP_9.0.0_linux-aarch64.run install success
Ascend-cann-nnal_9.0.0_linux-aarch64.run install success
```

### 6.4 Verify CANN and ATB files

```bash
find /home/n84449292/m84379596/CANN/CANN9.0.0 -name set_env.sh
```

Expected important files:

```text
/home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
/home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
/home/n84449292/m84379596/CANN/CANN9.0.0/cann-9.0.0/set_env.sh
```

Source the environment:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
```

Verify:

```bash
npu-smi info
which atc || true
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -Ei 'ascend|atb' | head -20
```

### 6.5 Important ATB warning

NNAL may also install `asdsip`:

```bash
/home/n84449292/m84379596/CANN/CANN9.0.0/nnal/asdsip/set_env.sh
```

Do not source both ATB and ASDSIP at the same time. For large-model vLLM scenarios, use ATB:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
```

---

## 7. GitHub SSH/proxy setup

On some servers, GitHub port 22 is blocked:

```text
ssh: connect to host github.com port 22: Connection timed out
```

Use GitHub SSH over port 443:

```bash
ssh -T -p 443 git@ssh.github.com
```

If the cluster requires an HTTP proxy, define a Git SSH alias named `github-real`.

### 7.1 Generate SSH key

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

ssh-keygen -t ed25519 -f ~/.ssh/github_ed25519 -C "n84449292@80.5.5.115"

cat ~/.ssh/github_ed25519.pub
```

Add the printed public key to:

```text
GitHub → Settings → SSH and GPG keys → New SSH key
```

### 7.2 SSH config without proxy

```bash
cat > ~/.ssh/config <<'EOF'
Host github-real
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_ed25519
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

Test:

```bash
ssh -T github-real
```

Expected:

```text
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

### 7.3 SSH config with proxy

Do not commit real proxy passwords into the repository. Use placeholders or environment variables in documentation.

First confirm `nc` exists:

```bash
command -v nc
ls -lh /usr/bin/nc
```

Template:

```bash
cat > ~/.ssh/config <<'EOF'
Host github-real
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_ed25519
    IdentitiesOnly yes
    ProxyCommand /usr/bin/nc --proxy <PROXY_HOST>:<PROXY_PORT> --proxy-type http --proxy-auth '<PROXY_USER>:<PROXY_PASSWORD>' --proxy-dns remote %h %p

Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_ed25519
    IdentitiesOnly yes
    ProxyCommand /usr/bin/nc --proxy <PROXY_HOST>:<PROXY_PORT> --proxy-type http --proxy-auth '<PROXY_USER>:<PROXY_PASSWORD>' --proxy-dns remote %h %p
EOF

chmod 600 ~/.ssh/config
```

Test:

```bash
ssh -T github-real
```

If it works, you can use `git@github-real:owner/repo.git`.

---

## 8. Clone parent repository and submodules

Clone the parent repository:

```bash
mkdir -p /home/n84449292/m84379596/DFlash
cd /home/n84449292/m84379596/DFlash

git clone --branch spec_main --recurse-submodules \
  git@github-real:MohammadMahdi1375/vLLM_NPU.git \
  vLLM_NPU_spec_main
```

If the parent repo cloned but the submodules failed because of HTTPS SSL issues, enter the parent repo and fix `.gitmodules`.

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
```

Set submodule URLs to your forks over SSH:

```bash
git submodule set-url speculators git@github-real:MohammadMahdi1375/speculators.git
git submodule set-url vllm git@github-real:MohammadMahdi1375/vllm.git
git submodule set-url vllm-ascend git@github-real:MohammadMahdi1375/vllm-ascend.git

git submodule sync --recursive
```

Optionally add a global rewrite rule so GitHub HTTPS submodule URLs are redirected to the working SSH alias:

```bash
git config --global url."git@github-real:".insteadOf "https://github.com/"
git config --global --get-regexp 'url.*insteadOf'
```

Clean failed partial submodule folders:

```bash
git submodule deinit -f --all || true

rm -rf speculators vllm vllm-ascend
rm -rf .git/modules/speculators .git/modules/vllm .git/modules/vllm-ascend
```

Clone submodules again:

```bash
git submodule update --init --recursive --jobs 3
```

Verify:

```bash
git submodule status --recursive
ls -lh
```

Expected:

```text
speculators
vllm
vllm-ascend
```

Nested dependencies such as `vllm-ascend/csrc/third_party/catlass` and `catlass/3rdparty/googletest` should also clone.

---

## 9. Submodule remote structure

Each submodule should have:

```text
origin   = your fork, used for pushing your changes
upstream = official vllm-project repository, used for pulling official updates
```

Set them explicitly:

```bash
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
```

### 9.1 speculators

```bash
cd "$SPEC_MAIN/speculators"

git remote set-url origin git@github-real:MohammadMahdi1375/speculators.git
git remote remove upstream 2>/dev/null || true
git remote add upstream git@github-real:vllm-project/speculators.git
```

### 9.2 vLLM

```bash
cd "$SPEC_MAIN/vllm"

git remote set-url origin git@github-real:MohammadMahdi1375/vllm.git
git remote remove upstream 2>/dev/null || true
git remote add upstream git@github-real:vllm-project/vllm.git
```

### 9.3 vllm-ascend

```bash
cd "$SPEC_MAIN/vllm-ascend"

git remote set-url origin git@github-real:MohammadMahdi1375/vllm-ascend.git
git remote remove upstream 2>/dev/null || true
git remote add upstream git@github-real:vllm-project/vllm-ascend.git
```

Verify:

```bash
cd "$SPEC_MAIN"

git submodule foreach 'echo "=== $name ==="; git branch --show-current; git remote -v'
```

### 9.4 Switch submodules to working branches

Submodules often clone in detached HEAD mode because the parent repo stores exact commits. To work on them, switch each one to `spec_main`.

```bash
cd "$SPEC_MAIN/speculators"
git switch spec_main || git switch -c spec_main --track origin/spec_main

cd "$SPEC_MAIN/vllm"
git switch spec_main || git switch -c spec_main --track origin/spec_main

cd "$SPEC_MAIN/vllm-ascend"
git switch spec_main || git switch -c spec_main --track origin/spec_main
```

Commit the corrected `.gitmodules` in the parent repository:

```bash
cd "$SPEC_MAIN"

git status
git add .gitmodules speculators vllm vllm-ascend
git commit -m "Use SSH fork URLs for submodules"
git push origin spec_main
```

If Git says there is nothing to commit, that is fine.

---

## 10. Create conda environment

Do not reuse older environments such as `vllm-dflash` for this stack.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y \
  -p /home/n84449292/m84379596/conda/vllm-ascend-0202 \
  python=3.11

conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

python -m pip install --upgrade pip setuptools wheel
```

---

## 11. Source CANN/ATB and set paths

Always clear old `PYTHONPATH` before sourcing CANN. This prevents old repo paths from leaking into the environment while preserving CANN Python packages such as `acl`.

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

unset PYTHONPATH

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
```

Verify:

```bash
python - <<'PY'
import sys
print("\n".join(sys.path[:10]))
PY
```

---

## 12. Install vLLM from source

On `aarch64`, do not run:

```bash
pip install vllm
```

It may fall back to a CUDA-oriented source build and fail with:

```text
CUDA_HOME is not set
```

Use:

```bash
VLLM_TARGET_DEVICE=empty
```

because the NPU kernels are supplied by `vllm-ascend`, not by the base vLLM package.

Install:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/vllm"

VLLM_TARGET_DEVICE=empty python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import vllm
print(vllm.__file__)
PY
```

Expected path:

```text
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm/vllm/__init__.py
```

---

## 13. Install vllm-ascend from source

Use editable source install because changes in `vllm-ascend/` Python files should affect training immediately.

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

unset PYTHONPATH
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

cd "$SPEC_MAIN/vllm-ascend"

git submodule update --init --recursive
```

Install build dependencies into the current environment. This avoids isolated-build failures where `triton-ascend` or CANN-related packages are not visible to the build environment.

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

Install editable:

```bash
python -m pip install -e . --no-build-isolation
```

If only Python files were modified later, no rebuild is usually needed. If C++/custom ops/NPU compiled code was modified, rerun:

```bash
cd "$SPEC_MAIN/vllm-ascend"
python -m pip install -e . --no-build-isolation
```

---

## 14. Install triton-ascend and pin NumPy

Install or reinstall `triton-ascend` after the core stack:

```bash
python -m pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Then pin NumPy:

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

## 15. Install speculators from source

Install the submodule in editable mode:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/speculators"

python -m pip install -e . --no-deps
```

Install runtime dependencies used by training:

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

TensorBoard is required if the training script uses:

```bash
--logger tensorboard
```

---

## 16. Create conda activation hook

This makes every environment activation reproducible.

```bash
ENV_DIR=/home/n84449292/m84379596/conda/vllm-ascend-0202

mkdir -p "$ENV_DIR/etc/conda/activate.d"

cat > "$ENV_DIR/etc/conda/activate.d/vllm_ascend_0202.sh" <<'HOOK'
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

# Clear old workspace paths first.
unset PYTHONPATH

# Source CANN/ATB after clearing PYTHONPATH so CANN Python packages such as acl are added.
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

# Prepend source repositories while preserving CANN Python paths.
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

## 17. Final health checks

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

Expected important output:

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

Also verify the vLLM CLI:

```bash
python -m vllm.entrypoints.cli.main --help >/tmp/vllm_cli_help.log && echo "vLLM CLI OK"
```

Expected:

```text
vLLM CLI OK
```

---

## 18. Training script requirements

The Qwen3-4B script should be here:

```bash
$SPEC_MAIN/speculators/examples/ascend_npu_dflash/dflash_qwen3_4b.sh
```

### 18.1 Correct source path setup

At the top of the script, use this order:

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

That removes the CANN Python paths and can cause:

```text
ModuleNotFoundError: No module named 'acl'
```

### 18.2 Use SDPA attention on Ascend

The training command must include:

```bash
--draft-attn-impl sdpa \
```

Do not use the default flex attention backend on Ascend. Without this, PyTorch may raise:

```text
ValueError: FlexAttention is only supported on CUDA, CPU or HPU devices. Found input tensors on npu device.
```

### 18.3 TensorBoard

If the script uses:

```bash
--logger tensorboard
```

then install:

```bash
python -m pip install tensorboard
```

---

## 19. Current Qwen3-4B DFlash training configuration

The working script uses the following configuration style:

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

Important argument:

```bash
--draft-attn-impl sdpa
```

Resource split:

- NPU 0 runs vLLM/data-generation server.
- NPUs 1-7 run training.
- `VLLM_DP=1` uses one vLLM data-parallel worker.
- `NUM_TRAIN_NPUS=7` launches training on seven NPUs.

---

## 20. Run training

Clean stale processes first:

```bash
pkill -9 -f "vllm serve|EngineCore|APIServer|launch_vllm.py|torchrun|scripts/train.py" 2>/dev/null || true
```

Run:

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators

mkdir -p logs

bash examples/ascend_npu_dflash/dflash_qwen3_4b.sh \
  2>&1 | tee logs/qwen3_4b_dflash_0202.log
```

Healthy signs:

```text
Server ready after ...s.
POST /v1/completions HTTP/1.1" 200 OK
Epoch 0 ...
```

Useful live monitoring:

```bash
npu-smi info
tail -f logs/qwen3_4b_dflash_0202.log
```

---

## 21. Development workflow with submodules

The parent repository only stores submodule commit pointers. It does not store the full contents of `vllm`, `vllm-ascend`, or `speculators`.

When modifying a submodule:

1. Make and test the change inside the submodule.
2. Commit inside the submodule.
3. Push the submodule branch to your fork.
4. Go back to the parent repository.
5. Commit the updated submodule pointer.
6. Push the parent repository.

### 21.1 Example: modifying speculators

```bash
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

cd "$SPEC_MAIN/speculators"

git status
git add examples/ascend_npu_dflash/dflash_qwen3_4b.sh
git commit -m "Update Qwen3-4B Ascend DFlash training script"
git push origin spec_main
```

Then update the parent pointer:

```bash
cd "$SPEC_MAIN"

git status
git add speculators
git commit -m "Update speculators submodule pointer"
git push origin spec_main
```

### 21.2 Example: modifying vLLM

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

### 21.3 Example: modifying vllm-ascend

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

## 22. Pulling upstream changes

Each submodule should have:

```text
origin   = your fork
upstream = official vllm-project repo
```

### 22.1 Pull official changes into vLLM

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

### 22.2 Pull official changes into vllm-ascend

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

### 22.3 Pull official changes into speculators

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

If an upstream merge changes dependencies or compiled code, rerun the relevant installation step.

---

## 23. Testing or merging official PRs

Example for a `speculators` PR:

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

The same pattern works for `vllm` and `vllm-ascend`:

```bash
cd "$SPEC_MAIN/vllm"
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git switch pr-<PR_NUMBER>
```

---

## 24. Troubleshooting

### 24.1 GitHub port 22 timeout

Symptom:

```text
ssh: connect to host github.com port 22: Connection timed out
```

Fix: use `ssh.github.com` on port 443 through `github-real`.

```bash
ssh -T github-real
```

Expected:

```text
Hi <username>! You've successfully authenticated...
```

### 24.2 `github-real` cannot resolve

Symptom:

```text
ssh: Could not resolve hostname github-real: Name or service not known
```

Cause: `~/.ssh/config` does not define `Host github-real`.

Fix: create or repair `~/.ssh/config`.

### 24.3 Invalid proxy port

Symptom:

```text
Ncat: Invalid proxy port number "YOUR_PROXY_PORT". QUITTING.
```

Cause: SSH config still contains placeholder values.

Fix: replace `<PROXY_HOST>`, `<PROXY_PORT>`, `<PROXY_USER>`, and `<PROXY_PASSWORD>` with real values, or remove the proxy command if proxy is not needed.

### 24.4 Submodules fail with SSL certificate error

Symptom:

```text
fatal: unable to access 'https://github.com/...': SSL certificate problem: self signed certificate in certificate chain
```

Fix: use SSH URLs for submodules.

```bash
cd "$SPEC_MAIN"

git submodule set-url speculators git@github-real:MohammadMahdi1375/speculators.git
git submodule set-url vllm git@github-real:MohammadMahdi1375/vllm.git
git submodule set-url vllm-ascend git@github-real:MohammadMahdi1375/vllm-ascend.git

git submodule sync --recursive

git config --global url."git@github-real:".insteadOf "https://github.com/"

git submodule update --init --recursive --jobs 3
```

Avoid using:

```bash
git config --global http.sslVerify false
```

unless absolutely necessary.

### 24.5 `fatal: not a git repository`

Symptom:

```text
fatal: not a git repository (or any parent up to mount point /)
```

Cause: command was run in `/home/n84449292/m84379596/DFlash`, not inside the parent Git repo.

Fix:

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
git status
```

### 24.6 NNAL/ATB `set_env.sh` missing

Symptom:

```text
-bash: /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh: No such file or directory
```

Check:

```bash
find /home/n84449292/m84379596/CANN/CANN9.0.0 -name set_env.sh
```

If no ATB path exists, reinstall NNAL:

```bash
cd /home/n84449292/m84379596/CANN_9.0.0

export CANN_ROOT=/home/n84449292/m84379596/CANN/CANN9.0.0

./Ascend-cann-nnal_9.0.0_linux-aarch64.run \
  --install \
  --install-path="$CANN_ROOT"
```

### 24.7 EULA prompt interrupted installation

Symptom: installer prints EULA and returns to shell without success.

Fix: rerun each installer and type `Y` when prompted. Install order:

```text
toolkit → 910B ops → NNAL/ATB
```

### 24.8 `ModuleNotFoundError: No module named 'acl'`

Cause: CANN Python paths were removed from `PYTHONPATH`.

Correct order:

```bash
unset PYTHONPATH
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
```

Wrong order:

```bash
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
unset PYTHONPATH
```

### 24.9 `CUDA_HOME is not set` during vLLM install

Cause: plain `pip install vllm` or an incorrect vLLM build path tried to build CUDA components.

Fix:

```bash
cd "$SPEC_MAIN/vllm"
VLLM_TARGET_DEVICE=empty python -m pip install -e .
```

### 24.10 `Glm47MoeModelToolParser has no attribute _extract_tool_call_regions`

Cause: incompatible `vllm` and `vllm-ascend` commits.

Fix:

- keep `vllm` and `vllm-ascend` on the compatible submodule commits;
- ensure Python imports point to the submodules inside `vLLM_NPU_spec_main`;
- do not accidentally import a pip-installed global version.

Check:

```bash
python - <<'PY'
import vllm
import vllm_ascend
print(vllm.__file__)
print(vllm_ascend.__file__)
PY
```

### 24.11 `load_and_preprocess_dataset() got an unexpected keyword argument 'trust_remote_code'`

Cause: Python imported an older `speculators` package.

Fix:

```bash
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

python - <<'PY'
import inspect
import speculators
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

print(speculators.__file__)
print(inspect.signature(load_and_preprocess_dataset))
PY
```

The signature should include `trust_remote_code`.

### 24.12 FlexAttention not supported on NPU

Symptom:

```text
ValueError: FlexAttention is only supported on CUDA, CPU or HPU devices. Found input tensors on npu device.
```

Fix: add this to the training command:

```bash
--draft-attn-impl sdpa \
```

### 24.13 Missing TensorBoard

Symptom:

```text
ModuleNotFoundError: No module named 'tensorboard'
```

Fix:

```bash
python -m pip install tensorboard
```

### 24.14 NumPy conflict with triton-ascend

Symptom:

```text
triton-ascend 3.2.1 requires numpy==1.26.4
```

Fix:

```bash
python -m pip install --force-reinstall numpy==1.26.4
```

### 24.15 Editable installs and conda-pack

Editable installs are expected for development. Some packaging tools such as `conda-pack` may fail or warn because `vllm`, `vllm-ascend`, and `speculators` are installed in editable mode. This is normal for a source-development environment.

---

## 25. Minimal rerun checklist

Use this when the setup has already been installed and you only want to rerun training.

```bash
conda activate /home/n84449292/m84379596/conda/vllm-ascend-0202

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

pkill -9 -f "vllm serve|EngineCore|APIServer|launch_vllm.py|torchrun|scripts/train.py" 2>/dev/null || true

cd "$SPEC_MAIN/speculators"

mkdir -p logs

bash examples/ascend_npu_dflash/dflash_qwen3_4b.sh \
  2>&1 | tee logs/qwen3_4b_dflash_0202.log
```

Before rerunning, confirm:

```bash
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

Expected source paths:

```text
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm/
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/vllm-ascend/
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/
```

---

## 26. Quick status commands

Useful commands during development:

```bash
# Parent repo status
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
git status
git submodule status --recursive

# Show submodule branches and remotes
git submodule foreach 'echo "=== $name ==="; git branch --show-current; git remote -v'

# Check installed Python packages
python -m pip list | grep -Ei 'vllm|torch|npu|triton|numpy|speculators'

# Check NPU state
npu-smi info

# Check CANN/ATB libraries in env
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -Ei 'ascend|atb' | head -50
```
