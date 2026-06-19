#!/usr/bin/env bash
# =============================================================================
#  vLLM_NPU comprehensive installer
#  Sets up: conda env -> Ascend torch stack -> CANN 9.0.0 -> vllm, speculators,
#           vllm-ascend (all editable, from source).
#  Target: Huawei Ascend A2 (910B), aarch64, CANN 9.0.0 line, Python 3.11.
#
#  This version includes the manual fixes needed during installation:
#    - prefix conda env support
#    - existing CANN reuse
#    - missing catlass checkout
#    - protobuf-25.1 download saved as protobuf-all-25.1.tar.gz
#    - conda GCC/G++ >= 9 and explicit CC/CXX selection
#    - safer MAX_JOBS default for vllm-ascend custom op build
#    - Abseil <cstdint> patch needed with newer GCC
#    - runtime deps installed under constraints so numpy/torch-npu are not broken
#
#  Everything below is overridable via environment variables, e.g.:
#    ENV_PREFIX=/path/to/env CANN_HOME=/path/to/CANN9.0.0 bash install.sh
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- Config ---------------------------------------
# Conda environment: choose ONE — a named env (ENV_NAME) or a prefix path (ENV_PREFIX)
ENV_NAME="${ENV_NAME:-vllm-dflash}"
ENV_PREFIX="${ENV_PREFIX:-}"                 # if set, a prefix env at this path is used/created
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Ascend Python stack (pinned)
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.10.0}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-3.2.1}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"               # e.g. https://download.pytorch.org/whl/cpu (x86 only)
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"  # Ascend pip mirror
SKIP_TORCH_STACK="${SKIP_TORCH_STACK:-0}"            # 1 = you manage torch/torch-npu/triton yourself

# Component versions. Vendored repos may not have .git metadata, so pin pretend versions.
VLLM_PRETEND_VERSION="${VLLM_PRETEND_VERSION:-0.20.2}"
VLLM_ASCEND_PRETEND_VERSION="${VLLM_ASCEND_PRETEND_VERSION:-0.20.2rc2}"
SPECULATORS_PRETEND_VERSION="${SPECULATORS_PRETEND_VERSION:-0.6.0}"

# CANN
CANN_VERSION="${CANN_VERSION:-9.0.0}"
CANN_HOME="${CANN_HOME:-$HOME/CANN/CANN${CANN_VERSION}}"
CANN_DOWNLOAD_DIR="${CANN_DOWNLOAD_DIR:-$HOME/cann_dl}"
CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}"
CANN_REFERER="Referer: https://www.hiascend.com/"
CANN_FORCE_REINSTALL="${CANN_FORCE_REINSTALL:-0}"

# Build robustness knobs
MAX_JOBS="${MAX_JOBS:-4}"                         # avoid huge default like -j=192 during vllm-ascend build
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
INSTALL_CONDA_COMPILER="${INSTALL_CONDA_COMPILER:-1}" # install/use conda GCC/G++ if system GCC is too old
MIN_GCC_MAJOR="${MIN_GCC_MAJOR:-9}"
SKIP_RUNTIME_DEPS="${SKIP_RUNTIME_DEPS:-0}"

# vllm-ascend third-party fixes
CATLASS_REPO="${CATLASS_REPO:-https://gitcode.com/cann/catlass.git}"
CATLASS_COMMIT="${CATLASS_COMMIT:-41bf90da655bba3c66d0acd7e00abe33960ecfd6}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-25.1}"
PROTOBUF_URL="${PROTOBUF_URL:-https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-${PROTOBUF_VERSION}.tar.gz}"

# Proxy: if your node needs one, export https_proxy/http_proxy BEFORE running.
# Do NOT hard-code credentials here.

echo "[vLLM_NPU] repo root: ${REPO_ROOT}"

# Guard against README placeholder proxy values.
for _pv in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY; do
  _val="${!_pv:-}"
  if [ -n "${_val}" ] && printf '%s' "${_val}" | grep -qiE 'USER:PASS|HOST:PORT'; then
    echo "[proxy] WARNING: ${_pv}='${_val}' is the README placeholder, not a real proxy — unsetting."
    echo "        If your node needs a proxy, export a real one: https_proxy=http://user:pass@host:port"
    unset "${_pv}"
  fi
done

# ----------------------------- Conda env ------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[ -n "${CONDA_BASE}" ] || { echo "ERROR: conda not found on PATH. Install miniconda/anaconda first."; exit 1; }
set +u; source "${CONDA_BASE}/etc/profile.d/conda.sh"; set -u

if [ -n "${ENV_PREFIX}" ]; then
  ENV_PREFIX="${ENV_PREFIX%/}"                       # strip any trailing slash
  if [ -d "${ENV_PREFIX}/conda-meta" ]; then
    echo "[env] using existing prefix env: ${ENV_PREFIX}"
  elif [ -e "${ENV_PREFIX}" ] && [ -n "$(ls -A "${ENV_PREFIX}" 2>/dev/null)" ]; then
    echo "ERROR: '${ENV_PREFIX}' exists but is not a conda env (no conda-meta/)."
    echo "       Check the real path with: conda env list"
    echo "       Then re-run with the exact ENV_PREFIX, or use ENV_NAME for a named env."
    exit 1
  else
    echo "[env] creating prefix env at ${ENV_PREFIX} (python ${PYTHON_VERSION})"
    conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
  fi
  TARGET="${ENV_PREFIX}"
else
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[env] creating env '${ENV_NAME}' (python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  TARGET="${ENV_NAME}"
fi
set +u; conda activate "${TARGET}"; set -u
echo "[env] active python: $(python -c 'import sys;print(sys.executable)')"

# Global pip retry helper.
PIP_NET=(--retries 15 --timeout 300)
pip_retry() {
  local n
  for n in 1 2 3 4 5; do
    python -m pip install "${PIP_NET[@]}" "$@" && return 0
    echo "[pip] attempt ${n} failed; retrying in 5s..."
    sleep 5
  done
  return 1
}

# ----------------------------- CANN helpers ---------------------------------
source_cann() {
  local h="$1"
  set +u
  source "${h}/ascend-toolkit/set_env.sh"
  [ -f "${h}/nnal/atb/set_env.sh" ] && source "${h}/nnal/atb/set_env.sh"
  set -u
}

cann_complete() {
  local h="$1"
  [ -f "${h}/ascend-toolkit/set_env.sh" ] && \
  [ -d "${h}/ascend-toolkit/latest/opp" ] && \
  [ -f "${h}/nnal/atb/set_env.sh" ]
}

cann_detect_version() {
  local h="$1" vf v
  for vf in "${h}"/version.cfg "${h}"/ascend-toolkit/latest/version.cfg \
            "${h}"/ascend-toolkit/latest/*/version.info "${h}"/*/version.cfg; do
    [ -f "${vf}" ] || continue
    v="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${vf}" 2>/dev/null | head -1)"
    [ -n "${v}" ] && { echo "${v}"; return 0; }
  done
  echo "$2" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

install_cann() {
  local arch; arch="$(uname -m)"
  local tk="Ascend-cann-toolkit_${CANN_VERSION}_linux-${arch}.run"
  local ops="Ascend-cann-910b-ops_${CANN_VERSION}_linux-${arch}.run"
  local nn="Ascend-cann-nnal_${CANN_VERSION}_linux-${arch}.run"
  mkdir -p "${CANN_DOWNLOAD_DIR}" "${CANN_HOME}"

  (
    cd "${CANN_DOWNLOAD_DIR}"
    for f in "${tk}" "${ops}" "${nn}"; do
      if [ -s "${f}" ]; then
        echo "  have ${f}"
      else
        echo "  downloading ${f} ..."
        wget -c --no-check-certificate --header="${CANN_REFERER}" -O "${f}.part" "${CANN_BASE_URL}/${f}"
        mv "${f}.part" "${f}"
      fi
      chmod +x "${f}"
    done

    echo "  installing toolkit ..."
    "./${tk}" --full --quiet --install-path="${CANN_HOME}"

    source_cann "${CANN_HOME}"

    echo "  installing 910b ops ..."
    "./${ops}" --install --quiet --install-path="${CANN_HOME}"

    echo "  installing nnal ..."
    "./${nn}" --install --quiet --install-path="${CANN_HOME}"
  )
}

# ----------------------------- CANN: detect / version-check / install -------
need_cann=1
if [ "${CANN_FORCE_REINSTALL}" = "1" ]; then
  echo "[cann] CANN_FORCE_REINSTALL=1 — will (re)install ${CANN_VERSION}"
elif [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
  cv="$(cann_detect_version "${ASCEND_TOOLKIT_HOME}" "${ASCEND_TOOLKIT_HOME}")"
  if [ "${cv}" = "${CANN_VERSION}" ]; then
    echo "[cann] sourced CANN ${cv} matches required ${CANN_VERSION} — skipping install."
    need_cann=0
  else
    echo "[cann] sourced CANN version='${cv:-unknown}' != required ${CANN_VERSION} — will install to ${CANN_HOME}."
  fi
elif cann_complete "${CANN_HOME}"; then
  cv="$(cann_detect_version "${CANN_HOME}" "${CANN_HOME}")"
  if [ "${cv}" = "${CANN_VERSION}" ] || [ -z "${cv}" ]; then
    echo "[cann] found CANN at ${CANN_HOME} (version='${cv:-unknown}') — sourcing, skipping install."
    source_cann "${CANN_HOME}"
    need_cann=0
  else
    echo "[cann] CANN at ${CANN_HOME} is '${cv}', not ${CANN_VERSION} — will install."
  fi
else
  echo "[cann] no CANN ${CANN_VERSION} found — will download & install to ${CANN_HOME}."
fi

if [ "${need_cann}" = "1" ]; then
  install_cann
  source_cann "${CANN_HOME}"
fi

echo "[cann] ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

# ----------------------------- Ascend torch stack ---------------------------
torch_stack_ok() {
python - "$TORCH_VERSION" "$TORCH_NPU_VERSION" "$TRITON_ASCEND_VERSION" "$NUMPY_VERSION" <<'PY' 2>/dev/null
import importlib.metadata as m, sys
want = dict(zip(["torch", "torch-npu", "triton-ascend", "numpy"], sys.argv[1:5]))
for p, v in want.items():
    try:
        cur = m.version(p).split("+")[0]
    except Exception:
        sys.exit(1)
    if cur != v:
        sys.exit(1)
sys.exit(0)
PY
}

if [ "${SKIP_TORCH_STACK}" = "1" ]; then
  echo "[torch] SKIP_TORCH_STACK=1 — leaving torch stack as-is."
elif torch_stack_ok; then
  echo "[torch] torch/torch-npu/triton-ascend/numpy already at required versions — skipping."
else
  echo "[torch] installing Ascend torch stack (torch=${TORCH_VERSION}, torch-npu=${TORCH_NPU_VERSION}, triton-ascend=${TRITON_ASCEND_VERSION}, numpy=${NUMPY_VERSION}) ..."
  TIDX=()
  [ -n "${TORCH_INDEX_URL}" ] && TIDX=(--index-url "${TORCH_INDEX_URL}")

  XIDX=()
  [ -n "${PIP_EXTRA_INDEX_URL}" ] && XIDX=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")

  pip_retry "${TIDX[@]}" "torch==${TORCH_VERSION}"
  pip_retry pyyaml setuptools decorator
  pip_retry "${XIDX[@]}" "torch-npu==${TORCH_NPU_VERSION}"
  pip_retry "${XIDX[@]}" "triton-ascend==${TRITON_ASCEND_VERSION}" \
    || echo "WARN: triton-ascend still failed after retries — see README/manual fallback."
  pip_retry "numpy==${NUMPY_VERSION}"
fi

python -c "import torch, torch_npu" 2>/dev/null || {
  echo "ERROR: torch / torch_npu still not importable. Install them manually and re-run,"
  echo "       or set PIP_EXTRA_INDEX_URL to the Ascend index."
  exit 1
}

# ----------------------------- Compiler guard -------------------------------
gcc_major() {
  local bin="$1" v
  v="$("${bin}" -dumpfullversion 2>/dev/null || "${bin}" -dumpversion 2>/dev/null || true)"
  v="${v%%.*}"
  [ -n "${v}" ] && printf '%s\n' "${v}" || printf '0\n'
}

ensure_modern_compiler() {
  local cur=0
  if command -v gcc >/dev/null 2>&1; then
    cur="$(gcc_major gcc)"
  fi

  echo "[compiler] system gcc major: ${cur}"

  if [ "${INSTALL_CONDA_COMPILER}" = "1" ] && [ "${cur}" -lt "${MIN_GCC_MAJOR}" ]; then
    echo "[compiler] installing conda GCC/G++ because PyTorch extension builds require GCC >= ${MIN_GCC_MAJOR}."
    conda install -y -c conda-forge gcc_linux-aarch64 gxx_linux-aarch64
  elif [ "${INSTALL_CONDA_COMPILER}" = "1" ]; then
    if [ ! -x "${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-gcc" ]; then
      echo "[compiler] installing conda GCC/G++ for reproducible vllm-ascend build."
      conda install -y -c conda-forge gcc_linux-aarch64 gxx_linux-aarch64
    fi
  fi

  if [ -x "${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-gcc" ] && \
     [ -x "${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-g++" ]; then
    export CC="${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-g++"

    echo "[compiler] CC=${CC}"
    "${CC}" --version | head -1

    echo "[compiler] CXX=${CXX}"
    "${CXX}" --version | head -1
  else
    echo "[compiler] using system compiler: $(gcc --version | head -1 2>/dev/null || true)"
  fi
}

ensure_modern_compiler

# ----------------------------- build tool: patch ----------------------------
command -v patch >/dev/null 2>&1 || {
  echo "[deps] installing 'patch'..."
  conda install -y -c conda-forge patch
}

# numpy pin guard
pip_retry --no-build-isolation "numpy==${NUMPY_VERSION}"

# Build-time deps. With --no-build-isolation the editable builds run against THIS env.
echo "[build] installing build-time deps..."
pip_retry "setuptools>=64" setuptools_scm setuptools-git-versioning wheel packaging ninja cmake regex

# ----------------------------- vllm-ascend third-party repair ---------------
download_file() {
  local url="$1" out="$2"
  rm -f "${out}.part"

  if command -v curl >/dev/null 2>&1; then
    curl -kL --connect-timeout 30 --retry 5 --retry-delay 5 "${url}" -o "${out}.part"
  else
    wget --no-check-certificate -O "${out}.part" "${url}"
  fi

  mv "${out}.part" "${out}"
}

ensure_catlass() {
  local tp="${REPO_ROOT}/vllm-ascend/csrc/third_party"
  local d="${tp}/catlass"

  mkdir -p "${tp}"

  if [ -d "${d}/include" ]; then
    echo "[third_party] catlass found: ${d}"
    return 0
  fi

  echo "[third_party] catlass missing — cloning ${CATLASS_REPO}"
  rm -rf "${d}"
  git clone "${CATLASS_REPO}" "${d}"
  (
    cd "${d}"
    git checkout "${CATLASS_COMMIT}"
  )

  [ -d "${d}/include" ] || {
    echo "ERROR: catlass include directory not found after clone: ${d}/include"
    exit 1
  }
}

ensure_protobuf_tarball() {
  local pb_dir="${REPO_ROOT}/vllm-ascend/csrc/third_party/protobuf"
  local pb="${pb_dir}/protobuf-all-${PROTOBUF_VERSION}.tar.gz"

  mkdir -p "${pb_dir}"

  if [ -s "${pb}" ] && tar -tzf "${pb}" >/dev/null 2>&1; then
    echo "[third_party] protobuf tarball OK: ${pb}"
  else
    echo "[third_party] protobuf tarball missing/invalid — downloading ${PROTOBUF_URL}"
    rm -f "${pb}"
    download_file "${PROTOBUF_URL}" "${pb}"

    if ! tar -tzf "${pb}" >/dev/null 2>&1; then
      echo "ERROR: downloaded protobuf tarball is invalid: ${pb}"
      echo "       If GitHub is blocked, manually download ${PROTOBUF_URL} and save it as ${pb}"
      exit 1
    fi
  fi
}

patch_abseil_cstdint() {
  # GCC 15 can fail on vendored Abseil with:
  #   uintptr_t does not name a type
  # Fix by adding #include <cstdint> to container_memory.h.
  local vasc="${REPO_ROOT}/vllm-ascend"
  local pb="${vasc}/csrc/third_party/protobuf/protobuf-all-${PROTOBUF_VERSION}.tar.gz"
  local absl_dir="${vasc}/csrc/third_party/abseil-cpp"
  local hdr="${absl_dir}/absl/container/internal/container_memory.h"

  # On fresh repos, abseil-cpp may only exist inside the protobuf tarball; pre-extract it.
  if [ ! -f "${hdr}" ] && [ -s "${pb}" ]; then
    echo "[third_party] abseil-cpp source not found — trying to extract from protobuf tarball."
    local tmp
    tmp="$(mktemp -d)"

    if tar -xzf "${pb}" -C "${tmp}" "protobuf-${PROTOBUF_VERSION}/third_party/abseil-cpp" 2>/dev/null; then
      rm -rf "${absl_dir}"
      cp -a "${tmp}/protobuf-${PROTOBUF_VERSION}/third_party/abseil-cpp" "${vasc}/csrc/third_party/"
    fi

    rm -rf "${tmp}"
  fi

  if [ -f "${hdr}" ]; then
    if grep -q '#include <cstdint>' "${hdr}"; then
      echo "[patch] Abseil <cstdint> patch already present."
    else
      echo "[patch] patching Abseil: ${hdr}"
      python - "${hdr}" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

if "#include <cstdint>" not in text:
    if '#include "absl/utility/utility.h"\n' in text:
        text = text.replace(
            '#include "absl/utility/utility.h"\n',
            '#include "absl/utility/utility.h"\n#include <cstdint>\n'
        )
    else:
        text = text.replace(
            '#ifndef ABSL_CONTAINER_INTERNAL_CONTAINER_MEMORY_H_\n',
            '#ifndef ABSL_CONTAINER_INTERNAL_CONTAINER_MEMORY_H_\n#include <cstdint>\n'
        )

    p.write_text(text)

print(f"patched {p}")
PY
    fi
  else
    echo "[patch] Abseil header not found yet; build may generate it. If GCC complains about uintptr_t, rerun after first failure."
  fi
}

prepare_vllm_ascend_third_party() {
  ensure_catlass
  ensure_protobuf_tarball
  patch_abseil_cstdint
}

prepare_vllm_ascend_third_party

# ----------------------------- the three components -------------------------
echo "[build] vllm (editable, no deps)..."
(
  cd "${REPO_ROOT}/vllm"
  SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_PRETEND_VERSION}" \
  VLLM_TARGET_DEVICE=empty \
  python -m pip install -e . --no-build-isolation --no-deps
)

echo "[build] speculators (editable, no deps)..."
(
  cd "${REPO_ROOT}/speculators"
  SETUPTOOLS_SCM_PRETEND_VERSION="${SPECULATORS_PRETEND_VERSION}" \
  python -m pip install -e . --no-build-isolation --no-deps
)

PB="${REPO_ROOT}/vllm-ascend/csrc/third_party/protobuf/protobuf-all-${PROTOBUF_VERSION}.tar.gz"
[ -s "${PB}" ] || {
  echo "ERROR: ${PB} missing/empty — vllm-ascend op build needs protobuf source."
  exit 1
}

echo "[build] vllm-ascend (editable, compiling custom ops with MAX_JOBS=${MAX_JOBS})..."

# Re-source CANN and re-export compiler right before this build, because set_env.sh may alter env vars.
source_cann "${CANN_HOME}"
ensure_modern_compiler

export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_ASCEND_PRETEND_VERSION}"

(
  cd "${REPO_ROOT}/vllm-ascend"
  rm -rf csrc/build build dist ./*.egg-info
  python -m pip install -v -e . --no-build-isolation --no-deps 2>&1 | tee /tmp/vllm_ascend_build.log
) || {
  echo "ERROR: vllm-ascend build failed. Useful errors:"
  grep -nE "ERROR|Error|error:|fatal|failed|Failed|FAILED|Traceback|No such file|cannot|not found|Permission denied|Killed|killed|core dumped|does not name a type" /tmp/vllm_ascend_build.log | head -200 || true
  exit 1
}

# ----------------------------- runtime deps ---------------------------------
# The editable installs above used --no-deps so pip would not pull CUDA torch or numpy>=2.
# Install runtime deps with constraints. OpenCV is constrained below 4.12 because newer wheels
# require numpy>=2 on Python 3.11+, which conflicts with triton-ascend's numpy 1.26.4.
if [ "${SKIP_RUNTIME_DEPS}" = "1" ]; then
  echo "[deps] SKIP_RUNTIME_DEPS=1 — skipping runtime dependency install."
else
  echo "[deps] installing runtime dependencies under strict Ascend constraints..."

  CONSTRAINTS="$(mktemp)"
  cat > "${CONSTRAINTS}" <<EOF
numpy==${NUMPY_VERSION}
torch==${TORCH_VERSION}
torch-npu==${TORCH_NPU_VERSION}
torchvision==0.25.0
torchaudio==${TORCH_VERSION}
triton-ascend==${TRITON_ASCEND_VERSION}
opencv-python-headless<4.12
EOF

  # Try metadata-based vLLM runtime deps first.
  # If resolver conflicts due to opencv/numpy, fall back to explicit observed deps.
  (
    cd "${REPO_ROOT}/vllm"
    VLLM_TARGET_DEVICE=empty \
    python -m pip install "${PIP_NET[@]}" -c "${CONSTRAINTS}" --no-build-isolation -e .
  ) || {
    echo "[deps] WARN: full vLLM dependency solve failed. Falling back to explicit runtime dependency batch."
  }

  python -m pip install "${PIP_NET[@]}" -c "${CONSTRAINTS}" \
    pydantic pydantic-settings openai "datasets>=4.0.0,<=4.8.4" protobuf \
    transformers tokenizers safetensors accelerate sentencepiece regex loguru \
    msgspec cbor2 gguf pyzmq openai-harmony cloudpickle Pillow pybase64 cachetools uvloop py-cpuinfo \
    blake3 psutil prometheus-client prometheus-fastapi-instrumentator fastapi uvicorn starlette \
    lark partial-json-parser "opencv-python-headless<4.12"

  rm -f "${CONSTRAINTS}"
fi

# ----------------------------- verify ---------------------------------------
echo "[verify] package versions"
python - <<'PY'
import importlib.metadata as m

for p in ["numpy", "torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend", "speculators"]:
    try:
        print(f"{p:15s}", m.version(p))
    except Exception as e:
        print(f"{p:15s}", "<not found>", e)
PY

echo "[verify] import test"
python - <<'PY'
import numpy, torch, torch_npu, vllm, vllm_ascend
from vllm import SamplingParams
import speculators

print("numpy      ", numpy.__version__)
print("torch      ", torch.__version__)
print("torch_npu  ", torch_npu.__version__)
print("vllm       ", vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
print("speculators", speculators.__file__)
print("ALL IMPORTS OK")
PY

python -m pip check || true

echo "[vLLM_NPU] done. Activate with: conda activate ${TARGET}"
echo "[vLLM_NPU] If using an existing CANN, source it before runtime:"
echo "  source ${CANN_HOME}/ascend-toolkit/set_env.sh"
echo "  source ${CANN_HOME}/nnal/atb/set_env.sh"