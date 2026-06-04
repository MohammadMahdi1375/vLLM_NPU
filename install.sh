#!/usr/bin/env bash
# =============================================================================
#  vLLM_NPU installer
#  Installs (editable, from source): vllm, speculators, vllm-ascend
#  Auto-detects/installs CANN 9.0.0 if missing.
#  Target: Huawei Ascend A2 (910B), aarch64, CANN 9.0.0 line, Python 3.11
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[vLLM_NPU] repo root: ${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Config (override any of these via environment variables)
# -----------------------------------------------------------------------------
CANN_VERSION="${CANN_VERSION:-9.0.0}"
CANN_HOME="${CANN_HOME:-$HOME/CANN/CANN${CANN_VERSION}}"          # where CANN lives / will be installed
CANN_DOWNLOAD_DIR="${CANN_DOWNLOAD_DIR:-$HOME/cann_dl}"           # where .run files are cached
CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}"
CANN_REFERER="Referer: https://www.hiascend.com/"
# If your node reaches the internet through a proxy, export it BEFORE running, e.g.:
#   export https_proxy="http://USER:PASS@HOST:PORT"; export http_proxy="$https_proxy"
# (Never hard-code credentials in this file — it is published.)

# -----------------------------------------------------------------------------
# CANN helpers
# -----------------------------------------------------------------------------
source_cann() {
  local home="$1"; set +u
  source "${home}/ascend-toolkit/set_env.sh"
  [ -f "${home}/nnal/atb/set_env.sh" ] && source "${home}/nnal/atb/set_env.sh"   # ATB = LLM scenarios
  set -u
}

cann_complete() {  # "everything there" = toolkit + 910b ops + nnal/atb
  local home="$1"
  [ -f "${home}/ascend-toolkit/set_env.sh" ] \
    && [ -d "${home}/ascend-toolkit/latest/opp" ] \
    && [ -f "${home}/nnal/atb/set_env.sh" ]
}

install_cann() {
  local arch; arch="$(uname -m)"
  local toolkit="Ascend-cann-toolkit_${CANN_VERSION}_linux-${arch}.run"
  local ops="Ascend-cann-910b-ops_${CANN_VERSION}_linux-${arch}.run"   # A2 kernels/ops (renamed in 9.0.0)
  local nnal="Ascend-cann-nnal_${CANN_VERSION}_linux-${arch}.run"
  mkdir -p "${CANN_DOWNLOAD_DIR}" "${CANN_HOME}"
  ( cd "${CANN_DOWNLOAD_DIR}"
    for f in "${toolkit}" "${ops}" "${nnal}"; do
      if [ -s "${f}" ]; then
        echo "  have ${f}"
      else
        echo "  downloading ${f} ..."
        # -c resumes partials; --no-check-certificate for TLS-intercepting proxies
        wget -c --no-check-certificate --header="${CANN_REFERER}" -O "${f}.part" "${CANN_BASE_URL}/${f}"
        mv "${f}.part" "${f}"
      fi
      chmod +x "${f}"
    done
    echo "  installing toolkit ..."; "./${toolkit}" --full --quiet --install-path="${CANN_HOME}"
    source_cann "${CANN_HOME}"
    echo "  installing 910b ops ..."; "./${ops}"  --install --quiet --install-path="${CANN_HOME}"
    echo "  installing nnal ...";     "./${nnal}" --install --quiet --install-path="${CANN_HOME}"
  )
}

# -----------------------------------------------------------------------------
# 0. CANN: detect -> skip, else download + install
# -----------------------------------------------------------------------------
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
  echo "[vLLM_NPU] CANN already sourced (ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME}) — skipping CANN install."
elif cann_complete "${CANN_HOME}"; then
  echo "[vLLM_NPU] CANN ${CANN_VERSION} found at ${CANN_HOME} — sourcing, skipping install."
  source_cann "${CANN_HOME}"
else
  echo "[vLLM_NPU] CANN ${CANN_VERSION} not found at ${CANN_HOME} — downloading & installing (user-space)..."
  echo "           (set CANN_HOME / https_proxy first if you want a different path / proxy)"
  install_cann
  source_cann "${CANN_HOME}"
fi
echo "[vLLM_NPU] ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

# -----------------------------------------------------------------------------
# 1. Python-side prerequisites (these are version-pinned; install them yourself
#    from the Ascend index — this script only checks and pins numpy)
# -----------------------------------------------------------------------------
python -c "import torch, torch_npu" 2>/dev/null || {
  echo "ERROR: torch / torch_npu not importable. Install torch==2.10.0, torch-npu==2.10.0,"
  echo "       and triton-ascend==3.2.1 (from the Ascend index) before running this."
  exit 1
}

if ! command -v patch >/dev/null 2>&1; then
  echo "[vLLM_NPU] 'patch' not found — installing via conda-forge..."
  conda install -y -c conda-forge patch
fi
echo "[vLLM_NPU] patch: $(command -v patch)"

echo "[vLLM_NPU] pinning numpy==1.26.4 (triton-ascend requirement)..."
pip install --no-build-isolation "numpy==1.26.4"

# -----------------------------------------------------------------------------
# 2-4. The three components (editable, from source, dependency-free)
# -----------------------------------------------------------------------------
echo "[vLLM_NPU] installing vllm (editable)..."
( cd "${REPO_ROOT}/vllm"        && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps )

echo "[vLLM_NPU] installing speculators (editable)..."
( cd "${REPO_ROOT}/speculators" && pip install -e . --no-build-isolation --no-deps )

PB_SRC="${REPO_ROOT}/vllm-ascend/csrc/third_party/protobuf/protobuf-all-25.1.tar.gz"
[ -s "${PB_SRC}" ] || echo "WARNING: ${PB_SRC} missing/empty — vllm-ascend op build needs protobuf-25.1 source."
echo "[vLLM_NPU] installing vllm-ascend (editable, compiling custom ops)..."
( cd "${REPO_ROOT}/vllm-ascend" && rm -rf csrc/build build dist ./*.egg-info \
    && pip install -e . --no-build-isolation --no-deps )

# -----------------------------------------------------------------------------
# 5. Verify
# -----------------------------------------------------------------------------
echo "[vLLM_NPU] verifying..."
pip list --editable | grep -iE "vllm|speculators" || true
python - <<'PY'
import numpy, vllm, vllm_ascend, speculators
print("numpy      ", numpy.__version__)
print("vllm       ", vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
print("speculators", speculators.__file__)
print("ALL OK — no 'Failed to register custom ops' above means DFlash ops are compiled in.")
PY
echo "[vLLM_NPU] done."