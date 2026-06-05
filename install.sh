#!/usr/bin/env bash
# =============================================================================
#  vLLM_NPU comprehensive installer
#  Sets up: conda env -> Ascend torch stack -> CANN 9.0.0 -> vllm, speculators,
#           vllm-ascend (all editable, from source).
#  Target: Huawei Ascend A2 (910B), aarch64, CANN 9.0.0 line, Python 3.11.
#
#  Everything below is overridable via environment variables, e.g.:
#    ENV_PREFIX=/path/to/env  CANN_HOME=/opt/cann/9.0.0  bash install.sh
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
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"  # Ascend pip mirror (torch-npu / triton-ascend)
SKIP_TORCH_STACK="${SKIP_TORCH_STACK:-0}"            # 1 = you manage torch/torch-npu/triton yourself

# Component versions. These are vendored without their own .git, so setuptools_scm can't
# derive a version from tags — pin them explicitly (matches the Phase-1 build versions).
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
# Proxy: if your node needs one, export https_proxy/http_proxy BEFORE running.
# Do NOT hard-code credentials here (this file is public).

echo "[vLLM_NPU] repo root: ${REPO_ROOT}"

# Guard: if the README's placeholder proxy was pasted literally, it would crash
# conda/pip with a cryptic "Failed to parse" error. Detect and unset it.
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
    echo "       Check the real path with:  conda env list"
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

# ----------------------------- CANN helpers ---------------------------------
source_cann() { local h="$1"; set +u; source "${h}/ascend-toolkit/set_env.sh"; \
                [ -f "${h}/nnal/atb/set_env.sh" ] && source "${h}/nnal/atb/set_env.sh"; set -u; }
cann_complete() { local h="$1"; [ -f "${h}/ascend-toolkit/set_env.sh" ] && \
                  [ -d "${h}/ascend-toolkit/latest/opp" ] && [ -f "${h}/nnal/atb/set_env.sh" ]; }
cann_detect_version() {  # echo "X.Y.Z" found near a CANN home, else nothing
  local h="$1" vf v
  for vf in "${h}"/version.cfg "${h}"/ascend-toolkit/latest/version.cfg \
            "${h}"/ascend-toolkit/latest/*/version.info "${h}"/*/version.cfg; do
    [ -f "${vf}" ] || continue
    v="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${vf}" 2>/dev/null | head -1)"
    [ -n "${v}" ] && { echo "${v}"; return 0; }
  done
  echo "$2" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1   # fall back to a path hint
}
install_cann() {
  local arch; arch="$(uname -m)"
  local tk="Ascend-cann-toolkit_${CANN_VERSION}_linux-${arch}.run"
  local ops="Ascend-cann-910b-ops_${CANN_VERSION}_linux-${arch}.run"
  local nn="Ascend-cann-nnal_${CANN_VERSION}_linux-${arch}.run"
  mkdir -p "${CANN_DOWNLOAD_DIR}" "${CANN_HOME}"
  ( cd "${CANN_DOWNLOAD_DIR}"
    for f in "${tk}" "${ops}" "${nn}"; do
      if [ -s "${f}" ]; then echo "  have ${f}"; else
        echo "  downloading ${f} ..."
        wget -c --no-check-certificate --header="${CANN_REFERER}" -O "${f}.part" "${CANN_BASE_URL}/${f}"
        mv "${f}.part" "${f}"
      fi
      chmod +x "${f}"
    done
    echo "  installing toolkit ...";   "./${tk}"  --full    --quiet --install-path="${CANN_HOME}"
    source_cann "${CANN_HOME}"
    echo "  installing 910b ops ...";  "./${ops}" --install --quiet --install-path="${CANN_HOME}"
    echo "  installing nnal ...";      "./${nn}"  --install --quiet --install-path="${CANN_HOME}" )
}

# ----------------------------- CANN: detect / version-check / install -------
need_cann=1
if [ "${CANN_FORCE_REINSTALL}" = "1" ]; then
  echo "[cann] CANN_FORCE_REINSTALL=1 — will (re)install ${CANN_VERSION}"
elif [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
  cv="$(cann_detect_version "${ASCEND_TOOLKIT_HOME}" "${ASCEND_TOOLKIT_HOME}")"
  if [ "${cv}" = "${CANN_VERSION}" ]; then
    echo "[cann] sourced CANN ${cv} matches required ${CANN_VERSION} — skipping install."; need_cann=0
  else
    echo "[cann] sourced CANN version='${cv:-unknown}' != required ${CANN_VERSION} — will install to ${CANN_HOME}."
  fi
elif cann_complete "${CANN_HOME}"; then
  cv="$(cann_detect_version "${CANN_HOME}" "${CANN_HOME}")"
  if [ "${cv}" = "${CANN_VERSION}" ] || [ -z "${cv}" ]; then
    echo "[cann] found CANN at ${CANN_HOME} (version='${cv:-unknown}') — sourcing, skipping install."
    source_cann "${CANN_HOME}"; need_cann=0
  else
    echo "[cann] CANN at ${CANN_HOME} is '${cv}', not ${CANN_VERSION} — will install."
  fi
else
  echo "[cann] no CANN ${CANN_VERSION} found — will download & install to ${CANN_HOME}."
fi
if [ "${need_cann}" = "1" ]; then install_cann; source_cann "${CANN_HOME}"; fi
echo "[cann] ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

# ----------------------------- Ascend torch stack ---------------------------
torch_stack_ok() {
python - "$TORCH_VERSION" "$TORCH_NPU_VERSION" "$TRITON_ASCEND_VERSION" "$NUMPY_VERSION" <<'PY' 2>/dev/null
import importlib.metadata as m, sys
want=dict(zip(["torch","torch-npu","triton-ascend","numpy"], sys.argv[1:5]))
for p,v in want.items():
    try: cur=m.version(p).split("+")[0]
    except Exception: sys.exit(1)
    if cur!=v: sys.exit(1)
sys.exit(0)
PY
}
if [ "${SKIP_TORCH_STACK}" = "1" ]; then
  echo "[torch] SKIP_TORCH_STACK=1 — leaving torch stack as-is."
elif torch_stack_ok; then
  echo "[torch] torch/torch-npu/triton-ascend/numpy already at required versions — skipping."
else
  echo "[torch] installing Ascend torch stack (torch=${TORCH_VERSION}, torch-npu=${TORCH_NPU_VERSION}, triton-ascend=${TRITON_ASCEND_VERSION}, numpy=${NUMPY_VERSION}) ..."
  PIP_NET=(--retries 15 --timeout 300)            # survive flaky-proxy IncompleteRead on big wheels
  pip_retry() {                                   # retry whole pip install up to 5x (pip resumes partial downloads between tries)
    local n; for n in 1 2 3 4 5; do
      pip install "${PIP_NET[@]}" "$@" && return 0
      echo "[pip] attempt ${n} failed; retrying in 5s..."; sleep 5
    done
    return 1
  }
  TIDX=(); [ -n "${TORCH_INDEX_URL}" ]     && TIDX=(--index-url "${TORCH_INDEX_URL}")
  XIDX=(); [ -n "${PIP_EXTRA_INDEX_URL}" ] && XIDX=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
  pip_retry "${TIDX[@]}" "torch==${TORCH_VERSION}"
  pip install "${PIP_NET[@]}" pyyaml setuptools decorator
  pip_retry "${XIDX[@]}" "torch-npu==${TORCH_NPU_VERSION}"
  pip_retry "${XIDX[@]}" "triton-ascend==${TRITON_ASCEND_VERSION}" \
    || echo "WARN: triton-ascend still failed after retries — see the README 'triton-ascend' note for the manual wget fallback."
  pip install "${PIP_NET[@]}" "numpy==${NUMPY_VERSION}"
fi

python -c "import torch, torch_npu" 2>/dev/null || {
  echo "ERROR: torch / torch_npu still not importable. Install them manually (see README) and re-run,"
  echo "       or set PIP_EXTRA_INDEX_URL to the Ascend index."; exit 1; }

# ----------------------------- build tool: patch ----------------------------
command -v patch >/dev/null 2>&1 || { echo "[deps] installing 'patch'..."; conda install -y -c conda-forge patch; }

# numpy pin (guard against any resolver bumping it)
pip install --no-build-isolation "numpy==${NUMPY_VERSION}"

# Build-time deps. With --no-build-isolation the editable builds run against THIS env,
# so the PEP 518 build requirements (e.g. setuptools_scm for vllm) must be present here.
echo "[build] installing build-time deps (setuptools_scm, setuptools-git-versioning, wheel, packaging, ninja, cmake, regex)..."
pip install --retries 15 --timeout 300 "setuptools>=64" setuptools_scm setuptools-git-versioning wheel packaging ninja cmake regex

# ----------------------------- the three components -------------------------
echo "[build] vllm (editable)..."
( cd "${REPO_ROOT}/vllm" && SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_PRETEND_VERSION}" \
    VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps )

echo "[build] speculators (editable)..."
( cd "${REPO_ROOT}/speculators" && SETUPTOOLS_SCM_PRETEND_VERSION="${SPECULATORS_PRETEND_VERSION}" \
    pip install -e . --no-build-isolation --no-deps )

PB="${REPO_ROOT}/vllm-ascend/csrc/third_party/protobuf/protobuf-all-25.1.tar.gz"
[ -s "${PB}" ] || echo "WARNING: ${PB} missing/empty — vllm-ascend op build needs protobuf-25.1 source."
echo "[build] vllm-ascend (editable, compiling custom ops)..."
( cd "${REPO_ROOT}/vllm-ascend" && rm -rf csrc/build build dist ./*.egg-info \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_ASCEND_PRETEND_VERSION}" \
       pip install -e . --no-build-isolation --no-deps )

# ----------------------------- runtime deps ---------------------------------
# The editable installs above used --no-deps so pip wouldn't pull a CUDA torch or
# bump numpy. Install the actual runtime deps now, CONSTRAINED so the pinned
# torch/torch-npu/torchvision/torchaudio/triton-ascend/numpy are never changed.
# NOTE: speculators declares numpy>=2.0, but triton-ascend requires numpy 1.26.4 —
# numpy 1.26.4 wins, so speculators' extra deps are installed explicitly (not via
# its metadata) to avoid an unsatisfiable resolve.
echo "[deps] installing runtime dependencies (constrained)..."
CONSTRAINTS="$(mktemp)"
cat > "${CONSTRAINTS}" <<EOF
numpy==${NUMPY_VERSION}
torch==${TORCH_VERSION}
torch-npu==${TORCH_NPU_VERSION}
torchvision==0.25.0
torchaudio==${TORCH_VERSION}
triton-ascend==${TRITON_ASCEND_VERSION}
EOF
# vllm's full runtime dependency tree (empty target => pure-python, no recompile)
( cd "${REPO_ROOT}/vllm" && VLLM_TARGET_DEVICE=empty \
    pip install --retries 15 --timeout 300 -c "${CONSTRAINTS}" --no-build-isolation -e . )
# speculators' extra runtime deps (installed explicitly; its numpy>=2 pin is NOT applied)
pip install --retries 15 --timeout 300 -c "${CONSTRAINTS}" \
    openai "datasets>=4.0.0,<=4.8.4" pydantic-settings protobuf
rm -f "${CONSTRAINTS}"

# ----------------------------- verify ---------------------------------------
echo "[verify]"
pip list --editable | grep -iE "vllm|speculators" || true
python - <<'PY'
import numpy, torch, torch_npu, vllm, vllm_ascend, speculators
print("numpy      ", numpy.__version__)
print("torch      ", torch.__version__)
print("torch_npu  ", torch_npu.__version__)
print("vllm       ", vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
print("speculators", speculators.__file__)
print("ALL OK — no 'Failed to register custom ops' above means DFlash ops are compiled in.")
PY
echo "[vLLM_NPU] done. Activate with:  conda activate ${TARGET}"