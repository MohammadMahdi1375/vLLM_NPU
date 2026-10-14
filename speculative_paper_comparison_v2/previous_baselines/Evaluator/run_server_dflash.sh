#!/usr/bin/env bash
# Qwen3 + DFlash / DFlash-Prefix on Ascend. Edit defaults or override via env.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"

# ============================================================
# Paths and checkpoint
# ============================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-/home/n84449292/m84379596/conda/vllm-ascend-0202}}"
export PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-${DRAFT:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}}"

# To compare vanilla, change only DRAFT_MODEL, e.g.:
# export DRAFT_MODEL="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4"
# Block 8 and block 16 are read from config.json; all 7 or 15 proposals are kept.
# Set this only if you want an explicit consistency check against that config.
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-auto}"

# ============================================================
# Server settings
# ============================================================
export NPU="${NPU:-6}"
export PORT="${PORT:-8213}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-dflash}"
export SEED="${SEED:-42}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.96}"
export KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-128}" # KV-cache pages, NOT draft block size.

# The measured graph path uses one NPU, BF16, TP=1, DP=1, max-num-seqs=1.
# Target decode graph + separate selector graph; draft backbone remains eager.
export TARGET_EXECUTION="${TARGET_EXECUTION:-graph}" # graph | eager
export MODE="${MODE:-fast}"                         # fast | validate | reference
export GRAPH_METRICS="${GRAPH_METRICS:-0}"           # 1 is diagnostic, not a speed run.
export CHECK_ONLY="${CHECK_ONLY:-0}"                 # 1 checks config without CANN/NPU use.
export RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/evaluator_server_$(date +%Y%m%d_%H%M%S)_$$}"

# ============================================================
# Environment
# ============================================================
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN. Activate Conda or set ENV_ROOT." >&2
    exit 2
fi
if [[ "$CHECK_ONLY" != 0 && "$CHECK_ONLY" != 1 ]]; then
    echo 'CHECK_ONLY must be 0 or 1.' >&2
    exit 2
fi
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
if [[ "$CHECK_ONLY" == 0 ]]; then
    source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
    source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
fi
# Keep the installed CANN platform selection; do not force the old A3 SOC on A2.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ============================================================
# Start vLLM using the same command builder as the measured fast path.
# No separate serving launcher is required. The command is saved in RUN_DIR.
# ============================================================
exec "$PYTHON_BIN" - <<'PY_SERVER'
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from types import SimpleNamespace


def main():
    env = os.environ
    root = Path(env["SPEC_MAIN"]).expanduser().resolve(strict=True)
    helper = root / "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"
    if not helper.is_file():
        raise ValueError(f"The installed fast target-graph helper is missing: {helper}")
    spec = importlib.util.spec_from_file_location("dflash_target_graph_server", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        spec_main=root, target=Path(env["TARGET_MODEL"]), draft=Path(env["DRAFT_MODEL"]),
        npu=env["NPU"], port=int(env["PORT"]), target_execution=env["TARGET_EXECUTION"],
        mode=env["MODE"], served_model_name=env["SERVED_MODEL_NAME"],
        max_model_len=int(env["MAX_MODEL_LEN"]), max_batched_tokens=int(env["MAX_BATCHED_TOKENS"]),
        memory_utilization=float(env["MEMORY_UTILIZATION"]),
        graph_metrics=int(env["GRAPH_METRICS"]), check_only=int(env["CHECK_ONLY"]),
    )
    if args.graph_metrics not in (0, 1):
        raise ValueError("GRAPH_METRICS must be 0 or 1")
    root, command, server_env, record = module.command_and_record(args)
    requested = env["NUM_SPECULATIVE_TOKENS"]
    if requested != "auto" and int(requested) != record["proposals"]:
        raise ValueError("NUM_SPECULATIVE_TOKENS must equal the checkpoint's block_size - 1")
    seed, kv_block = int(env["SEED"]), int(env["KV_BLOCK_SIZE"])
    if not 0 <= seed < 2**32 or kv_block < 1:
        raise ValueError("SEED must be in [0, 2**32); KV_BLOCK_SIZE must be positive")
    command[command.index("--seed") + 1] = str(seed)
    command[command.index("--block-size") + 1] = str(kv_block)
    output = Path(env["RUN_DIR"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    log = output / "server.log"
    prefix = record["kind"] != "dflash"
    selector_env = record["selector_environment"]
    mode = selector_env["DFLASH_PREFIX_INFERENCE"]
    # Use the PID visible in this procfs mount (also works in a PID namespace).
    process_stat = Path("/proc/self/stat").read_text()
    proc_pid = int(process_stat.split(" ", 1)[0])
    fields = process_stat.rsplit(")", 1)[1].split()
    record.update(
        format="dflash_evaluator_integration_v2", command=command,
        base_url=f"http://127.0.0.1:{args.port}", served_model=args.served_model_name,
        variant="prefix" if prefix else "vanilla", prefix_mode=mode,
        prefix_npu_graph=selector_env["DFLASH_PREFIX_NPU_GRAPH"] == "1",
        prefix_token_cache=False, prefix_validate_steps=32 if prefix and mode != "reference" else 0,
        num_speculative_tokens=record["proposals"], server_log=str(log),
        launcher_pid=proc_pid, launcher_start_ticks=fields[19],
        suggested_method=f"{record['kind']}_bs{record['block_size']}_{args.target_execution}_{mode}",
        device=f"npu:{args.npu}", tp_size=1, dp_size=1, dtype="bfloat16", seed=seed,
        max_num_seqs=1, kv_block_size=kv_block, prefix_caching=False,
        enforce_eager=args.target_execution == "eager", model_runner="ascend_v1",
        launcher_sha256=module.digest(root / "Evaluator/run_server_dflash.sh"),
        helper_sha256=module.digest(helper),
    )
    launch = output / "launch.json"
    launch.write_text(json.dumps(record, indent=2) + "\n")
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    if not args.check_only:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError as error:
                raise ValueError(f"Port {args.port} is unavailable; choose a free PORT: {error}") from error
        # Keep terminal output and an exact log for automatic warmup confirmation.
        tee = subprocess.Popen(["tee", str(log)], stdin=subprocess.PIPE)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        tee.stdin.close()
        registry = root / "output/evaluator_servers"
        registry.mkdir(parents=True, exist_ok=True)
        temporary = registry / f"port_{args.port}.{os.getpid()}.tmp"
        temporary.symlink_to(launch)
        os.replace(temporary, registry / f"port_{args.port}.json")
    print(f"TARGET_GRAPH_REQUEST mode={args.target_execution} kind={record['kind']} "
          f"block={record['block_size']} proposals={record['proposals']} "
          "draft_backbone=eager full_verification=True", flush=True)
    print(f"Target: {record['target']}\nDraft:  {record['draft']}", flush=True)
    print(f"Server log: {log}\nLaunch record: {launch}", flush=True)
    if args.check_only:
        print("Configuration passed. No server started. Use a fresh RUN_DIR for serving.")
        return
    os.chdir(root / "speculators")
    os.execve(sys.executable, command, server_env)


try:
    main()
except (OSError, ValueError, KeyError) as error:
    raise SystemExit(str(error)) from error
PY_SERVER
