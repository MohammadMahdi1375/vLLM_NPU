#!/usr/bin/env python3
"""Matched Qwen3/Ascend experiment: preflight, parity, then repeated benchmarks.

This orchestrates the normal run_server_*.sh and evaluator.py entrypoints.
It changes no checkpoints, training configuration, or model arithmetic.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


METHODS = ("dflash", "prefix_dflash", "dspark")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_helper(root):
    path = root / "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"
    spec = importlib.util.spec_from_file_location("paper_command_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights(directory):
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights in {directory}")
    print(f"Hashing checkpoint weights once for provenance: {directory}", flush=True)
    return {p.name: digest(p) for p in files}


def preflight(args):
    helper = load_helper(args.spec_main)
    records = {}
    for method in METHODS:
        local = argparse.Namespace(**vars(args))
        local.draft = getattr(args, method)
        local.speculative_method = "dspark" if method == "dspark" else "dflash"
        local.target_execution = "graph"
        local.draft_enforce_eager = "1" if args.backbone_policy == "matched_eager" else "omit"
        local.comparison_profile = "matched_fast" if args.backbone_policy == "matched_eager" else "backend_default"
        local.head_execution = local.mode = "fast"
        local.graph_metrics, local.check_only = 0, 1
        local.served_model_name = "paper-qwen3"
        _, _, _, record = helper.command_and_record(local)
        expected = {"dflash": {"dflash"}, "prefix_dflash": helper.PREFIX_KINDS, "dspark": {"dspark"}}[method]
        if record["kind"] not in expected:
            raise ValueError(f"{method} points to the wrong checkpoint family: {record['kind']}")
        records[method] = record
    common = ("target", "target_config_sha256", "proposals", "verification_tokens",
              "compilation_config", "additional_config", "seed", "kv_block_size", "npu",
              "max_model_len", "max_batched_tokens", "memory_utilization", "draft_enforce_eager")
    for field in common:
        if any(r[field] != records["dflash"][field] for r in records.values()):
            raise ValueError(f"Shared serving setting differs: {field}")
    # Freeze evaluated inputs before starting any expensive process.
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("thinking") is not False:
        raise ValueError("Use the frozen no-thinking prompt manifest")
    if manifest.get("target_config_sha256") != records["dflash"]["target_config_sha256"]:
        raise ValueError("Prompt manifest target config differs")
    for dataset in args.datasets:
        if dataset not in manifest.get("datasets", {}):
            raise ValueError(f"Dataset absent from frozen manifest: {dataset}")
    common_sources = {}
    for record in records.values():
        common_sources.update(record["source_sha256"])
    for path in ("Evaluator/evaluator.py", "Evaluator/compare_speculative.py",
                 "Evaluator/run_server_dflash.sh", "Evaluator/run_server_prefix_dflash.sh",
                 "Evaluator/run_server_dspark.sh",
                 "vllm-ascend/vllm_ascend/envs.py",
                 "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"):
        common_sources[path] = digest(args.spec_main / path)
    # Include all selector implementations, including optional kernels used by
    # local_prefix_v2; the launcher alone does not hash every imported module.
    for p in (args.spec_main / "speculators/src/speculators/models/dflash_prefix").glob("*.py"):
        common_sources[str(p.relative_to(args.spec_main))] = digest(p)
    versions = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers", "requests"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unregistered"
    checkpoint_records = {}
    for method in METHODS:
        config = json.loads((getattr(args, method) / "config.json").read_text())
        checkpoint_records[method] = dict(
            path=str(getattr(args, method)), config_sha256=digest(getattr(args, method) / "config.json"),
            weights_sha256=weights(getattr(args, method)), config=config)
    return dict(format="speculative_paper_comparison_v1", target=str(args.target),
                target_config_sha256=digest(args.target / "config.json"), target_weights=weights(args.target),
                checkpoints=checkpoint_records, sources=common_sources, versions=versions,
                host=platform.node(), python=sys.version, npu=args.npu, backbone_policy=args.backbone_policy,
                manifest_sha256=digest(args.manifest), datasets=args.datasets,
                num_prompts=args.num_prompts, max_new_tokens=args.max_new_tokens,
                serving={key: records["dflash"][key] for key in common},
                records=records, training_records=(json.loads(args.training_records.read_text())
                                                  if args.training_records else None))


def stop_owned(process):
    # Popen starts a new session. Never use pkill or attach to an existing server.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        process.poll()
        alive = False
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                alive |= int(fields[2]) == process.pid and fields[0] != "Z"
            except (OSError, ValueError, IndexError):
                continue
        if not alive:
            return
        time.sleep(.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def check_report(report, method, diagnostic):
    if not report.get("complete") or not report.get("cells"):
        raise ValueError(f"Incomplete evaluator run for {method}")
    if not report.get("startup_warmup", {}).get("complete"):
        raise ValueError("Startup warmup did not finish")
    if not report["startup_warmup"].get("excluded_from_measurement"):
        raise ValueError("Warmup exclusion missing")
    launch = report.get("server_manifest") or {}
    profile = launch.get("comparison_profile", "matched_fast")
    eager = launch.get("draft_enforce_eager")
    if (profile in {"matched_fast", "matched_reference"} and eager is not True
            or profile == "backend_default" and eager is not None
            or profile not in {"matched_fast", "matched_reference", "backend_default"}
            or launch.get("proposals") != launch.get("num_speculative_tokens")):
        raise ValueError("Declared draft policy missing from launch record")
    if report.get("performance_measurement") != (not diagnostic):
        raise ValueError("Diagnostic/timed run classification mismatch")
    if launch.get("dspark_head_graph", "off") != "off" and not report.get("dspark_runtime_confirmation", {}).get("verified"):
        raise ValueError("DSpark head graph was requested but not verified")
    if launch.get("prefix_npu_graph") and not report.get("prefix_runtime_confirmation", {}).get("verified"):
        raise ValueError("Prefix graph was requested but not verified")
    for cell in report["cells"]:
        if cell.get("temperature") != 0 or cell.get("concurrency") != 1:
            raise ValueError("This comparison requires greedy concurrency=1")
        if not cell.get("output_hashes") or not cell.get("speculative_hashes"):
            raise ValueError("Per-request output/counter fingerprints missing; update evaluator.py")
        summary = cell["summary"]
        if len(cell["output_hashes"]) != summary["requests"]:
            raise ValueError("Output fingerprint count differs from completed requests")
        if not summary.get("exclusive_server_interval", False) or not summary.get("counters_crosscheck", False):
            raise ValueError("Request contamination or acceptance-counter disagreement")


def compare_outputs(reference, current, *, same_method):
    if reference["manifest_sha256"] != current["manifest_sha256"]:
        raise ValueError("Prompt manifests differ")
    settings = ("target_config_sha256", "tokenizer_sha256", "generation_seed", "prompt_seed",
                "max_new_tokens", "thinking", "concurrencies", "temperatures", "resolved_sampling")
    if any(reference["comparison_settings"].get(k) != current["comparison_settings"].get(k) for k in settings):
        raise ValueError("Evaluation sampling settings differ")
    if len(reference["cells"]) != len(current["cells"]):
        raise ValueError("Dataset count differs")
    for a, b in zip(reference["cells"], current["cells"]):
        for key in ("benchmark", "temperature", "concurrency", "prompt_hashes", "output_hashes"):
            if a[key] != b[key]:
                raise ValueError(f"{b['benchmark']}: {key} differ; inspect before comparing performance")
        # Acceptance is EXPECTED to differ between methods, but must match each
        # method's own eager/reference run and remain stable across repeats.
        if same_method and a["speculative_hashes"] != b["speculative_hashes"]:
            raise ValueError(f"{b['benchmark']}: acceptance counters changed for the same checkpoint")


def full_dispatch(text, k):
    return re.search(rf"\|\s*{k+1}\s*\|\s*{k+1}\s*\|\s*0\s*\|\s*FULL\s*\|\s*[1-9][0-9]*\s*\|", text) is not None


def run_one(args, output, method, phase):
    reference = phase == "reference"
    diagnostic = phase != "timed"
    for path, expected in args.identity["sources"].items():
        if digest(args.spec_main / path) != expected:
            raise ValueError(f"Source changed after preflight: {path}; start a new comparison")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    server_dir = output / "server"
    env = dict(os.environ, SPEC_MAIN=str(args.spec_main), TARGET_MODEL=str(args.target),
               DRAFT_MODEL=str(getattr(args, method)), NPU=args.npu, PORT=str(args.port),
               PYTHON_BIN=sys.executable, SERVED_MODEL_NAME="paper-qwen3",
               SPECULATIVE_METHOD="dspark" if method == "dspark" else "dflash",
               COMPARISON_PROFILE=("matched_reference" if reference else
                                   "matched_fast" if args.backbone_policy == "matched_eager" else "backend_default"),
               TARGET_EXECUTION="eager" if reference else "graph",
               DRAFT_ENFORCE_EAGER="1" if reference or args.backbone_policy == "matched_eager" else "omit",
               HEAD_EXECUTION="reference" if reference else "validate" if diagnostic else "fast",
               GRAPH_METRICS="1" if diagnostic else "0", CHECK_ONLY="0", RUN_DIR=str(server_dir),
               NUM_SPECULATIVE_TOKENS=args.num_speculative_tokens, SEED=str(args.seed),
               MAX_MODEL_LEN=str(args.max_model_len), MAX_BATCHED_TOKENS=str(args.max_batched_tokens),
               MEMORY_UTILIZATION=str(args.memory_utilization), KV_BLOCK_SIZE=str(args.kv_block_size))
    command = ["bash", str(args.spec_main / "Evaluator" / f"run_server_{method}.sh")]
    process = None
    print(f"Starting {method} / {phase}; output: {output}", flush=True)
    try:
        with (output / "launcher.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline, announced = time.monotonic() + 1800, time.monotonic()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; see {output / 'launcher.log'}")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/v1/models", timeout=5) as response:
                        ids = [m["id"] for m in json.load(response)["data"]]
                    if ids != ["paper-qwen3"]:
                        raise ValueError("Unexpected model on comparison port")
                    break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 30 minutes")
                if time.monotonic() - announced >= 30:
                    print(f"Waiting for {method}; log: {output / 'launcher.log'}", flush=True)
                    announced = time.monotonic()
                time.sleep(1)
            evaluation = [sys.executable, str(args.spec_main / "Evaluator/evaluator.py"),
                          "--base-url", f"http://127.0.0.1:{args.port}", "--model", "paper-qwen3",
                          "--method", method, "--target", str(args.target), "--manifest", str(args.manifest),
                          "--datasets", *args.datasets, "--num-prompts", str(args.num_prompts),
                          "--temperatures", "0", "--concurrencies", "1", "--top-p", "1", "--top-k", "-1",
                          "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed),
                          "--prompt-seed", str(args.seed), "--warmup-requests", "1",
                          "--selector-warmup-tokens", "512", "--selector-warmup-steps", "32", "--no-thinking",
                          "--server-manifest", str(server_dir / "launch.json"), "--output", str(output / "eval")]
            write(output / "eval_command.json", evaluation)
            subprocess.run(evaluation, env=env, check=True)
            report = json.loads((output / "eval/run_info.json").read_text())
            check_report(report, method, diagnostic)
            if phase == "validate":
                text = (server_dir / "server.log").read_text(errors="replace")
                if not full_dispatch(text, report["server_manifest"]["proposals"]):
                    raise ValueError("Actual FULL target dispatch not observed in diagnostic log")
            return report
    finally:
        if process is not None:
            stop_owned(process)


def summarize(reports):
    rows = []
    for cell_index, first in enumerate(reports["dflash"][0]["cells"]):
        base = [r["cells"][cell_index]["summary"]["tokens_per_second"] for r in reports["dflash"]]
        for method in METHODS:
            cells = [r["cells"][cell_index]["summary"] for r in reports[method]]
            tps = [c["tokens_per_second"] for c in cells]
            tau = [c["acceptance_length_with_bonus_pooled"] for c in cells]
            rows.append(dict(benchmark=first["benchmark"], method=method, repeats=len(cells),
                             requests_per_repeat=cells[0]["requests"],
                             tokens_per_second_mean=statistics.mean(tps),
                             tokens_per_second_std=statistics.stdev(tps),
                             tau_mean=statistics.mean(tau), tau_std=statistics.stdev(tau),
                             relative_tps_vs_dflash=statistics.mean(tps)/statistics.mean(base),
                             paired_speed_ratio_mean=statistics.mean(a/b for a, b in zip(tps, base)),
                             tpot_ms_mean=statistics.mean(c["tpot_ms_request_mean"] for c in cells)))
    return rows


def failure_bundle(output, result):
    """Small existing-log bundle for the user's 1 MB upload limit; no rerun."""
    path = output / "failure_small.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", json.dumps(result, indent=2))
        logs = sorted(output.rglob("*.log"), key=lambda p: p.stat().st_mtime)[-8:]
        for log in logs:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 48 * 1024))
                archive.writestr(str(log.relative_to(output)) + ".tail.txt", stream.read())
        identity = output / "identity.json"
        if identity.is_file():
            data = json.loads(identity.read_text())
            archive.writestr("identity_summary.json", json.dumps(
                {k: data.get(k) for k in ("format", "npu", "host", "versions", "backbone_policy",
                                          "sources", "datasets", "manifest_sha256", "serving")}, indent=2))
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-main", type=Path, default=Path(__file__).resolve().parent.parent)
    for name in ("target", "dflash", "prefix-dflash", "dspark", "manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--backbone-policy", choices=("matched_eager", "backend_default"), default="matched_eager")
    parser.add_argument("--phase", choices=("all", "validate", "benchmark"), default="all")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--training-records", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--npu", default="6")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--num-speculative-tokens", default="15")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"])
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-batched-tokens", type=int, default=32768)
    parser.add_argument("--memory-utilization", type=float, default=.96)
    parser.add_argument("--kv-block-size", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 3 or args.num_prompts < 1 or args.max_new_tokens < 1:
        parser.error("Use at least 3 timing repeats and positive prompt/token counts")
    for name in ("spec_main", "target", "dflash", "prefix_dflash", "dspark", "manifest"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, phase=args.phase, hardware_validation_pending=True,
                  training_budget_independently_verified=False,
                  scope="Matched single-request greedy implementation comparison; not a proof of algorithmic superiority")
    try:
        identity = preflight(args)
        args.identity = identity
        write(args.output / "identity.json", identity)
        if args.check_only:
            result.update(complete=True, check_only=True)
            print("Preflight passed. No NPU process started; use a fresh output directory for the run.")
            return 0
        # Record hardware/software evidence without assuming a particular SKU.
        try:
            info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
            (args.output / "npu_smi.txt").write_text(info.stdout + info.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            (args.output / "npu_smi.txt").write_text(str(error))
        validation = {}
        if args.phase == "benchmark":
            if not args.validation_dir:
                raise ValueError("--phase benchmark requires --validation-dir from a successful validation")
            validation_dir = args.validation_dir.resolve(strict=True)
            previous = json.loads((validation_dir / "result.json").read_text())
            if not previous.get("validation_passed"):
                raise ValueError("Validation did not pass")
            if json.loads((validation_dir / "identity.json").read_text()) != identity:
                raise ValueError("Weights/source/settings/device changed since validation; revalidate")
            for method in METHODS:
                reference = json.loads((validation_dir / method / "reference/eval/run_info.json").read_text())
                validation[method] = json.loads((validation_dir / method / "validate/eval/run_info.json").read_text())
                check_report(reference, method, True)
                check_report(validation[method], method, True)
                compare_outputs(reference, validation[method], same_method=True)
                text = (validation_dir / method / "validate/server/server.log").read_text(errors="replace")
                if not full_dispatch(text, validation[method]["server_manifest"]["proposals"]):
                    raise ValueError("Saved target dispatch evidence missing")
        else:
            for method in METHODS:
                reference = run_one(args, args.output / method / "reference", method, "reference")
                validation[method] = run_one(args, args.output / method / "validate", method, "validate")
                compare_outputs(reference, validation[method], same_method=True)
        for method in METHODS[1:]:
            compare_outputs(validation["dflash"], validation[method], same_method=False)
        result.update(validation_passed=True, hardware_validation_pending=False)
        if args.phase == "validate":
            result["complete"] = True
            return 0
        reports = {m: [] for m in METHODS}
        result["timing_order"] = []
        for repeat in range(args.repeats):
            order = METHODS[repeat % 3:] + METHODS[:repeat % 3]
            result["timing_order"].append(list(order))
            for method in order:
                report = run_one(args, args.output / f"repeat_{repeat+1}" / method, method, "timed")
                compare_outputs(validation[method], report, same_method=True)
                reports[method].append(report)
        rows = summarize(reports)
        for row in rows:
            row["backbone_policy"] = args.backbone_policy
        with (args.output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write(args.output / "comparison.json", rows)
        result.update(complete=True, timing_repeats=args.repeats)
        print(f"Completed matched comparison: {args.output / 'comparison.csv'}")
        print("Report training budget and hardware separately; this does not certify those are matched.")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
        try:
            result["failure_bundle"] = failure_bundle(args.output, result)
            print(f"Small diagnostic ZIP: {result['failure_bundle']}", file=sys.stderr)
        except Exception as collector_error:
            result["collection_error"] = str(collector_error)
        return 1
    finally:
        write(args.output / "result.json", result)


if __name__ == "__main__":
    raise SystemExit(main())
