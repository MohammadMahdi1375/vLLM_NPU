"""CPU checks for parity policy, controlled settings, and reporting."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from test_server import fixture, load, PACKAGE, HELPER


def comparison_module():
    return load(PACKAGE / "overlay/Evaluator/compare_speculative.py")


@pytest.mark.parametrize("kind", ["dflash", "compiled_prefix_v3", "local_prefix_v2", "dspark"])
@pytest.mark.parametrize("head", ["reference", "validate", "fast"])
def test_matched_profiles(tmp_path, kind, head):
    root, args = fixture(tmp_path, kind)
    args.draft_enforce_eager = "1"
    args.comparison_profile = "matched_reference" if head == "reference" else "matched_fast"
    args.head_execution = head
    base = root / "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"
    base.write_text(base.read_text() + "\n# _dspark_head_runtime\n")
    module = load(root / HELPER)
    _, cmd, env, record = module.command_and_record(args)
    assert record["speculative_config"]["enforce_eager"] is True
    assert "--enforce-eager" not in cmd
    assert record["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert record["head_execution_requested"] == ("argmax" if kind == "dflash" else head)
    assert env["VLLM_ASCEND_DSPARK_HEAD_GRAPH"] == (head if kind == "dspark" and head != "reference" else "off")
    assert env["DFLASH_PREFIX_INFERENCE"] == (head if kind in module.PREFIX_KINDS else "reference")


def test_matched_profile_rejects_accidental_draft_graph(tmp_path):
    root, args = fixture(tmp_path)
    args.comparison_profile = "matched_fast"
    for flag in ("0", "omit"):
        args.draft_enforce_eager = flag
        with pytest.raises(ValueError, match="ALL draft backbones"):
            load(root / HELPER).command_and_record(args)


def report(method="dflash", diagnostic=False):
    return dict(complete=True, manifest_sha256="manifest", comparison_settings={},
                performance_measurement=not diagnostic,
                startup_warmup=dict(complete=True, excluded_from_measurement=True),
                server_manifest=dict(draft_enforce_eager=True, proposals=15, num_speculative_tokens=15),
                cells=[dict(benchmark="gsm8k", temperature=0, concurrency=1,
                            prompt_hashes=["prompt"], output_hashes=["same"], speculative_hashes=[method],
                            summary=dict(requests=1, tokens_per_second=100., acceptance_length_with_bonus_pooled=5.,
                                         tpot_ms_request_mean=10., exclusive_server_interval=True, counters_crosscheck=True))])


def test_different_methods_may_have_different_tau_but_not_outputs():
    m = comparison_module()
    a, b = report(), report("prefix")
    m.check_report(b, "prefix", False)
    m.compare_outputs(a, b, same_method=False)
    with pytest.raises(ValueError, match="acceptance counters"):
        m.compare_outputs(a, b, same_method=True)
    b["cells"][0]["output_hashes"] = ["changed"]
    with pytest.raises(ValueError, match="output_hashes"):
        m.compare_outputs(a, b, same_method=False)


@pytest.mark.parametrize("change", ["incomplete", "contamination", "counter", "diagnostic", "warmup", "nohash"])
def test_reject_invalid_timing_reports(change):
    m, r = comparison_module(), report()
    if change == "incomplete": r["complete"] = False
    if change == "contamination": r["cells"][0]["summary"]["exclusive_server_interval"] = False
    if change == "counter": r["cells"][0]["summary"]["counters_crosscheck"] = False
    if change == "diagnostic": r["performance_measurement"] = False
    if change == "warmup": r["startup_warmup"]["excluded_from_measurement"] = False
    if change == "nohash": r["cells"][0]["speculative_hashes"] = []
    with pytest.raises(ValueError): m.check_report(r, "dflash", False)


def test_repeated_summary_uses_all_samples():
    m = comparison_module()
    reports = {name: [report(name) for _ in range(3)] for name in m.METHODS}
    for index, value in enumerate([100., 110., 120.]):
        reports["prefix_dflash"][index]["cells"][0]["summary"]["tokens_per_second"] = value
    rows = m.summarize(reports)
    prefix = next(r for r in rows if r["method"] == "prefix_dflash")
    assert prefix["tokens_per_second_mean"] == 110
    assert prefix["tokens_per_second_std"] == 10
    assert prefix["relative_tps_vs_dflash"] == 1.1


def test_runtime_graph_evidence_not_requested_flag():
    m = comparison_module()
    assert not m.full_dispatch("FULL_DECODE_ONLY requested", 15)
    assert not m.full_dispatch("| 16 | 16 | 0 | FULL | 0 |", 15)
    assert not m.full_dispatch("| 8 | 8 | 0 | FULL | 2 |", 15)
    assert m.full_dispatch("| 16 | 16 | 0 | FULL | 2 |", 15)


def test_failure_bundle_stays_under_upload_limit(tmp_path):
    for i in range(10):
        (tmp_path / f"server_{i}.log").write_bytes(os.urandom(70 * 1024))
    archive = Path(comparison_module().failure_bundle(tmp_path, {"complete": False, "error": "test failure"}))
    assert archive.stat().st_size < 1_000_000
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == 9
        assert not json.loads(z.read("result.json"))["complete"]


def test_original_dspark_fallback_kept_exactly_and_dummy_guard():
    relative = "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"
    old = (PACKAGE / "baselines" / relative).read_text()
    new = (PACKAGE / "overlay" / relative).read_text()
    a = old.index("                draft_token_ids = self._dspark_draft_buffer[:num_blk]")
    b = old.index("                # Dynamic verify-length path, implemented in DynamicSpecScheduler", a)
    original = old[a:b]
    indented = "".join("    " + line if line.strip() else line for line in original.splitlines(keepends=True))
    assert indented in new
    assert 'runtime is not None and not getattr(self, "_dspark_in_dummy_run", False)' in new
    proposer = (PACKAGE / "overlay/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py").read_text()
    assert "finally:\n            self._dspark_in_dummy_run = False" in proposer
    assert "self.use_cuda_graph = False" in proposer  # No unsupported backbone capture.


def test_evaluator_aggregation_and_sampling_unchanged():
    def functions(folder):
        tree = ast.parse((PACKAGE / folder / "Evaluator/evaluator.py").read_text())
        return {f.name: ast.dump(f, include_attributes=False) for f in tree.body if isinstance(f, ast.FunctionDef)}
    old, new = functions("baselines"), functions("overlay")
    for name in ("aggregate", "sampling", "validate_spec", "run_selector_warmup", "run_request_batch"):
        assert old[name] == new[name]


@pytest.mark.parametrize("fail_graph,policy", [(False, "matched_eager"), (True, "matched_eager"), (False, "backend_default")])
def test_complete_orchestration_with_http_fixture(tmp_path, fail_graph, policy):
    root, args = fixture(tmp_path, "dflash")
    main = root / "vllm/vllm/entrypoints/cli/main.py"
    main.parent.mkdir(parents=True)
    shutil.copy2(PACKAGE / "tests/http_fixture.py", main)
    if fail_graph:
        main.write_text(main.read_text().replace('[42] * (count // 2)',
                        '[43 if "--enforce-eager" not in argv else 42] * (count // 2)'))
    for folder in (main.parent, main.parent.parent, main.parent.parent.parent):
        (folder / "__init__.py").write_text("")
    (args.target / "model.safetensors").write_bytes(b"synthetic target weights")
    drafts = {"dflash": args.draft}
    for name, family, selector in [("prefix-dflash", "dflash_prefix", "compiled_prefix_v3"),
                                   ("dspark", "dspark", "dspark")]:
        dest = root / name
        shutil.copytree(args.draft, dest)
        config = json.loads((dest / "config.json").read_text())
        config.update(speculators_model_type=family, prefix_selector_kind=selector)
        (dest / "config.json").write_text(json.dumps(config))
        drafts[name] = dest
    evaluator = load(root / "Evaluator/evaluator.py")
    tokens = [1, 2, 3]
    manifest = root / "prompts.json"
    manifest.write_text(json.dumps(dict(thinking=False, target_config_sha256=evaluator.file_hash(args.target / "config.json"),
        datasets={"gsm8k": {"results": [dict(prompt_token_ids=tokens, sha256=evaluator.digest(tokens)) for _ in range(2)]}})))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = root / "paper-output"
    command = [sys.executable, str(root / "Evaluator/compare_speculative.py"),
               "--spec-main", str(root), "--target", str(args.target), "--manifest", str(manifest),
               "--output", str(output), "--port", str(port), "--num-prompts", "2", "--max-new-tokens", "32",
               "--backbone-policy", policy]
    for name, path in drafts.items():
        command += ["--" + name, str(path)]
    env = dict(os.environ, ENV_ROOT=sys.prefix, CANN_ENV_SH="/dev/null", ATB_ENV_SH="/dev/null")
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=90)
    data = json.loads((output / "result.json").read_text())
    if fail_graph:
        assert result.returncode == 1, result.stdout + result.stderr
        assert not data["complete"] and "output_hashes" in data["error"]
        assert not list(output.glob("repeat_*"))
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert data["complete"] and data["validation_passed"]
        assert not data["hardware_validation_pending"]  # fixture evidence only
        assert len(data["timing_order"]) == 3
        assert [row[0] for row in data["timing_order"]] == list(comparison_module().METHODS)
        rows = json.loads((output / "comparison.json").read_text())
        assert len(rows) == 3 and all(r["repeats"] == 3 for r in rows)
        assert len(list(output.glob("repeat_*/*/eval/run_info.json"))) == 9
    # Every server started by the runner has been stopped and the port released.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
