"""CPU command/HTTP tests, using synthetic counters; not NPU validation."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import build_opener, ProxyHandler

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
HELPER = "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"


def load(path):
    spec = importlib.util.spec_from_file_location("server_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, kind="dspark", block=16, sample_anchor=False):
    root = tmp_path / "project with spaces"
    shutil.copytree(PACKAGE / "overlay", root)
    source = root / HELPER
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.endswith(".py") and value.startswith(("vllm/", "vllm-ascend/", "speculators/")):
                p = root / value
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("# self.runner._use_aclgraph() and not self.speculative_config.enforce_eager\n")
    for name in ("inference_v2.py", "compiled_inference.py"):
        p = root / "speculators/src/speculators/models/dflash_prefix" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# CPU test only\n")
    target, draft = root / "target", root / 'draft "quoted"'
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(json.dumps(dict(model_type="qwen3", hidden_size=2560, vocab_size=151936)))
    (target / "generation_config.json").write_text('{"eos_token_id":151645}')
    family = kind if kind in ("dflash", "dspark") else "dflash_prefix"
    (draft / "config.json").write_text(json.dumps(dict(speculators_model_type=family, block_size=block,
        prefix_selector_kind=kind, prefix_top_k=16, sample_from_anchor=sample_anchor,
        transformer_layer_config=dict(hidden_size=2560, vocab_size=151936))))
    (draft / "model.safetensors").write_bytes(b"fixture only; not weights")
    args = SimpleNamespace(spec_main=root, draft=draft, target=target, npu="6", port=8213,
        target_execution="graph", mode="fast", served_model_name="test-model", max_model_len=32768,
        max_batched_tokens=32768, memory_utilization=.96, graph_metrics=0, check_only=1,
        speculative_method="auto", num_speculative_tokens="auto", draft_enforce_eager="omit")
    return root, args


@pytest.mark.parametrize("kind", ["dflash", "local_prefix_v2", "compiled_prefix_v3", "dspark"])
@pytest.mark.parametrize("block", [8, 16])
def test_method_routing_graph_shape_and_no_eager_override(tmp_path, kind, block):
    root, args = fixture(tmp_path, kind, block)
    module = load(root / HELPER)
    before = (args.draft / "config.json").read_bytes()
    _, command, env, record = module.command_and_record(args)
    spec = record["speculative_config"]
    assert spec["method"] == ("dspark" if kind == "dspark" else "dflash")
    assert "enforce_eager" not in spec and "--enforce-eager" not in command
    assert spec["num_speculative_tokens"] == block - 1
    assert record["compilation_config"]["cudagraph_capture_sizes"] == [block]
    assert record["compilation_config"]["mode"] == 3
    assert record["draft_execution_requested"] == "backend_default"
    prefix = kind in module.PREFIX_KINDS
    assert env["DFLASH_PREFIX_NPU_GRAPH"] == ("1" if prefix else "0")
    assert (args.draft / "config.json").read_bytes() == before
    if kind == "dspark":
        assert not any("prefix_proposer" in name for name in record["source_sha256"])


@pytest.mark.parametrize("override,expected", [("omit", None), ("1", True), ("0", False)])
def test_explicit_eager_control(tmp_path, override, expected):
    root, args = fixture(tmp_path)
    args.draft_enforce_eager = override
    module = load(root / HELPER)
    _, graph, _, record = module.command_and_record(args)
    assert record["speculative_config"].get("enforce_eager") is expected
    assert "--enforce-eager" not in graph
    args.target_execution = "eager"
    _, eager, _, other = module.command_and_record(args)
    assert "--enforce-eager" in eager
    assert record["speculative_config"] == other["speculative_config"]
    assert other["compilation_config"]["cudagraph_mode"] == "NONE"


@pytest.mark.parametrize("kind", ["dflash", "local_prefix_v2", "compiled_prefix_v3"])
def test_previous_measured_settings_can_be_reproduced(tmp_path, kind):
    root, args = fixture(tmp_path, kind)
    args.draft_enforce_eager = "1"
    _, old_command, old_env, _ = load(PACKAGE / "baselines" / HELPER).command_and_record(args)
    _, command, env, _ = load(root / HELPER).command_and_record(args)
    assert command == old_command
    assert env == dict(old_env, VLLM_ASCEND_DSPARK_HEAD_GRAPH="off")


def test_dspark_anchor_semantics_and_explicit_shorter_proposals(tmp_path):
    root, args = fixture(tmp_path, block=16, sample_anchor=True)
    module = load(root / HELPER)
    with pytest.raises(ValueError, match="does not silently truncate"):
        module.command_and_record(args)
    args.num_speculative_tokens = "15"
    _, _, _, record = module.command_and_record(args)
    assert record["proposals"] == 15 and record["verification_tokens"] == 16
    args.num_speculative_tokens = "7"
    _, _, _, record = module.command_and_record(args)
    assert record["block_size"] == 16 and record["compilation_config"]["cudagraph_capture_sizes"] == [8]


@pytest.mark.parametrize("kind,method", [("dspark", "dflash"), ("dflash", "dspark"), ("compiled_prefix_v3", "dspark")])
def test_mismatched_methods_fail(tmp_path, kind, method):
    root, args = fixture(tmp_path, kind)
    args.speculative_method = method
    with pytest.raises(ValueError, match="requires method"):
        load(root / HELPER).command_and_record(args)


def test_prefix_count_is_not_silently_reduced(tmp_path):
    root, args = fixture(tmp_path, "compiled_prefix_v3")
    args.num_speculative_tokens = "7"
    with pytest.raises(ValueError, match="block_size - 1"):
        load(root / HELPER).command_and_record(args)


def test_inherited_prefix_and_profiling_flags_do_not_affect_dspark(tmp_path):
    root, _ = fixture(tmp_path)
    env = load(root / HELPER).serving_environment(dict(DFLASH_PREFIX_TOKEN_CACHE="1",
        DFLASH_PREFIX_FUSION="local_unpaired", DFLASH_PREFIX_SELECTOR_PROFILE_STEPS="8",
        TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1", KEEP="yes"), "dspark", "fast")
    assert {k:v for k,v in env.items() if k.startswith("DFLASH_PREFIX_")} == {
        "DFLASH_PREFIX_INFERENCE":"reference", "DFLASH_PREFIX_NPU_GRAPH":"0"}
    assert "TORCH_COMPILE_DISABLE" not in env and env["KEEP"] == "yes"


def test_evaluator_metric_and_sampling_functions_are_unchanged():
    before = ast.parse((PACKAGE / "baselines/Evaluator/evaluator.py").read_text())
    after = ast.parse((PACKAGE / "overlay/Evaluator/evaluator.py").read_text())
    for name in ("aggregate", "sampling", "validate_spec", "prepare_manifest", "select_prompts", "Client", "run_selector_warmup", "confirm_prefix_runtime"):
        old = next(n for n in before.body if getattr(n, "name", None) == name)
        new = next(n for n in after.body if getattr(n, "name", None) == name)
        assert ast.dump(old) == ast.dump(new), name


def test_common_runtime_settings_match_across_methods(tmp_path):
    shared = None
    for kind in ("dflash", "compiled_prefix_v3", "dspark"):
        case = tmp_path / kind
        case.mkdir()
        root, args = fixture(case, kind)
        _, command, env, record = load(root / HELPER).command_and_record(args)
        # Only checkpoint/method/model-name selectors may differ. Target paths
        # are distinct fixture copies; all other CLI options must be identical.
        normalized = list(command)
        normalized[normalized.index("serve") + 1] = "TARGET"
        spec_index = normalized.index("--speculative-config") + 1
        spec = json.loads(normalized[spec_index])
        spec.pop("model")
        spec.pop("method")
        normalized[spec_index] = json.dumps(spec, sort_keys=True)
        common_env = {k:v for k,v in env.items() if not k.startswith("DFLASH_PREFIX_")}
        common = (normalized, common_env, record["compilation_config"])
        if shared is None:
            shared = common
        assert common == shared


@pytest.mark.parametrize("kind,entry", [("dflash", "run_server_dflash.sh"),
    ("compiled_prefix_v3", "run_server_prefix_dflash.sh"), ("dspark", "run_server_dspark.sh")])
def test_shell_check_only_does_not_load_cann_or_start_server(tmp_path, kind, entry):
    root, args = fixture(tmp_path, kind)
    env = dict(os.environ, SPEC_MAIN=str(root), ENV_ROOT=sys.prefix, PYTHON_BIN=sys.executable,
        TARGET_MODEL=str(args.target), DRAFT_MODEL=str(args.draft),
        RUN_DIR=str(root / "check"), CHECK_ONLY="1", CANN_ROOT="/does/not/exist")
    for name in ("SPECULATIVE_METHOD", "NPU", "PORT", "DRAFT_ENFORCE_EAGER", "NUM_SPECULATIVE_TOKENS"):
        env.pop(name, None)
    result = subprocess.run(["bash", str(root / "Evaluator" / entry)], env=env,
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "check/launch.json").read_text())
    assert record["npu"] == "6" and record["port"] == 8213
    assert record["proposals"] == 15
    assert record["speculative_method"] == ("dspark" if kind == "dspark" else "dflash")
    assert not (root / "output/evaluator_servers").exists()
    assert not (root / "check/server.log").exists()


@pytest.mark.parametrize("kind", ["dspark", "dflash", "compiled_prefix_v3", "local_prefix_v2"])
def test_launcher_and_evaluator_over_real_http(tmp_path, kind):
    root, args = fixture(tmp_path, kind)
    main = root / "vllm/vllm/entrypoints/cli/main.py"
    main.parent.mkdir(parents=True)
    shutil.copy2(PACKAGE / "tests/http_fixture.py", main)
    for folder in (main.parent, main.parent.parent, main.parent.parent.parent):
        (folder / "__init__.py").write_text("")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, SPEC_MAIN=str(root), ENV_ROOT=sys.prefix, PYTHON_BIN=sys.executable,
        TARGET_MODEL=str(args.target), DRAFT_MODEL=str(args.draft), NPU="6", PORT=str(port),
        RUN_DIR=str(root / "launch"), CHECK_ONLY="0", CANN_ENV_SH="/dev/null", ATB_ENV_SH="/dev/null",
        SPECULATIVE_METHOD="dspark" if kind == "dspark" else "dflash", NUM_SPECULATIVE_TOKENS="15",
        DRAFT_ENFORCE_EAGER="omit", COMPARISON_PROFILE="custom", TARGET_EXECUTION="graph", MODE="fast", GRAPH_METRICS="0",
        SERVED_MODEL_NAME="test-model")
    entry = ("run_server_dspark.sh" if kind == "dspark" else "run_server_dflash.sh"
             if kind == "dflash" else "run_server_prefix_dflash.sh")
    log = (root / "fixture_launcher.log").open("w+")
    server = subprocess.Popen(["bash", str(root / "Evaluator" / entry)], env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    opener = build_opener(ProxyHandler({}))
    try:
        deadline = time.monotonic() + 10
        while True:
            if server.poll() is not None:
                log.seek(0)
                pytest.fail(log.read())
            try:
                with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.fail("Fixture startup timeout")
                time.sleep(.02)
        module = load(root / "Evaluator/evaluator.py")
        tokens = [1, 2, 3]
        manifest = root / "prompts.json"
        manifest.write_text(json.dumps(dict(thinking=False, target_config_sha256=module.file_hash(args.target / "config.json"),
            datasets={"gsm8k":{"results":[dict(prompt_token_ids=tokens, sha256=module.digest(tokens)) for _ in range(2)]}})))
        output = root / "results"
        command = [sys.executable, str(root / "Evaluator/evaluator.py"), "--base-url", f"http://127.0.0.1:{port}",
            "--model", "test-model", "--target", str(args.target), "--manifest", str(manifest),
            "--datasets", "gsm8k", "--num-prompts", "2", "--temperatures", "0", "--concurrencies", "1",
            "--max-new-tokens", "32", "--no-thinking", "--selector-warmup-tokens", "512",
            "--selector-warmup-steps", "32", "--output", str(output)]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        record = json.loads((output / "run_info.json").read_text())
        assert record["complete"] and record["server_manifest"]["automatically_discovered"]
        assert record["prefix_runtime_confirmation"]["verified"] == (kind not in ("dflash", "dspark"))
        if kind == "dspark":
            assert "DSpark: no prefix selector checks apply" in result.stdout
            assert record["server_manifest"]["variant"] == "dspark"
        assert record["startup_warmup"]["verification_rounds"] == 64
        summary = record["cells"][0]["summary"]
        assert summary["requests"] == 2 and summary["output_tokens"] == 64
        assert summary["verification_rounds"] == 8 and summary["accepted_draft_tokens"] == 56
        assert "enforce_eager" not in record["server_manifest"]["speculative_config"]
        module.check_server_process(record["server_manifest"])
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=5)
        log.close()
