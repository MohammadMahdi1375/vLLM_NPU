"""Temporary filesystem tests; no real repositories or checkpoints are changed."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer_under_test", PACKAGE / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def setup_root(tmp_path):
    root = tmp_path / "project with spaces"
    for relative in installer.FILES:
        source = PACKAGE / "baselines" / relative
        if source.exists():
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for relative in installer.PREREQUISITES:
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
    weights = root / "output/preserved_checkpoint/model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"existing model weights must be preserved")
    return root


def run(root, *args):
    return subprocess.run([sys.executable, str(PACKAGE / "install.py"),
        "--spec-main", str(root), *args], text=True, capture_output=True)


def snapshot(root):
    return {p:str((root / p).read_bytes()) if (root / p).exists() else None for p in installer.FILES}


def test_check_install_reinstall_and_restore(tmp_path):
    root = setup_root(tmp_path)
    before = snapshot(root)
    check = run(root, "--check")
    assert check.returncode == 0, check.stdout + check.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()
    applied = run(root)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    for relative in installer.FILES:
        assert (root / relative).read_bytes() == (PACKAGE / "overlay" / relative).read_bytes()
    modified = root / "Evaluator/run_server_dflash.sh"
    modified.write_text(modified.read_text().replace('PORT="${PORT:-8213}"', 'PORT="${PORT:-8231}"'))
    edited = snapshot(root)
    assert run(root).returncode == 0
    assert snapshot(root) == edited
    receipt_path = root / "Evaluator/integration_backups/paper_comparison_v1.json"
    receipt = json.loads(receipt_path.read_text())
    restore = [sys.executable, str(Path(receipt["backup"]) / "restore.py")]
    blocked = subprocess.run(restore, text=True, capture_output=True)
    assert blocked.returncode != 0 and "Later changes" in blocked.stderr
    assert snapshot(root) == edited
    shutil.copy2(PACKAGE / "overlay/Evaluator/run_server_dflash.sh", modified)
    result = subprocess.run(restore, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(root) == before
    assert not receipt_path.exists()
    assert (root / "output/preserved_checkpoint/model.safetensors").read_bytes() == b"existing model weights must be preserved"
    assert run(root).returncode == 0  # install after rollback


def test_unrelated_evaluator_edit_survives_three_way_merge(tmp_path):
    root = setup_root(tmp_path)
    path = root / "Evaluator/evaluator.py"
    path.write_text("# local dataset note\n" + path.read_text())
    result = run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    text = path.read_text()
    assert text.startswith("# local dataset note\n")
    assert '"dflash_evaluator_integration_v3"' in text


def test_overlapping_local_changes_stop_before_any_writes(tmp_path):
    root = setup_root(tmp_path)
    path = root / "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"
    path.write_text(path.read_text().replace('env = serving_environment(os.environ, kind, args.mode)', 'env = my_environment()'))
    before = snapshot(root)
    result = run(root)
    assert result.returncode != 0 and "conflict with the bundled baseline" in result.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()


def test_previous_simple_launcher_is_a_supported_baseline(tmp_path):
    root = setup_root(tmp_path)
    shutil.copy2(PACKAGE / "baselines/Evaluator/run_server_dflash_simple_previous.sh",
                 root / "Evaluator/run_server_dflash.sh")
    result = run(root, "--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_symlink_destination_is_rejected(tmp_path):
    root = setup_root(tmp_path)
    path = root / "Evaluator/evaluator.py"
    outside = tmp_path / "outside.py"
    path.rename(outside)
    original = outside.read_bytes()
    path.symlink_to(outside)
    result = run(root)
    assert result.returncode != 0 and "redirected" in result.stderr
    assert outside.read_bytes() == original


def test_crlf_checkout_does_not_cause_false_conflicts(tmp_path):
    root = setup_root(tmp_path)
    path = root / "Evaluator/run_server_dspark.sh"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    before = path.read_bytes()
    result = run(root, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == before
    assert run(root).returncode == 0
    receipt = json.loads((root / "Evaluator/integration_backups/paper_comparison_v1.json").read_text())
    assert (Path(receipt["backup"]) / "original/Evaluator/run_server_dspark.sh").read_bytes() == before


def test_explicit_launcher_replacement_is_backed_up_and_reviewable(tmp_path):
    root = setup_root(tmp_path)
    path = root / "Evaluator/run_server_dspark.sh"
    path.write_text('#!/usr/bin/env bash\n# separately committed launcher\nDRAFT_MODEL="/my/epoch5"\nexit 0\n')
    original = path.read_bytes()
    before = snapshot(root)
    result = run(root)
    assert result.returncode != 0
    assert snapshot(root) == before
    flags = ("--replace-launcher", "Evaluator/run_server_dspark.sh")
    check = run(root, "--check", *flags)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "(proposed)" in check.stdout and "/my/epoch5" in check.stdout
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()
    result = run(root, *flags)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((root / "Evaluator/integration_backups/paper_comparison_v1.json").read_text())
    backup = Path(receipt["backup"])
    assert (backup / "original/Evaluator/run_server_dspark.sh").read_bytes() == original
    restore = subprocess.run([sys.executable, str(backup / "restore.py")], text=True, capture_output=True)
    assert restore.returncode == 0, restore.stderr
    assert snapshot(root) == before


def test_launcher_override_never_bypasses_python_conflict(tmp_path):
    root = setup_root(tmp_path)
    path = root / "speculators/scripts/dflash_prefix_compiled/target_graph_server.py"
    path.write_text(path.read_text().replace('env = serving_environment(os.environ, kind, args.mode)', 'env = my_environment()'))
    before = snapshot(root)
    result = run(root, "--replace-launcher", "Evaluator/run_server_dspark.sh")
    assert result.returncode != 0 and "target_graph_server.py" in result.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()


def test_launcher_override_rejects_non_launcher_paths(tmp_path):
    root = setup_root(tmp_path)
    before = snapshot(root)
    result = run(root, "--replace-launcher", "Evaluator/evaluator.py")
    assert result.returncode != 0 and "invalid choice" in result.stderr
    assert snapshot(root) == before


@pytest.mark.parametrize("newlines", (0, 1, 4))
def test_python_eof_newlines_do_not_block_runtime_patch(tmp_path, newlines):
    root = setup_root(tmp_path)
    relatives = (
        "vllm-ascend/vllm_ascend/envs.py",
        "vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py",
        "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
    )
    originals = {}
    for relative in relatives:
        path = root / relative
        path.write_bytes(path.read_bytes().rstrip(b"\n") + b"\n" * newlines)
        originals[relative] = path.read_bytes()
    before = snapshot(root)
    check = run(root, "--check")
    assert check.returncode == 0, check.stdout + check.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()
    result = run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((root / "Evaluator/integration_backups/paper_comparison_v1.json").read_text())
    backup = Path(receipt["backup"])
    for relative, original in originals.items():
        assert (backup / "original" / relative).read_bytes() == original
        assert (root / relative).read_bytes() == (PACKAGE / "overlay" / relative).read_bytes()


def test_eof_normalization_does_not_override_a_real_python_conflict(tmp_path):
    root = setup_root(tmp_path)
    path = root / "vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py"
    text = path.read_text()
    assert "def dummy_run(" in text
    path.write_text(text.replace("def dummy_run(", "def custom_dummy_run(").rstrip("\n") + "\n")
    before = snapshot(root)
    result = run(root, "--replace-launcher", "Evaluator/run_server_dspark.sh")
    assert result.returncode != 0 and "dspark_proposer.py" in result.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()


def uploaded_root(tmp_path):
    root = setup_root(tmp_path)
    with zipfile.ZipFile(PACKAGE / "tests/fixtures/uploaded_source.zip") as archive:
        for name in archive.namelist():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    return root


def test_uploaded_conflict_markers_produce_actionable_error_without_writes(tmp_path):
    root = uploaded_root(tmp_path)
    before = snapshot(root)
    result = run(root, "--check")
    assert result.returncode != 0
    assert "--replace-launcher Evaluator/run_server_dspark.sh" in result.stderr
    assert "no files changed" in result.stderr.lower()
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()


def test_exact_uploaded_files_install_and_restore(tmp_path):
    root = uploaded_root(tmp_path)
    launcher = root / "Evaluator/run_server_dspark.sh"
    launcher.chmod(0o750)
    before = snapshot(root)
    flags = ("--replace-launcher", "Evaluator/run_server_dspark.sh")
    check = run(root, "--check", *flags)
    assert check.returncode == 0, check.stdout + check.stderr
    assert snapshot(root) == before
    assert not (root / "Evaluator/integration_backups").exists()
    result = run(root, *flags)
    assert result.returncode == 0, result.stdout + result.stderr
    assert launcher.stat().st_mode & 0o777 == 0o750
    for relative in installer.FILES:
        if relative != "Evaluator/evaluator.py":
            assert (root / relative).read_bytes() == (PACKAGE / "overlay" / relative).read_bytes()
    # Keep the user's updated example; the installed evaluator still contains
    # the new comparison fingerprints and graph warmup checks.
    evaluator = (root / "Evaluator/evaluator.py").read_text()
    assert '--output "results/spec_compare_$(date +%Y%m%d_%H%M%S)"' in evaluator
    assert '"dflash_evaluator_integration_v3"' in evaluator
    for relative in installer.LAUNCHERS:
        assert subprocess.run(["bash", "-n", str(root / relative)], capture_output=True).returncode == 0
    receipt = json.loads((root / "Evaluator/integration_backups/paper_comparison_v1.json").read_text())
    backup = Path(receipt["backup"])
    with zipfile.ZipFile(PACKAGE / "tests/fixtures/uploaded_source.zip") as archive:
        for name in archive.namelist():
            assert (backup / "original" / name).read_bytes() == archive.read(name)
    restored = subprocess.run([sys.executable, str(backup / "restore.py")], capture_output=True, text=True)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert snapshot(root) == before
    assert launcher.stat().st_mode & 0o777 == 0o750
    assert (root / "output/preserved_checkpoint/model.safetensors").read_bytes() == b"existing model weights must be preserved"
