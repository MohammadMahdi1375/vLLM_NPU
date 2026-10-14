"""Temporary filesystem tests; no real repositories or checkpoints are changed."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

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
    receipt_path = root / "Evaluator/integration_backups/three_methods_v1.json"
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
    path.write_text(path.read_text().replace('"method": "dflash"', '"method": "my_custom_method"'))
    before = snapshot(root)
    result = run(root)
    assert result.returncode != 0 and "Overlapping local edits" in result.stderr
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
