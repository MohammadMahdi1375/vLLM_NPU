#!/usr/bin/env python3
"""Install configurable DFlash / Prefix / DSpark serving entrypoints."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

VERSION = "DFlash / Prefix / DSpark fast server integration v1"
FILES = (
    "Evaluator/run_server_dflash.sh",
    "Evaluator/run_server_prefix_dflash.sh",
    "Evaluator/run_server_dspark.sh",
    "Evaluator/evaluator.py",
    "speculators/scripts/dflash_prefix_compiled/target_graph_server.py",
    "speculators/scripts/dflash_prefix_compiled/check_target_graph.py",
)
PREREQUISITES = (
    "vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py",
    "vllm-ascend/vllm_ascend/spec_decode/dflash_prefix_proposer.py",
    "vllm/vllm/model_executor/models/qwen3_dspark.py",
    "speculators/scripts/dflash_prefix_compiled/profile_server.py",
    "speculators/scripts/dflash_prefix_compiled/probe_target_graph.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_file(dest, content, mode):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.spec_", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge(before, base, updated, relative):
    if before is None or before in (base, updated):
        return updated
    if base is None:
        raise ValueError(f"An unrelated file already exists: {relative}. No files changed.")
    if not shutil.which("git"):
        raise ValueError("git is required to merge local edits; no files changed")
    with tempfile.TemporaryDirectory(prefix="spec_server_merge_") as temp:
        paths = [Path(temp) / name for name in ("current", "baseline", "updated")]
        for path, content in zip(paths, (before, base, updated)):
            path.write_bytes(content)
        result = subprocess.run(["git", "merge-file", "--stdout", *map(str, paths)], capture_output=True)
        if result.returncode:
            raise ValueError(f"Overlapping local edits in {relative}; no files changed. "
                             "Compare that file with the package baseline and overlay before merging.")
        return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--spec-main", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    package, root = Path(__file__).resolve().parent, args.spec_main.resolve(strict=True)
    for line in (package / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = package / relative
        if not path.resolve().is_relative_to(package) or sha(path.read_bytes()) != expected:
            raise SystemExit(f"Package integrity check failed: {relative}")
    for relative in PREREQUISITES:
        if not (root / relative).is_file():
            raise SystemExit(f"Existing runtime prerequisite is missing: {relative}; no files changed")

    # An installation receipt makes reinstalling a no-op, even after the user
    # edits settings in the new launchers. Later packages should use new receipts.
    receipt = root / "Evaluator/integration_backups/three_methods_v1.json"
    current_package = sha(b"".join((package / "overlay" / p).read_bytes() for p in FILES))
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    same_package = previous.get("package_sha256") == current_package
    if same_package and all((root / p).is_file() and not (root / p).is_symlink()
                            and (root / p).resolve().is_relative_to(root) for p in FILES):
        print(f"{VERSION}: already installed; subsequent local edits retained.")
        return

    plan = []
    for relative in FILES:
        dest = root / relative
        if dest.is_symlink() or not dest.resolve().is_relative_to(root) or (dest.exists() and not dest.is_file()):
            raise SystemExit(f"Refusing redirected or non-file destination: {dest}")
        before = dest.read_bytes() if dest.exists() else None
        content = (package / "overlay" / relative).read_bytes()
        base_path = package / "baselines" / relative
        base = base_path.read_bytes() if base_path.exists() else None
        if relative == "Evaluator/run_server_dflash.sh":
            alternate = (package / "baselines/Evaluator/run_server_dflash_simple_previous.sh").read_bytes()
            if before == alternate:
                base = alternate
        mode = dest.stat().st_mode & 0o777 if dest.exists() else (0o755 if relative.endswith(".sh") else 0o644)
        try:
            content = merge(before, base, content, relative)
            if relative.endswith(".py"):
                ast.parse(content.decode(), filename=relative)
            else:
                check = subprocess.run(["bash", "-n"], input=content, capture_output=True)
                if check.returncode:
                    raise ValueError(f"Shell syntax failed: {relative}: {check.stderr.decode()}")
        except (ValueError, SyntaxError) as error:
            raise SystemExit(str(error)) from error
        if before != content:
            plan.append((relative, dest, before, content, mode))
    print(f"{VERSION}: {len(plan)} files to update/add")
    for relative, *_ in plan:
        print("  " + relative)
    if args.check:
        print("Package, prerequisite, merge and syntax checks passed. No files changed.")
        return
    if not plan:
        print("All files already match. No changes needed.")
        return

    backup = root / "Evaluator/integration_backups" / f"three_methods_v1_{time.time_ns()}"
    backup.mkdir(parents=True)
    entries = []
    for relative, _, before, content, mode in plan:
        if before is not None:
            saved = backup / "original" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(before)
        entries.append(dict(path=relative, before_sha256=sha(before) if before is not None else None,
                            installed_sha256=sha(content), mode=mode))
    receipt_content = (json.dumps(dict(package_sha256=current_package, backup=str(backup))) + "\n").encode()
    (backup / "manifest.json").write_text(json.dumps(dict(root=str(root), files=entries,
        receipt=dict(path=str(receipt.relative_to(root)), sha256=sha(receipt_content))), indent=2) + "\n")
    shutil.copy2(package / "restore.py", backup / "restore.py")
    applied = []
    try:
        for relative, dest, before, content, mode in plan:
            if dest.is_symlink() or not dest.resolve().is_relative_to(root):
                raise RuntimeError(f"Destination changed during installation: {dest}")
            if (dest.read_bytes() if dest.exists() else None) != before:
                raise RuntimeError(f"File changed during installation: {dest}")
            replace_file(dest, content, mode)
            applied.append((dest, before, content, mode))
        replace_file(receipt, receipt_content, 0o644)
    except Exception:
        for dest, before, content, mode in reversed(applied):
            if dest.is_file() and dest.read_bytes() == content:
                if before is None:
                    dest.unlink()
                else:
                    replace_file(dest, before, mode)
        raise
    print(f"Installed {len(plan)} files. Backup: {backup}")
    print(f"Rollback: python {backup / 'restore.py'}")
    print("Launchers: Evaluator/run_server_{dflash,prefix_dflash,dspark}.sh")
    print("Same NPU/port defaults: run sequentially. Select checkpoints with matched training budgets.")
    print("No checkpoints, training code, kernels or acceptance formulas were changed.")


if __name__ == "__main__":
    main()
