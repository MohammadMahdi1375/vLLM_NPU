#!/usr/bin/env python3
"""Restore package destinations only if installed files have no later edits."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    backup = Path(__file__).resolve().parent
    manifest = json.loads((backup / "manifest.json").read_text())
    root = Path(manifest["root"]).resolve(strict=True)
    plan = []
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        dest = root / rel
        saved = backup / "original" / rel
        if rel.is_absolute() or ".." in rel.parts or dest.is_symlink() or not dest.resolve().is_relative_to(root):
            raise SystemExit(f"Refusing redirected destination: {dest}")
        before = saved.read_bytes() if entry["before_sha256"] is not None else None
        if before is not None and sha(before) != entry["before_sha256"]:
            raise SystemExit(f"Backup integrity failure: {saved}")
        if dest.exists() and not dest.is_file():
            raise SystemExit(f"Destination is not a file: {dest}")
        current = dest.read_bytes() if dest.exists() else None
        if current == before:
            continue
        if current is None or sha(current) != entry["installed_sha256"]:
            raise SystemExit(f"Later changes found; no files restored: {dest}")
        plan.append((dest, before, current, entry["mode"]))
    for dest, before, current, mode in plan:
        if dest.is_symlink() or dest.read_bytes() != current:
            raise SystemExit(f"File changed during restore: {dest}")
        if before is None:
            dest.unlink()
            continue
        fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.restore_", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(before)
            os.chmod(temporary, mode)
            os.replace(temporary, dest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if "receipt" in manifest:
        receipt = root / manifest["receipt"]["path"]
        if (not receipt.is_symlink() and receipt.resolve().is_relative_to(root)
                and receipt.is_file() and sha(receipt.read_bytes()) == manifest["receipt"]["sha256"]):
            receipt.unlink()
    print(f"Restored {len(plan)} package files. Existing checkpoints were not changed.")


if __name__ == "__main__":
    main()
