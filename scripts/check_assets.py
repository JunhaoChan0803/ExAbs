"""Verify that packaged model and task files match their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=["cube", "two-room", "counter-flow", "two_room", "counter_flow"],
    )
    args = parser.parse_args()
    scene = args.scene.replace("-", "_") if args.scene else None
    manifest_path = ROOT / "assets.json"
    if not manifest_path.is_file():
        print(
            "Missing assets.json. Extract the complete release archive.",
            file=sys.stderr,
        )
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = []
    selected = [
        item
        for item in manifest["files"]
        if scene is None or item["path"].startswith(scene + "/")
    ]
    if not selected:
        print("No assets found for the selected scene.", file=sys.stderr)
        return 1
    for item in selected:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failed.append(f"{relative}: invalid manifest path")
            continue
        path = ROOT / relative
        if not path.is_file():
            failed.append(f"{relative}: missing")
            continue
        with path.open("rb") as stream:
            prefix = stream.read(100)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            failed.append(f"{relative}: run git lfs pull to download this file")
        elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            failed.append(f"{relative}: checksum mismatch")
    if failed:
        print("Asset verification failed:", file=sys.stderr)
        for message in failed:
            print(f"  {message}", file=sys.stderr)
        return 1
    total = sum(item["bytes"] for item in selected)
    print(f"Verified {len(selected)} files ({total / 1024**2:.1f} MiB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
