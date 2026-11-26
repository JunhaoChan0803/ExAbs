"""Read and write a complete ExAbs model in one tensor-only checkpoint."""

import hashlib
import json
from pathlib import Path

import torch


def verify_training_export(assets, checkpoints):
    """Check that a locally trained model uses its matching runtime resources."""
    assets, checkpoints = Path(assets), Path(checkpoints)
    manifest = assets / "checkpoint_manifest.json"
    if not manifest.exists():
        return
    expected = json.loads(manifest.read_text())
    for root, section in ((checkpoints, "checkpoints"), (assets, "resources")):
        for relative, checksum in expected[section].items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid path in checkpoint manifest")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != checksum:
                raise ValueError(f"Training export checksum mismatch: {path}")


def load_exabs(directory, scene):
    path = Path(directory) / f"ExAbs_{scene}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "exabs" or payload.get("scene") != scene:
        raise ValueError(f"Invalid ExAbs checkpoint for {scene}: {path}")
    components = payload["components"]
    if set(components) != {"l2", "am", "petal"}:
        raise ValueError("An ExAbs checkpoint must contain L2, AM and PETAL")
    return components


def save_exabs(directory, scene, *, l2, am, petal):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ExAbs_{scene}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "exabs",
            "scene": scene,
            "components": {"l2": l2, "am": am, "petal": petal},
        },
        temporary,
    )
    temporary.replace(path)
    return path
