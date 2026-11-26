"""Raw trajectory access and bounded feature caching for Counter-flow training."""

import hashlib
import json
import shutil
import tarfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .models import image_tensor


def digest(path):
    with Path(path).open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _digest(stream)
        )


def _digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def extract_pool(root, pool, manifest):
    """Extract regular NPZ members after checking the supplied archive digest."""
    import zstandard

    archive = root / manifest["archive"]
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing {archive}; extract ExAbs-Training-Data.zip first"
        )
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    expected = {
        str(Path(e["path"]).relative_to("trajectories")): e
        for e in manifest["episodes"]
    }
    with (
        archive.open("rb") as stream,
        zstandard.ZstdDecompressor().stream_reader(stream) as decoded,
    ):
        with tarfile.open(fileobj=decoded, mode="r|") as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                destination = root / "trajectories" / member.name
                if not destination.resolve().is_relative_to(
                    (root / "trajectories").resolve()
                ):
                    raise ValueError("Archive path escapes the trajectory directory")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                with tar.extractfile(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if digest(temporary) != expected[member.name]["sha256"]:
                    raise ValueError(f"Episode checksum mismatch: {member.name}")
                temporary.replace(destination)


class Trajectories:
    """Use training-pool layouts for fitting, validation and calibration."""

    def __init__(self, root, low, device, cache_size=8, feature_cache=None):
        self.root, self.low, self.device = Path(root), low, device
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.raw_cache = OrderedDict()
        self.feature_cache = Path(feature_cache) if feature_cache else None
        if self.feature_cache:
            self.feature_cache.mkdir(parents=True, exist_ok=True)
        self.entries, self.maps, self.metadata_hashes = [], {}, {}
        for pool in ("long", "balanced"):
            path = self.root / "metadata" / f"{pool}.json"
            document = read_json(path)
            self.metadata_hashes[pool] = digest(path)
            if any(not (self.root / e["path"]).is_file() for e in document["episodes"]):
                extract_pool(self.root, pool, document)
            maps_path = self.root / "metadata" / f"{pool}_maps.json"
            self.metadata_hashes[f"{pool}_maps"] = digest(maps_path)
            self.maps[pool] = read_json(maps_path)
            self.entries.extend(
                dict(e, pool=pool)
                for e in document["episodes"]
                if e["split"] == "train"
            )
        groups = sorted({e["group"] for e in self.entries})
        if len(groups) != 40:
            raise ValueError(f"Expected 40 training layout groups, found {len(groups)}")
        self.groups = dict(
            train=groups[:28],
            val=groups[28:32],
            completion=groups[32:36],
            calibration=groups[36:],
        )
        self.indices = {
            split: [i for i, e in enumerate(self.entries) if e["group"] in selected]
            for split, selected in self.groups.items()
        }

    def arrays(self, index):
        index = int(index)
        if index in self.raw_cache:
            self.raw_cache.move_to_end(index)
            return self.raw_cache[index]
        entry = self.entries[index]
        path = self.root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Episode checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.raw_cache[index] = arrays
        while len(self.raw_cache) > self.cache_size:
            self.raw_cache.popitem(last=False)
        return arrays

    @torch.no_grad()
    def episode(self, index):
        index = int(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        arrays = dict(self.arrays(index))
        path = (
            self.feature_cache / f"episode_{index:04d}.npy"
            if self.feature_cache
            else None
        )
        if path and path.exists():
            arrays["features"] = torch.from_numpy(np.load(path, allow_pickle=False))
        else:
            arrays["features"] = torch.cat(
                [
                    self.low.encode(
                        image_tensor(arrays["images"][begin : begin + 32], self.device)
                    ).cpu()
                    for begin in range(0, len(arrays["images"]), 32)
                ]
            )
            if path:
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, arrays["features"].numpy(), allow_pickle=False)
                temporary.replace(path)
        del arrays["images"]
        self.cache[index] = arrays
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return arrays

    def sample(self, split, count, seed, grouped=False):
        rng = np.random.default_rng(seed)
        rows = []
        while len(rows) < count:
            index = int(rng.choice(self.indices[split]))
            start = int(rng.integers(0, self.entries[index]["steps"] - 40 + 1))
            gaps = (
                rng.choice(np.arange(6, 15), size=8, replace=False)
                if grouped
                else [int(rng.integers(6, 15))]
            )
            group = len(rows) // 8 if grouped else len(rows)
            for gap in gaps:
                rows.append(dict(episode=index, start=start, gap=int(gap), group=group))
        return rows[:count]

    @torch.no_grad()
    def encode_pairs(self, rows):
        source, target, actions, masks = [], [], [], []
        for row in rows:
            data = self.arrays(row["episode"])
            start, gap = row["start"], row["gap"]
            source.append(data["images"][start])
            target.append(data["images"][start + gap])
            action = np.zeros((14, 2), np.float32)
            mask = np.arange(14) < gap
            action[:gap] = data["actions"][start : start + gap]
            actions.append(action)
            masks.append(mask)
        images = np.stack(source + target)
        features = torch.cat(
            [
                self.low.encode(image_tensor(images[begin : begin + 32], self.device))
                for begin in range(0, len(images), 32)
            ]
        )
        source_features, target_features = features.chunk(2)
        return (
            source_features,
            target_features,
            torch.as_tensor(np.stack(actions), device=self.device),
            torch.as_tensor(np.stack(masks), device=self.device),
        )

    def layout(self, index):
        entry = self.entries[int(index)]
        return self.maps[entry["pool"]][entry["map_uid"]]
