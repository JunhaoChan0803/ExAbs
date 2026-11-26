"""Train PETAL, execution reachability and AM-segmented L2 from raw trajectories."""

import argparse
import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from exabs.checkpoint import save_exabs

from .calibration import fit_calibration
from .collection import execution_outcomes, teacher_record
from .data import Trajectories, digest, read_json, write_json
from .models import (
    AM,
    LowModel,
    SegmentedHigh,
    SelectorTraining,
    freeze,
    make_selector,
    petal_model,
)
from .segmentation import optimal_boundaries, score_edges, training_windows
from .training import PETALTraining, am_loss, high_loss

ROOT = Path(__file__).parent
# Sampling streams for the fixed-waypoint execution requests of each split.
OUTCOME_SEEDS = {"train": 811005, "val": 811003, "calibration": 811011}


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_state(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Trainer:
    def __init__(self, args):
        self.args, self.device = args, args.device
        self.output = args.output
        self.work = self.output / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.config = read_json(args.config)
        if args.smoke_test:
            self.config.update(
                selector_updates=2,
                selector_batch=4,
                selector_microbatch=2,
                petal_updates=2,
                petal_batch=1,
                teacher_queries=4,
                teacher_candidates=16,
                validation_queries=2,
                validation_interval=1,
                am_train_attempts=24,
                am_val_attempts=16,
                am_calibration_attempts=16,
                outcome_batch=4,
                am_updates=2,
                am_batch=8,
                am_microbatch=4,
                l2_updates=2,
                l2_batch=2,
                l2_microbatch=1,
                windows_per_episode=2,
                prior_windows=4,
            )
        positive = [
            key
            for key in self.config
            if any(
                word in key
                for word in (
                    "updates",
                    "batch",
                    "queries",
                    "attempts",
                    "interval",
                    "windows",
                )
            )
        ]
        if any(
            not isinstance(self.config[key], int) or self.config[key] <= 0
            for key in positive
        ):
            raise ValueError(
                "Training counts and batch sizes must be positive integers"
            )
        if self.config["teacher_candidates"] < 14:
            raise ValueError("The controller needs at least 14 candidate slots")
        torch.set_num_threads(args.threads)
        self.seed(0)
        self.low = LowModel().to(self.device)
        self.low.load_state_dict(load_state(args.low_checkpoint), strict=True)
        freeze(self.low)
        self.data = Trajectories(
            args.data_root, self.low, self.device, feature_cache=self.work / "features"
        )
        settings = dict(
            config=self.config,
            low_sha256=digest(args.low_checkpoint),
            data=self.data.metadata_hashes,
            groups=self.data.groups,
            smoke_test=args.smoke_test,
        )
        previous = self.output / "training_config.json"
        if previous.exists() and read_json(previous) != settings:
            raise ValueError(
                "The output directory belongs to another training configuration; "
                "choose a new --output"
            )
        write_json(previous, settings)
        self.runtime = read_json(ROOT / "assets/config.json")
        self.runtime["selector_coverage"] = self.config["coverage"]
        self.runtime["low_candidates"] = self.config["teacher_candidates"]
        if args.smoke_test:
            self.runtime.update(high_candidates=16, budget=20, final_goal_steps=10)

    def seed(self, offset):
        seed = self.config["seed"] + offset
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def optimize(
        self, name, model, updates, lr, train_batch, validate, weight_decay=0.0
    ):
        """Resume AdamW state and save the checkpoint with lowest validation loss."""
        directory = self.work / name
        directory.mkdir(exist_ok=True)
        latest, best_path = directory / "latest.pt", directory / "best.pt"
        if (directory / "done.json").exists():
            model.load_state_dict(load_state(best_path), strict=True)
            return model
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        start, best = 0, float("inf")
        if latest.exists():
            state = load_state(latest)
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            start, best = state["step"], state["best"]
        for step in range(start + 1, updates + 1):
            self.seed(100000 + step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = train_batch(model, step)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite gradient in {name}")
            optimizer.step()
            if step % self.config["validation_interval"] == 0 or step == updates:
                model.eval()
                with torch.no_grad():
                    score = float(validate(model))
                if not np.isfinite(score):
                    raise RuntimeError(f"Nonfinite validation loss in {name}")
                if score < best:
                    best = score
                    save_state(best_path, cpu_state(model))
                save_state(
                    latest,
                    dict(
                        model=cpu_state(model),
                        optimizer=optimizer.state_dict(),
                        step=step,
                        best=best,
                    ),
                )
                print(
                    json.dumps(
                        dict(stage=name, step=step, loss=loss_value, validation=score)
                    ),
                    flush=True,
                )
        model.load_state_dict(load_state(best_path), strict=True)
        write_json(
            directory / "done.json",
            dict(updates=updates, checkpoint_sha256=digest(best_path)),
        )
        return model

    def selector(self):
        self.seed(100)
        model = SelectorTraining().to(self.device)
        count = self.config["selector_batch"]
        micro = self.config["selector_microbatch"]

        def loss(m, rows):
            source, target, actions, mask = self.data.encode_pairs(rows)
            pred = m(source, target, mask, self.config["coverage"])
            return ((pred - actions).square() * mask[..., None]).sum() / (
                2 * mask.sum()
            ).clamp_min(1)

        def train(m, step):
            rows = self.data.sample("train", count, 42100 + step)
            total = 0.0
            for begin in range(0, count, micro):
                batch = rows[begin : begin + micro]
                value = loss(m, batch) * len(batch) / count
                value.backward()
                total += float(value.detach())
            return total

        validation = self.data.sample("val", self.config["validation_queries"], 42900)

        def validate(m):
            return sum(
                float(loss(m, validation[i : i + micro]))
                * len(validation[i : i + micro])
                for i in range(0, len(validation), micro)
            ) / len(validation)

        self.optimize(
            "selector",
            model,
            self.config["selector_updates"],
            self.config["selector_lr"],
            train,
            validate,
            1e-4,
        )
        freeze(model)
        save_state(self.work / "selector.pt", cpu_state(model.selector))
        return model.selector

    def petal(self):
        selector = self.selector()
        self.seed(200)
        model = PETALTraining().to(self.device)
        train_rows = self.data.sample("train", self.config["teacher_queries"], 781000)
        val_rows = self.data.sample("val", self.config["validation_queries"], 791000)
        # Keep four complete candidate pools in RAM; larger pools are regenerated.
        cache = OrderedDict()

        def batch(split, index):
            key = (split, index)
            if key in cache:
                cache.move_to_end(key)
                return tuple(x.to(self.device) for x in cache[key])
            row = (train_rows if split == "train" else val_rows)[index]
            record = teacher_record(
                self.data,
                row,
                (781000 if split == "train" else 791000) + index,
                self.config["teacher_candidates"],
            )
            cache[key] = tuple(x.cpu() for x in record)
            while len(cache) > 4:
                cache.popitem(last=False)
            return record

        def loss(m, record):
            with torch.no_grad():
                support = selector.hard_support(
                    record[0], record[1], self.config["coverage"]
                )
            return m(record, support)[0]

        def train(m, step):
            ids = np.random.default_rng(782000 + step).integers(
                len(train_rows), size=self.config["petal_batch"]
            )
            total = 0.0
            for index in ids:
                value = loss(m, batch("train", int(index))) / len(ids)
                value.backward()
                total += float(value.detach())
            return total

        def validate(m):
            return sum(
                float(loss(m, batch("val", i))) for i in range(len(val_rows))
            ) / len(val_rows)

        self.optimize(
            "petal",
            model,
            self.config["petal_updates"],
            self.config["petal_lr"],
            train,
            validate,
        )
        save_state(self.work / "petal.pt", cpu_state(freeze(model.petal)))

    def executor(self):
        selector = make_selector(16).to(self.device)
        petal = petal_model().to(self.device)
        for model, name in ((selector, "selector"), (petal, "petal")):
            path = self.work / f"{name}.pt"
            if not path.exists():
                raise FileNotFoundError("Run --stage petal first")
            model.load_state_dict(load_state(path), strict=True)
            freeze(model)
        return petal, selector

    def collect_outcomes(self, split, count, petal, selector):
        folder = self.work / "outcomes" / split
        folder.mkdir(parents=True, exist_ok=True)
        identity = dict(
            petal=digest(self.work / "petal.pt"),
            selector=digest(self.work / "selector.pt"),
            config=self.config,
            split=split,
            count=count,
        )
        manifest = folder / "identity.json"
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("Executor changed; choose a new training output directory")
        write_json(manifest, identity)
        rows = self.data.sample(split, count, OUTCOME_SEEDS[split], grouped=True)
        for begin in range(0, count, self.config["outcome_batch"]):
            path = folder / f"batch_{begin:05d}.pt"
            if path.exists():
                continue
            batch = rows[begin : begin + self.config["outcome_batch"]]
            save_state(
                path,
                execution_outcomes(
                    self.data,
                    batch,
                    petal,
                    selector,
                    self.config,
                    819000 + begin // self.config["outcome_batch"],
                ),
            )
            print(
                f"Collected {split} outcomes: {min(count, begin + len(batch))}/{count}",
                flush=True,
            )
        return OutcomeDataset(folder, self.device)

    def am(self):
        petal, selector = self.executor()
        sets = {
            split: self.collect_outcomes(split, self.config[key], petal, selector)
            for split, key in [
                ("train", "am_train_attempts"),
                ("val", "am_val_attempts"),
                ("calibration", "am_calibration_attempts"),
            ]
        }
        self.seed(300)
        model = AM().to(self.device)
        count, micro = self.config["am_batch"], self.config["am_microbatch"]

        def loss(m, batch):
            s, t, h, g, strata = batch
            return am_loss(m(s, t), h, g, strata, self.config)

        def train(m, step):
            ids = np.random.default_rng(831000 + step).integers(
                len(sets["train"]), size=count
            )
            total = 0.0
            for begin in range(0, count, micro):
                part = ids[begin : begin + micro]
                value = loss(m, sets["train"].batch(part)) * len(part) / count
                value.backward()
                total += float(value.detach())
            return total

        def validate(m):
            data = sets["val"]
            total = 0.0
            for begin in range(0, len(data), micro):
                ids = range(begin, min(begin + micro, len(data)))
                total += float(loss(m, data.batch(ids))) * len(ids)
            return total / len(data)

        self.optimize(
            "am",
            model,
            self.config["am_updates"],
            self.config["am_lr"],
            train,
            validate,
        )
        freeze(model)
        save_state(self.work / "am.pt", cpu_state(model))
        probabilities, uncertainty, hits = [], [], []
        with torch.no_grad():
            data = sets["calibration"]
            for begin in range(0, len(data), micro):
                s, t, h, *_ = data.batch(range(begin, min(begin + micro, len(data))))
                logits = model(s, t)
                probabilities.append(AM.cdf(logits.mean(0)).cpu().numpy())
                uncertainty.append(AM.cdf(logits).std(0, unbiased=False).cpu().numpy())
                hits.append(h.cpu().numpy())
        calibration = fit_calibration(
            np.concatenate(probabilities),
            np.concatenate(uncertainty),
            np.concatenate(hits),
        )
        write_json(self.work / "calibration.json", calibration)

    def l2(self):
        for name in ("am.pt", "petal.pt", "selector.pt", "calibration.json"):
            if not (self.work / name).exists():
                raise FileNotFoundError("Run --stage am before --stage l2")
        am = AM().to(self.device)
        am.load_state_dict(load_state(self.work / "am.pt"), strict=True)
        freeze(am)
        calibration = read_json(self.work / "calibration.json")
        self.seed(400)
        model = SegmentedHigh(self.low.shape).to(self.device)
        windows = {}
        boundary_dir = self.work / "segments"
        boundary_dir.mkdir(exist_ok=True)
        for split in ("train", "val"):
            windows[split] = []
            indices = self.data.indices[split]
            if self.args.smoke_test:
                indices = indices[:2]
            for index in indices:
                count = min(
                    self.config["windows_per_episode"],
                    self.data.entries[index]["steps"] - 39,
                )
                starts = np.random.default_rng(861000 + index).choice(
                    self.data.entries[index]["steps"] - 39, count, replace=False
                )
                windows[split].extend((index, int(start)) for start in starts)
        am_identity = dict(
            am=digest(self.work / "am.pt"),
            calibration=digest(self.work / "calibration.json"),
        )

        def window(index, start):
            arrays = self.data.episode(index)
            path = boundary_dir / f"episode_{index:04d}.json"
            if path.exists():
                document = read_json(path)
                if document["identity"] != am_identity:
                    raise ValueError("AM changed after segmentation")
            else:
                edges = score_edges(am, calibration, arrays["features"])
                starts = [
                    s for records in windows.values() for i, s in records if i == index
                ]
                document = dict(
                    identity=am_identity,
                    boundaries={
                        str(s): optimal_boundaries(edges, s).tolist() for s in starts
                    },
                )
                write_json(path, document)
            return tuple(
                x.to(self.device)
                for x in training_windows(
                    arrays["features"],
                    arrays["actions"],
                    [document["boundaries"][str(start)]],
                )
            )

        def batch(split, ids):
            rows = [window(*windows[split][int(i)]) for i in ids]
            return tuple(torch.cat([row[k] for row in rows]) for k in range(4))

        def loss(m, split, ids):
            s, t, a, mask = batch(split, ids)
            return high_loss(m(s, a, mask), t, self.low)

        count, micro = self.config["l2_batch"], self.config["l2_microbatch"]

        def train(m, step):
            ids = np.random.default_rng(865000 + step).integers(
                len(windows["train"]), size=count
            )
            total = 0.0
            for begin in range(0, count, micro):
                part = ids[begin : begin + micro]
                value = loss(m, "train", part) * len(part) / count
                value.backward()
                total += float(value.detach())
            return total

        valid_ids = np.random.default_rng(867000).choice(
            len(windows["val"]), min(384, len(windows["val"])), replace=False
        )

        def validate(m):
            return sum(
                float(loss(m, "val", valid_ids[i : i + micro]))
                * len(valid_ids[i : i + micro])
                for i in range(0, len(valid_ids), micro)
            ) / len(valid_ids)

        self.optimize(
            "l2",
            model,
            self.config["l2_updates"],
            self.config["l2_lr"],
            train,
            validate,
        )
        freeze(model)
        save_state(self.work / "l2.pt", cpu_state(model))
        ids = np.random.default_rng(866000).choice(
            len(windows["train"]),
            min(self.config["prior_windows"], len(windows["train"])),
            replace=False,
        )
        codes = []
        with torch.no_grad():
            for begin in range(0, len(ids), micro):
                _, _, a, mask = batch("train", ids[begin : begin + micro])
                codes.append(model.encode_actions(a, mask).flatten(0, 1).cpu())
        codes = torch.cat(codes).double()
        eigenvalues, eigenvectors = torch.linalg.eigh(torch.cov(codes.T))
        prior = dict(
            mean=codes.mean(0).tolist(),
            factor=(
                eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt())
            ).tolist(),
            minimum=codes.min(0).values.tolist(),
            maximum=codes.max(0).values.tolist(),
        )
        write_json(self.work / "prior.json", prior)
        self.export()

    def export(self):
        checkpoints = self.output / "checkpoints"
        assets = self.output / "assets"
        checkpoints.mkdir(exist_ok=True)
        (assets / "runtime").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.args.low_checkpoint, checkpoints / "l1.pt")
        save_exabs(
            checkpoints,
            "counter_flow",
            **{
                name: load_state(self.work / f"{name}.pt")
                for name in ("l2", "am", "petal")
            },
        )
        shutil.copy2(self.work / "selector.pt", assets / "runtime/selector.pt")
        for name in ("prior.json", "calibration.json"):
            shutil.copy2(self.work / name, assets / name)
        shutil.copy2(ROOT / "assets/tasks.json", assets / "tasks.json")
        write_json(assets / "config.json", self.runtime)
        write_json(
            assets / "checkpoint_manifest.json",
            dict(
                checkpoints={p.name: digest(p) for p in checkpoints.glob("*.pt")},
                resources={
                    str(p.relative_to(assets)): digest(p)
                    for p in assets.rglob("*")
                    if p.is_file() and p.name != "checkpoint_manifest.json"
                },
            ),
        )
        print(f"Exported {checkpoints / 'ExAbs_counter_flow.pt'}", flush=True)


class OutcomeDataset:
    def __init__(self, folder, device):
        self.device, self.cache = device, OrderedDict()
        self.files = sorted(folder.glob("batch_*.pt"))
        self.rows = []
        for index, path in enumerate(self.files):
            self.rows.extend(
                (index, row) for row in range(len(load_state(path)["hit"]))
            )
        if not self.rows:
            raise RuntimeError(
                "No nontrivial execution outcomes were collected; increase the sample count"
            )

    def __len__(self):
        return len(self.rows)

    def batch(self, ids):
        selected = []
        for index in ids:
            file_index, row = self.rows[int(index)]
            if file_index not in self.cache:
                self.cache[file_index] = load_state(self.files[file_index])
            self.cache.move_to_end(file_index)
            selected.append(
                {key: value[row] for key, value in self.cache[file_index].items()}
            )
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
        return tuple(
            torch.stack([row[key] for row in selected]).to(self.device)
            for key in ("source", "target", "hit", "group", "strata")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Counter-flow directory from the training-data archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for training state and exported evaluation files",
    )
    parser.add_argument("--stage", choices=("petal", "am", "l2", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument(
        "--low-checkpoint", type=Path, default=ROOT / "checkpoints/l1.pt"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Exercise all stages with small batches and search populations",
    )
    args = parser.parse_args()
    trainer = Trainer(args)
    for stage in ("petal", "am", "l2") if args.stage == "all" else (args.stage,):
        getattr(trainer, stage)()


if __name__ == "__main__":
    main()
