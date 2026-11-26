"""Evaluate the released Counter-flow models on the bundled task suite."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from exabs.checkpoint import load_exabs, verify_training_export

from .environment import CounterFlowEnv, Layout, transition
from .models import (
    AM,
    LowModel,
    SegmentedHigh,
    freeze,
    image_tensor,
    make_selector,
    petal_model,
)
from .planner import HighLevelPlanner, PETALPlanner, Planner

ASSETS = Path(__file__).parent / "assets"
CHECKPOINTS = Path(__file__).parent / "checkpoints"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_models(device, method="exabs", assets=ASSETS, checkpoints=CHECKPOINTS):
    """Load tensor-only checkpoints and the corresponding scoring parameters."""
    if method not in {"exabs", "hwm", "single"}:
        raise ValueError(f"Unknown method: {method}")
    assets = Path(assets)
    checkpoints = Path(checkpoints)
    verify_training_export(assets, checkpoints)
    config = read_json(assets / "config.json")

    components = load_exabs(checkpoints, "counter_flow") if method == "exabs" else {}

    def load(model, name, directory=checkpoints):
        state = (
            components[name]
            if name in components
            else torch.load(
                directory / f"{name}.pt", map_location="cpu", weights_only=True
            )
        )
        model.load_state_dict(state, strict=True)
        return freeze(model.to(device))

    low = load(LowModel(), "l1")
    if method == "single":
        return low, None, None, None, None, None, None, config
    high = load(SegmentedHigh(low.shape), "l2" if method == "exabs" else "hwm")
    prior_name = "prior" if method == "exabs" else "baseline_prior"
    prior = {
        key: torch.tensor(value, device=device, dtype=torch.float32)
        for key, value in read_json(assets / f"{prior_name}.json").items()
    }
    if method == "exabs":
        am = load(AM(), "am")
        petal = load(petal_model(), "petal")
        selector = load(make_selector(16), "selector", assets / "runtime")
        calibration = read_json(assets / "calibration.json")
    else:
        am = petal = selector = calibration = None
    return low, high, prior, am, petal, selector, calibration, config


def make_planners(models, batch_index, method):
    low, high, prior, am, petal, selector, calibration, config = models
    seed = config["low_seed"] + batch_index
    host = Planner(
        low,
        iterations=1,
        candidates=config["low_candidates"],
        horizon=config["low_horizon"],
        execute=config["execute_steps"],
        seed=seed,
    )
    if method == "single":
        return host, host, None
    local = (
        PETALPlanner(
            low,
            petal,
            selector,
            config["selector_coverage"],
            seed=seed,
            horizon=config["low_horizon"],
            execute=config["execute_steps"],
            candidates=config["low_candidates"],
            **config["petal_config"],
        )
        if method == "exabs"
        else host
    )
    high_planner = HighLevelPlanner(
        high,
        low,
        prior,
        seed=config["high_seed"] + batch_index,
        candidates=config["high_candidates"],
        am=am,
        cal=calibration,
        project=config["project"],
        aggregation=config["aggregation"],
        am_weight=config["am_weight"],
        barrier_weight=config["barrier_weight"],
        raw_disagreement=config["raw_disagreement"],
        uncertainty_beta=config["uncertainty_beta"],
    )
    return host, local, high_planner


def check_rollout(record):
    """Check that saved positions agree with the executed physical actions."""
    layout = Layout(**record["task"]["layout"])
    position = np.array(record["task"]["start"], dtype=np.float64)
    for action, recorded in zip(record["actions"], record["positions"][1:]):
        position, _, _ = transition(layout, position, action)
        if not np.allclose(position, recorded, atol=1e-10, rtol=0):
            raise RuntimeError(
                "Recorded rollout does not match environment transitions"
            )


@torch.no_grad()
def evaluate(
    device="cuda",
    method="exabs",
    limit=None,
    output=None,
    assets=ASSETS,
    start_batch=0,
    end_batch=None,
    checkpoints=CHECKPOINTS,
):
    """Run consecutive batches while preserving the original task order and seeds."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch"
        )
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    models = load_models(device, method, assets, checkpoints)
    low, *_, config = models
    tasks = read_json(Path(assets) / "tasks.json")
    batch_size = config["batch_size"]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        # Keep full batches so the random draws for each task remain identical.
        tasks = tasks[
            : min(len(tasks), ((limit + batch_size - 1) // batch_size) * batch_size)
        ]
    if start_batch < 0 or (end_batch is not None and end_batch <= start_batch):
        raise ValueError("Invalid batch interval")
    destination = Path(output or f"outputs/counter_flow/{method}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    began = time.time()
    for batch_index, start in enumerate(range(0, len(tasks), batch_size)):
        if batch_index < start_batch or (
            end_batch is not None and batch_index >= end_batch
        ):
            continue
        batch = tasks[start : start + batch_size]
        envs, observations, goals, positions, actions = [], [], [], [], []
        done, hits = [False] * len(batch), [None] * len(batch)
        for task in batch:
            env = CounterFlowEnv(Layout(**task["layout"]), max_steps=config["budget"])
            observation, _ = env.reset(start=task["start"], goal=task["goal"])
            envs.append(env)
            observations.append(observation)
            goals.append(env.goal_image())
            positions.append([task["start"]])
            actions.append([])
        goal = low.encode(image_tensor(np.stack(goals), device))
        host, local, high = make_planners(models, batch_index, method)
        planning_seconds = 0.0
        for step in range(0, config["budget"], config["execute_steps"]):
            tick = time.time()
            source = low.encode(image_tensor(np.stack(observations), device))
            if (
                method != "single"
                and step < config["budget"] - config["final_goal_steps"]
            ):
                waypoint, _ = high.plan(source, goal)
                planned, _ = local.plan(source, waypoint)
            else:
                if local is not host and host.nominal is None:
                    host.nominal = local.nominal.clone()
                planned, _ = host.plan(source, goal)
            planned = planned[:, : config["execute_steps"]].cpu().numpy()
            planning_seconds += time.time() - tick
            for j in range(config["execute_steps"]):
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    observations[i], _, hit, truncated, _ = env.step(planned[i, j])
                    actions[i].append(planned[i, j].tolist())
                    positions[i].append(env.position.tolist())
                    if hit:
                        hits[i] = env.steps
                    done[i] = hit or truncated
            if all(done):
                break
        records = [
            dict(
                task=task,
                success=hits[i] is not None,
                first_hit=hits[i],
                steps=envs[i].steps,
                actions=actions[i],
                positions=positions[i],
                low_seed=config["low_seed"] + batch_index,
                high_seed=config["high_seed"] + batch_index
                if high is not None
                else None,
                planning_seconds=planning_seconds / len(batch),
            )
            for i, task in enumerate(batch)
        ]
        for record in records:
            check_rollout(record)
        write_json(destination / f"batch_{batch_index:02d}.json", records)
        rows.extend(records)
        print(
            f"Completed {len(rows)} episodes; successes {sum(r['success'] for r in rows)}",
            flush=True,
        )
    if not rows:
        raise ValueError("The batch interval contains no tasks")
    successes = [row for row in rows if row["success"]]
    summary = dict(
        method=method,
        episodes=len(rows),
        successes=len(successes),
        success_rate=len(successes) / len(rows),
        mean_success_steps=float(np.mean([r["steps"] for r in successes]))
        if successes
        else None,
        elapsed_seconds=time.time() - began,
        device=str(device),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--method", choices=("exabs", "hwm", "single"), default="exabs")
    parser.add_argument(
        "--limit",
        type=int,
        help="Episode limit, rounded up to complete batches of four",
    )
    parser.add_argument("--output", help="Directory for rollouts and summary.json")
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument(
        "--end-batch",
        type=int,
        help="Exclusive end of the batch interval (0 through 50)",
    )
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
