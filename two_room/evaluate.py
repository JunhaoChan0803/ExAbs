"""Run the Two-room benchmark from RGB observations."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import HWMPlanner, SinglePlanner, baseline_dictionary
from .environment import GOAL_RADIUS, Layout, TwoRoom, render_position
from .models import image_tensor, load_models
from .planner import Executor, Planner, command_dictionary, dictionary_duration

ASSETS = Path(__file__).parent / "assets"


def load_suite():
    suite = json.loads((ASSETS / "tasks.json").read_text())
    layouts = {Layout(**row).uid: Layout(**row) for row in suite["layouts"]}
    return suite["tasks"], layouts


@torch.inference_mode()
def run_batch(
    models,
    tasks,
    layouts,
    config,
    calibration,
    seed,
    *,
    max_steps=300,
    book=None,
    method="exabs",
    progress=None,
):
    low = models["low"]
    device = next(low.parameters()).device
    if method == "single":
        planner = SinglePlanner(low, seed=seed)
    elif method == "hwm":
        if book is None:
            book = baseline_dictionary(models["high"], command_dictionary(device))
        planner = HWMPlanner(low, models["high"], book, seed, config)
    elif method == "exabs":
        if book is None:
            book = dictionary_duration(
                models["high"], command_dictionary(device), config["macro_steps"]
            )
        executor = Executor(low, "petal", models["selector"], models["petal"], seed)
        planner = Planner(
            low, models["high"], book, executor, models["am"], calibration, seed, config
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    envs = [TwoRoom(layouts[task["layout_id"]], max_steps=max_steps) for task in tasks]
    observations = [
        env.reset(start=task["start"], goal=task["goal"])[0]
        for env, task in zip(envs, tasks)
    ]
    target = low.encode(
        image_tensor(np.stack([env.goal_image() for env in envs]), device)
    )
    done = np.zeros(len(envs), dtype=bool)
    hits = [None] * len(envs)
    positions = [[env.position.tolist()] for env in envs]
    actions = [[] for env in envs]
    planning_trace = []
    begun = time.perf_counter()
    for step in range(0, max_steps, config["execute_steps"]):
        source = low.encode(image_tensor(np.stack(observations), device))
        plan = planner.plan(source, target)
        controls = plan.actions[:, : config["execute_steps"]].cpu().numpy()
        high_enabled = method != "single" and not plan.high.get("final_low_only", False)
        high_trace = getattr(planner, "trace", [])
        planning_trace.append(
            dict(
                step=step,
                high_enabled=high_enabled,
                waypoint_count=plan.high.get("waypoints", 0),
                macro_steps=plan.high.get("macro_steps", 0),
                prediction_steps=plan.actions.shape[1],
                execute_steps=controls.shape[1],
                low_candidate_evaluations=plan.low.get("candidate_evaluations"),
                warm_available=bool(
                    high_enabled
                    and high_trace
                    and high_trace[-1].get("warm_available", False)
                ),
                online_am=plan.high.get("online_am", False),
            )
        )
        if not np.isfinite(controls).all():
            raise RuntimeError("Planner returned nonfinite actions")
        for j in range(min(config["execute_steps"], max_steps - step)):
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                observations[i], _, success, timeout, _ = env.step(controls[i, j])
                actions[i].append(controls[i, j].tolist())
                positions[i].append(env.position.tolist())
                if success:
                    hits[i] = env.steps
                done[i] = success or timeout
            if done.all():
                break
        if progress is not None:
            progress(min(step + config["execute_steps"], max_steps), int(done.sum()))
        if done.all():
            break
    elapsed = time.perf_counter() - begun
    rows = []
    for i, task in enumerate(tasks):
        distances = np.linalg.norm(np.asarray(positions[i][1:]) - task["goal"], axis=-1)
        crossed = np.flatnonzero(distances <= GOAL_RADIUS)
        recomputed = int(crossed[0]) + 1 if len(crossed) else None
        if recomputed != hits[i]:
            raise RuntimeError(
                "Recorded success does not match the executed trajectory"
            )
        rows.append(
            dict(
                method=method,
                task_id=task["id"],
                layout_id=task["layout_id"],
                planner_seed=seed,
                success=hits[i] is not None,
                first_hit=hits[i],
                steps=len(actions[i]),
                positions=positions[i],
                actions=actions[i],
                planning_trace=[
                    entry for entry in planning_trace if entry["step"] < len(actions[i])
                ],
            )
        )
    return rows, elapsed


def save_video(path, rows, tasks, layouts):
    import cv2

    tasks = {task["id"]: task for task in tasks}
    path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        task = tasks[row["task_id"]]
        layout = layouts[task["layout_id"]]
        writer = cv2.VideoWriter(
            str(path / (row["task_id"] + ".mp4")),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (392, 392),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open video output")
        for position in row["positions"]:
            frame = render_position(layout, position)
            frame = cv2.resize(frame, (392, 392), interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["exabs", "single", "hwm"], default="exabs")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--limit", type=int, default=64, help="Number of evaluation episodes (1–64)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=300, help="Episode step budget"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: outputs/two_room/METHOD)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the selected method and check observation encoding",
    )
    parser.add_argument(
        "--video", action="store_true", help="Also write episode MP4 videos"
    )
    parser.add_argument("--seed", type=int, help="Override the starting planner seed")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    args = parser.parse_args()
    if not 1 <= args.limit <= 64 or not 1 <= args.max_steps <= 300:
        parser.error("Use 1–64 episodes and a step budget of 1–300")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    random.seed(42000001)
    np.random.seed(42000001)
    torch.manual_seed(42000001)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = load_models(args.device, method=args.method)
    tasks, layouts = load_suite()
    if args.check:
        with torch.inference_mode():
            task = tasks[0]
            image = render_position(layouts[task["layout_id"]], task["start"])
            features = models["low"].encode(image_tensor(image[None], args.device))
        print(
            json.dumps(
                dict(
                    models=list(models),
                    feature_shape=list(features.shape),
                    episodes=len(tasks),
                    device=args.device,
                )
            )
        )
        return
    config_file = {
        "exabs": "config.json",
        "hwm": "baseline_config.json",
        "single": "single_config.json",
    }[args.method]
    config = json.loads((ASSETS / config_file).read_text())
    # The planners shift their stored plans by these fixed amounts.
    fixed = (
        {"execute_steps": 10}
        if args.method == "single"
        else {"execute_steps": 6, "macro_steps": 20, "segments": 5}
    )
    for key, value in fixed.items():
        if config[key] != value:
            parser.error(f"The {args.method} planner requires {key}={value}")
    config["budget"] = args.max_steps
    if args.seed is not None:
        config["seed"] = args.seed
    calibration = (
        json.loads((ASSETS / "calibration.json").read_text())
        if args.method == "exabs"
        else {}
    )
    args.output = args.output or Path("outputs/two_room") / args.method
    tasks = tasks[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if args.method == "exabs":
            book = dictionary_duration(
                models["high"], command_dictionary(args.device), config["macro_steps"]
            )
        elif args.method == "hwm":
            book = baseline_dictionary(models["high"], command_dictionary(args.device))
        else:
            book = None
    rows = []
    total_seconds = 0.0
    for begin in range(0, len(tasks), config["batch_size"]):
        batch, seconds = run_batch(
            models,
            tasks[begin : begin + config["batch_size"]],
            layouts,
            config,
            calibration,
            config["seed"] + begin,
            max_steps=args.max_steps,
            book=book,
            method=args.method,
        )
        rows.extend(batch)
        total_seconds += seconds
        print(
            f"Episodes {len(rows)}/{len(tasks)} | successes {sum(row['success'] for row in rows)}",
            flush=True,
        )
        (args.output / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    successes = sum(row["success"] for row in rows)
    summary = dict(
        method=args.method,
        planner_config=config,
        episodes=len(rows),
        successes=successes,
        success_rate=successes / len(rows),
        budget=args.max_steps,
        mean_success_steps=float(
            np.mean([row["first_hit"] for row in rows if row["success"]])
        )
        if successes
        else None,
        evaluation_seconds=total_seconds,
        device=args.device,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.video:
        save_video(args.output / "videos", rows, tasks, layouts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
