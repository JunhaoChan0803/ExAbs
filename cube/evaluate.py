"""Evaluate ExAbs, LeWM, and Hi-LeWM on the released Cube manipulation tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(
            "exabs",
            "lewm",
            "hwm_online",
            "hwm_staged",
            "hwm_online_constrained",
            "hwm_staged_constrained",
        ),
        default="exabs",
    )
    parser.add_argument("--distance", type=int, choices=(50, 75), default=50)
    parser.add_argument("--seed", type=int, choices=(260933,), default=260933)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate the first N released tasks."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Check the complete pipeline with a small planning budget.",
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2 as transforms

    from .determinism import configure_process_determinism
    from .models.loading import load_components, load_single_level, load_world_model
    from .planning.config import CostConfig
    from .planning.cost import PlanningCost
    from .planning.policy import HierarchicalWorldModelPolicy

    config = json.loads(
        (ROOT / "configs" / f"distance_{args.distance}.json").read_text()
    )
    queries = json.loads(
        (ROOT / "assets" / f"queries_d{args.distance}.json").read_text()
    )["tasks"]
    if args.method != "exabs":
        baselines = json.loads((ROOT / "configs" / "baselines.json").read_text())
        config.pop("cost")
        if args.method == "lewm":
            config["flat_cem"] = baselines["lewm"]["cem"]
        else:
            mode = "hwm_staged" if "staged" in args.method else "hwm_online"
            config["high_cem"]["horizon"] = baselines[mode]["high_horizon"]
            config["stage_duration_steps"] = (args.distance + 1) // 2
            config["empirical"] = baselines["empirical"]
    count = args.limit or (1 if args.smoke_test else len(queries))
    queries = queries[:count]
    budget = config["eval_budget_env_steps"]
    if args.smoke_test:
        count = min(count, 2)
        queries = queries[:count]
        budget = 2
        for name in ("high_cem", "low_cem", "flat_cem"):
            if name not in config:
                continue
            config[name].update(num_samples=8, iterations=1, topk=4)
    configure_process_determinism(args.seed)
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu or install a CUDA build of PyTorch."
        )
    if args.method == "exabs":
        base = load_world_model(ROOT / "checkpoints", args.device)
        am, petal, duration = load_components(ROOT / "checkpoints", args.device)
        model = (
            PlanningCost(
                base_model=base,
                config=CostConfig(**config["cost"]),
                duration_posterior=duration,
                am=am,
                petal=petal,
            )
            .to(args.device)
            .eval()
        )
        model.set_episode_remaining_env_steps(budget)
    elif args.method == "lewm":
        model = load_single_level(ROOT / "checkpoints", args.device)
    else:
        model = load_world_model(ROOT / "checkpoints", args.device, "hwm.pt")
    process = {}
    normalizers = json.loads((ROOT / "assets" / "normalizers.json").read_text())
    for name, params in normalizers.items():
        scaler = StandardScaler()
        scaler.mean_ = np.array(params["mean"])
        scaler.scale_ = np.array(params["scale"])
        scaler.var_ = np.array(params["var"])
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = params["count"]
        process[name] = scaler
        if name != "action":
            process["goal_" + name] = scaler
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(size=224),
        ]
    )

    def solver(name):
        c = config[name]
        return swm.solver.CEMSolver(
            model=model,
            batch_size=1,
            num_samples=c["num_samples"],
            var_scale=1.0,
            n_steps=c["iterations"],
            topk=c["topk"],
            device=args.device,
            seed=args.seed,
        )

    def plan(name):
        c = config[name]
        return swm.policy.PlanConfig(
            horizon=c["horizon"],
            action_block=c["action_block"],
            receding_horizon=c["receding_horizon"],
        )

    image_transforms = {"pixels": image_transform, "goal": image_transform}
    if args.method == "lewm":
        policy = swm.policy.WorldModelPolicy(
            solver=solver("flat_cem"),
            config=plan("flat_cem"),
            process=process,
            transform=image_transforms,
        )
    else:
        bounds_file = (
            ROOT
            / "assets"
            / (
                "latent_bounds.npz"
                if args.method == "exabs"
                else "runtime/hwm_bounds.npz"
            )
        )
        with np.load(bounds_file, allow_pickle=False) as saved:
            bounds = dict(saved)
        high_solver = solver("high_cem")
        if args.method.endswith("_constrained"):
            from .planning.empirical import EmpiricalMacroActionSolver

            bank_name = (
                "staged_bank.npz" if "staged" in args.method else "online_bank.npz"
            )
            with np.load(
                ROOT / "assets" / "runtime" / bank_name, allow_pickle=False
            ) as saved:
                bank = saved["actions"]
            high = config["high_cem"]
            high_solver = EmpiricalMacroActionSolver(
                model=model,
                macro_bank=bank,
                batch_size=1,
                num_samples=high["num_samples"],
                var_scale=1.0,
                n_steps=high["iterations"],
                topk=high["topk"],
                device=args.device,
                seed=args.seed,
                **config["empirical"],
            )
        policy_class = HierarchicalWorldModelPolicy
        extra = {}
        if "staged" in args.method:
            from .planning.staged import StagedHierarchicalWorldModelPolicy

            policy_class = StagedHierarchicalWorldModelPolicy
            extra["stage_duration_steps"] = config["stage_duration_steps"]
        policy = policy_class(
            model=model,
            high_solver=high_solver,
            low_solver=solver("low_cem"),
            high_config=plan("high_cem"),
            low_config=plan("low_cem"),
            macro_replan_interval=config["high_replan_interval_env_steps"],
            process=process,
            transform=image_transforms,
            high_latent_bounds=bounds,
            **extra,
        )
    archive = np.load(
        ROOT / "assets" / f"tasks_d{args.distance}.npz", allow_pickle=False
    )

    class TaskDataset:
        column_names = [
            k.removeprefix("start_") for k in archive.files if k.startswith("start_")
        ]

        def load_chunk(self, episodes_idx, start, end):
            chunks = []
            for idx in episodes_idx:
                chunk = {
                    k: torch.from_numpy(
                        np.stack(
                            [archive["start_" + k][idx], archive["goal_" + k][idx]]
                        )
                    )
                    for k in self.column_names
                }
                chunk["pixels"] = chunk["pixels"].permute(0, 3, 1, 2)
                chunks.append(chunk)
            return chunks

    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=count,
        max_episode_steps=2 * config["eval_budget_env_steps"],
        image_shape=(224, 224),
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    world.set_policy(policy)
    steps = np.zeros(count, dtype=np.int64)
    done = np.zeros(count, dtype=bool)
    original_run = world._run

    def instrumented_run(*run_args, **run_kwargs):
        upstream_on_step = run_kwargs.get("on_step")

        def on_step(current):
            if upstream_on_step:
                upstream_on_step(current)
            active = ~done
            steps[active] += 1
            done[:] |= active & (current.terminateds | current.truncateds)
            remaining = budget - int(steps.max())
            if remaining > 0 and args.method == "exabs":
                model.set_episode_remaining_env_steps(remaining)

        run_kwargs["on_step"] = on_step
        return original_run(*run_args, **run_kwargs)

    world._run = instrumented_run
    setup = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]
    started = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=TaskDataset(),
            episodes_idx=list(range(count)),
            start_steps=[0] * count,
            goal_offset=args.distance,
            eval_budget=budget,
            callables=setup,
            video=None,
        )
    finally:
        world.envs.close()
        archive.close()
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    episodes = [
        dict(
            task=i,
            source_episode=q["episode_idx"],
            start_step=q["start_step"],
            goal_step=q["goal_step"],
            success=bool(successes[i]),
        )
        for i, q in enumerate(queries)
    ]
    result = {
        "scene": "cube",
        "method": args.method,
        "distance": args.distance,
        "seed": args.seed,
        "profile": "smoke" if args.smoke_test else "evaluation",
        "episodes": episodes,
        "successes": int(successes.sum()),
        "total": count,
        "success_rate": float(successes.mean()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    prefix = "smoke_" if args.smoke_test else ""
    suffix = f"_first{count}" if count < 50 and not args.smoke_test else ""
    output = (
        args.output
        or Path("outputs")
        / "cube"
        / f"{prefix}{args.method}_d{args.distance}_seed{args.seed}{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Cube {args.method} d={args.distance}: {result['successes']}/{count} ({100 * result['success_rate']:.1f}%). Results: {output}"
    )


if __name__ == "__main__":
    main()
