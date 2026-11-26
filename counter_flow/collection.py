"""Collect controller decisions and first-hit labels from training trajectories."""

import numpy as np
import torch

from .environment import GOAL_RADIUS, CounterFlowEnv, Layout
from .models import image_tensor
from .planner import PETALPlanner, Planner


class TeacherPlanner(Planner):
    @torch.no_grad()
    def cost(self, source, target, actions):
        b, k, horizon, _ = actions.shape
        scores, endpoints = [], []
        goal_position = self.model.position(target)
        for begin in range(0, k, 64):
            selected = actions[:, begin : begin + 64]
            count = selected.shape[1]
            state = (
                source[:, None]
                .expand(b, count, *source.shape[1:])
                .reshape(b * count, *source.shape[1:])
            )
            goal = target[:, None].expand(b, count, *target.shape[1:]).reshape_as(state)
            position = goal_position[:, None].expand(b, count, 2).reshape(b * count, 2)
            cost = state.new_zeros(b * count)
            for step in range(horizon):
                state = self.model.step(
                    state, selected[:, :, step].reshape(b * count, 2)
                ).float()
                if step >= horizon - 5:
                    cost += (self.model.position(state) - position).square().sum(
                        -1
                    ) + 0.01 * (state - goal).square().flatten(1).mean(-1)
            scores.append(cost.reshape(b, count) / 5)
            endpoints.append(state.reshape(b, count, *source.shape[1:]))
        costs = torch.cat(scores, 1)
        # Only the full pool is recorded; the follow-up call rescoring the
        # weighted-mean and best sequences has two candidates.
        if k == self.candidates:
            self.record = (torch.cat(endpoints, 1), actions.clone(), costs.clone())
        return costs


@torch.no_grad()
def teacher_record(data, row, seed, candidates):
    source, target, *_ = data.encode_pairs([row])
    planner = TeacherPlanner(
        data.low, iterations=1, candidates=candidates, horizon=15, execute=10, seed=seed
    )
    # Odd-seeded queries first plan and act for ten steps from an earlier state,
    # so the recorded query starts from a warm, shifted nominal plan.
    warm_start = seed % 2 == 1 and row["start"] >= 10
    if warm_start:
        arrays = data.arrays(row["episode"])
        env = CounterFlowEnv(Layout(**data.layout(row["episode"])), max_steps=30)
        observation, _ = env.reset(
            start=arrays["positions"][row["start"] - 10],
            goal=arrays["positions"][row["start"] + row["gap"]],
        )
        actions, _ = planner.plan(
            data.low.encode(image_tensor(observation[None], data.device)), target
        )
        for action in actions[0, :10].cpu().numpy():
            observation, *_ = env.step(action)
        source = data.low.encode(image_tensor(observation[None], data.device))
    nominal = (
        planner.nominal.clone()
        if planner.nominal is not None
        else source.new_zeros(1, 15, 2)
    )
    planner.plan(source, target)
    endpoints, actions, costs = planner.record
    weights = torch.softmax(
        -(costs - costs.min(-1, keepdim=True).values) / planner.temperature, -1
    )
    update = (weights[..., None, None] * (actions - nominal[:, None])).sum(1)
    return tuple(
        x.detach()
        for x in (source, target, endpoints, actions, costs, weights, nominal, update)
    )


@torch.no_grad()
def execution_outcomes(data, rows, petal, selector, config, seed):
    """Collect first-hit labels with a frozen PETAL executor and a 30-step budget.
    Timeouts are right-censored; initially completed pairs are excluded.
    """
    envs, observations, goals = [], [], []
    for row in rows:
        arrays = data.arrays(row["episode"])
        start, gap = row["start"], row["gap"]
        env = CounterFlowEnv(Layout(**data.layout(row["episode"])), max_steps=30)
        observation, _ = env.reset(
            start=arrays["positions"][start], goal=arrays["positions"][start + gap]
        )
        envs.append(env)
        observations.append(observation)
        goals.append(env.goal_image())
    source = data.low.encode(image_tensor(np.stack(observations), data.device))
    target = data.low.encode(image_tensor(np.stack(goals), data.device))
    initial = np.array(
        [np.linalg.norm(e.position - e.goal) <= GOAL_RADIUS for e in envs]
    )
    done = initial.copy()
    hits = np.full(len(rows), -1, dtype=np.int64)
    planner = PETALPlanner(
        data.low,
        petal,
        selector,
        config["coverage"],
        seed=seed,
        candidates=config["teacher_candidates"],
    )
    for step in range(0, 30, 10):
        if done.all():
            break
        current = data.low.encode(image_tensor(np.stack(observations), data.device))
        actions, _ = planner.plan(current, target)
        actions = actions.cpu().numpy()
        for j in range(10):
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                observations[i], _, success, truncated, _ = env.step(actions[i, j])
                if success:
                    hits[i] = step + j + 1
                done[i] = success or truncated
    valid = torch.as_tensor(~initial, device=source.device)
    return dict(
        source=source[valid].cpu(),
        target=target[valid].cpu(),
        hit=torch.tensor(hits[~initial]),
        group=torch.tensor([r["group"] for i, r in enumerate(rows) if not initial[i]]),
        strata=torch.tensor([r["gap"] for i, r in enumerate(rows) if not initial[i]]),
    )
