"""Hierarchical image-based planning with AM and PETAL."""

import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .networks.scoring import risk


def bounded_actions(x):
    x = x.clamp(-1.0, 1.0)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1.0)


def mean_distance(a, b):
    return (a.float() - b.float()).square().flatten(-3).mean(-1)


@dataclass
class PlanResult:
    actions: torch.Tensor
    waypoint: torch.Tensor
    low: dict
    high: dict


def command_dictionary(device):
    angle = torch.arange(8, device=device) * torch.pi / 4
    directions = torch.stack([angle.cos(), angle.sin()], -1)
    return torch.cat(
        [
            torch.zeros(1, 2, device=device),
            (
                directions[:, None]
                * torch.tensor([0.25, 0.5, 0.75, 1.0], device=device)[None, :, None]
            ).flatten(0, 1),
        ]
    )


def shift_primitive_plan(raw, steps=6):
    return torch.cat([raw[:, steps:], torch.zeros_like(raw[:, :steps])], 1)


@torch.no_grad()
def dictionary_duration(high, commands, duration):
    n = len(commands)
    pairs = torch.cartesian_prod(
        torch.arange(n, device=commands.device), torch.arange(n, device=commands.device)
    )
    out = []
    for ix in pairs.split(128):
        ac = torch.zeros(len(ix), 25, 2, device=commands.device)
        ac[:, : duration // 2] = commands[ix[:, 0], None]
        ac[:, duration // 2 : duration] = commands[ix[:, 1], None]
        mask = torch.arange(25, device=ac.device)[None].expand(len(ac), -1) < duration
        out.append(high.encode_actions(ac, mask))
    return torch.cat(out)


class LocalCandidates:
    def __init__(
        self,
        model,
        *,
        feature_weight=0.01,
        seed=0,
        low_horizon=15,
        low_candidates=500,
        execute_steps=6,
        chunk=128,
    ):
        if low_horizon != 15 or low_candidates != 500 or execute_steps != 6:
            raise ValueError(
                "Local planning requires horizon=15, candidates=500, and execute_steps=6"
            )
        self.model = model
        self.feature_weight = feature_weight
        self.horizon = 15
        self.nominal_low = None
        self.calls = []
        self.records = []
        self.chunk = chunk
        self.generator = torch.Generator(
            device=next(model.parameters()).device
        ).manual_seed(seed)

    def candidates(self, source):
        b = source.shape[0]
        h = 15
        commands = command_dictionary(source.device)
        n = len(commands)
        if self.nominal_low is None:
            self.nominal_low = torch.zeros(b, h, 2, device=source.device)
        constant = commands[None, :, None].expand(b, n, h, 2)
        duration = torch.tensor([2, 5, 7, 10, 12], device=source.device)
        moving = commands[1:][None, :, None, None].expand(b, n - 1, 5, h, 2)
        stopped = moving * (
            torch.arange(h, device=source.device)[None, None, None, :, None]
            < duration[None, None, :, None, None]
        )
        fixed = torch.cat(
            [self.nominal_low[:, None], constant, stopped.reshape(b, -1, h, 2)], 1
        )
        remaining = 500 - fixed.shape[1]
        legs = torch.randint(
            n, (b, remaining, 3), device=source.device, generator=self.generator
        )
        cuts = (
            torch.rand(
                b, remaining, h - 1, device=source.device, generator=self.generator
            )
            .topk(2, largest=False)
            .indices.sort(-1)
            .values
            + 1
        )
        t = torch.arange(h, device=source.device)[None, None]
        segment = (t >= cuts[..., :1]).long() + (t >= cuts[..., 1:]).long()
        return torch.cat([fixed, commands[legs.gather(-1, segment)]], 1)

    def directed(self, source, target):
        delta = self.model.position(target) - self.model.position(source)
        distance = delta.norm(dim=-1, keepdim=True)
        direction = delta / distance.clamp_min(1e-8)
        remaining = (
            distance / 0.015 - torch.arange(15, device=source.device)[None]
        ).clamp(0, 1)
        return direction[:, None] * remaining[:, :, None]

    @torch.no_grad()
    def cost(self, source, target, actions):
        b, k, h, _ = actions.shape
        out = []
        for start in range(0, k, self.chunk):
            a = actions[:, start : start + self.chunk]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            goal = target[:, None].expand(b, kk, *target.shape[1:]).reshape_as(z)
            pgoal = self.model.position(goal)
            score = torch.zeros(b * kk, device=z.device)
            for t in range(h):
                z = self.model.step(z, a[:, :, t].reshape(b * kk, 2)).float()
                feature = (z - goal).square().flatten(1).mean(1)
                value = (self.model.position(z) - pgoal).square().sum(
                    1
                ) + self.feature_weight * feature
                score += (1 / h) * value
            out.append(score.reshape(b, kk))
        return torch.cat(out, 1)

    @torch.no_grad()
    def plan(self, source, target, **unused):

        begun = time.perf_counter()
        b = len(source)
        samples = self.candidates(source)
        direct = self.directed(source, target)
        samples[:, -4:] = torch.stack(
            [bounded_actions(direct * s) for s in (0.5, 0.75, 1.0, 1.25)], 1
        )
        costs = self.cost(source, target, samples)
        ix = costs.argmin(1)
        best = samples[torch.arange(b, device=source.device), ix]
        weights = torch.softmax(
            -(costs - costs.min(1, keepdim=True).values) / 0.0025, -1
        )
        mean = bounded_actions((samples * weights[:, :, None, None]).sum(1))
        mean_cost = self.cost(source, target, mean[:, None])[:, 0]
        reject = mean_cost > costs.min(1).values
        selected = torch.where(reject[:, None, None], best, mean)
        self.nominal_low = torch.cat(
            [selected[:, 6:], torch.zeros_like(selected[:, :6])], 1
        )

        elapsed = time.perf_counter() - begun
        self.calls.append(elapsed)
        self.records.append(
            dict(
                mean_rejected=int(reject.sum()),
                candidate_evaluations=501,
            )
        )
        return PlanResult(selected, target, self.records[-1], {})


class Executor(LocalCandidates):
    def __init__(self, low, mode="base", selector=None, petal=None, seed=0):
        super().__init__(low, seed=seed)
        self.mode = mode
        self.selector = selector
        self.petal = petal

    @torch.no_grad()
    def rollout(self, source, target, actions, teacher=False):
        b, k, h, _ = actions.shape
        ends = []
        costs = []
        for start in range(0, k, self.chunk):
            a = actions[:, start : start + self.chunk]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            goal = target[:, None].expand(b, kk, *target.shape[1:]).reshape_as(z)
            score = torch.zeros(b * kk, device=z.device)
            pg = self.model.position(goal) if teacher else None
            for t in range(h):
                z = self.model.step(z, a[:, :, t].reshape(b * kk, 2)).float()
                if teacher:
                    score += (
                        (self.model.position(z) - pg).square().sum(1)
                        + 0.01 * (z - goal).square().flatten(1).mean(1)
                    ) / h
            ends.append(z.reshape(b, kk, *z.shape[1:]))
            costs.append(score.reshape(b, kk))
        return torch.cat(ends, 1), torch.cat(costs, 1)

    @torch.no_grad()
    def target_context(self, source, target):
        support = self.selector.hard_support(source, target, 0.2).float()
        return self.petal.build_target(
            source_feature=source,
            target_feature=target,
            target_support=support,
            requested_duration=source.new_full((len(source),), 15.0),
        )

    @torch.no_grad()
    def endpoint_scores(self, source, end, context, base):
        self.petal.base_costs = base
        try:
            return self.petal.score_candidate_endpoints(
                source_features=source,
                candidate_endpoints=end,
                target=context,
                achieved_duration=source.new_full((len(source),), 15.0),
                candidate_chunk_size=32,
            ).candidate_costs.float()
        finally:
            self.petal.base_costs = None

    @torch.no_grad()
    def teacher_query(self, source, target):
        samples = self.candidates(source)
        direct = self.directed(source, target)
        samples[:, -4:] = torch.stack(
            [bounded_actions(direct * s) for s in (0.5, 0.75, 1.0, 1.25)], 1
        )
        endpoints, costs = self.rollout(source, target, samples, True)
        weights = torch.softmax(
            -(costs - costs.min(1, keepdim=True).values) / 0.0025, -1
        )
        return samples, endpoints, costs, weights

    @torch.no_grad()
    def plan(self, source, target, **unused):
        if self.mode == "base":
            return super().plan(source, target)
        begun = time.perf_counter()
        samples = self.candidates(source)
        direct = self.directed(source, target)
        samples[:, -4:] = torch.stack(
            [bounded_actions(direct * s) for s in (0.5, 0.75, 1.0, 1.25)], 1
        )
        context = self.target_context(source, target)
        end, base = self.rollout(source, target, samples, True)
        score = self.endpoint_scores(source, end, context, base)
        weights = torch.softmax(
            -(score - score.min(1, keepdim=True).values) / 0.0025, -1
        )
        mean = bounded_actions((samples * weights[:, :, None, None]).sum(1))
        mean_end, mean_base = self.rollout(source, target, mean[:, None], True)
        pool = torch.cat([end, mean_end], 1)
        scores = self.endpoint_scores(
            source, pool, context, torch.cat([base, mean_base], 1)
        )
        controls = torch.cat([samples, mean[:, None]], 1)
        ix = scores.argmin(1)
        selected = controls[torch.arange(len(source), device=source.device), ix]
        self.nominal_low = torch.cat(
            [selected[:, 6:], torch.zeros_like(selected[:, :6])], 1
        )
        details = dict(
            candidate_evaluations=501,
            mean_selected=int((ix == 500).sum()),
            mode=self.mode,
        )
        self.records.append(details)
        self.calls.append(time.perf_counter() - begun)
        return PlanResult(selected, target, details, {})


class Planner:
    def __init__(self, low, high, book, executor, am, cal, seed, config):
        self.low_model = low
        self.high_model = high
        self.codebook = book
        self.executor = executor
        self.am = am
        self.cal = cal
        self.config = config
        self.am_budget = config["am_budget"]
        if am is not None and not 1 <= self.am_budget <= len(cal["thresholds"]):
            raise ValueError("AM reference budget is outside the calibration range")
        self.chunk = 128
        self.elapsed_steps = 0
        self.previous_raw = None
        self.trace = []
        self.execution_trace = []
        self.commands = command_dictionary(next(low.parameters()).device)
        self.generator = torch.Generator(
            device=next(low.parameters()).device
        ).manual_seed(seed)

    def structured(self, b, k):
        legs = torch.randint(
            len(self.commands),
            (b, k, 3),
            device=self.commands.device,
            generator=self.generator,
        )
        cuts = (
            torch.rand(b, k, 9, device=self.commands.device, generator=self.generator)
            .topk(2, largest=False)
            .indices.sort(-1)
            .values
            + 1
        )
        t = torch.arange(10, device=self.commands.device)[None, None]
        segment = (t >= cuts[..., :1]).long() + (t >= cuts[..., 1:]).long()
        return legs.gather(-1, segment)

    @torch.no_grad()
    def _rollout(self, source, samples, target, model, high=False):
        b, k, h, d = samples.shape
        end = []
        first = []
        costs = []
        for j in range(0, k, self.chunk):
            a = samples[:, j : j + self.chunk]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            goal = target[:, None].expand(b, kk, *target.shape[1:]).reshape_as(z)
            acc = torch.zeros(b * kk, device=z.device)
            first_z = None
            for t in range(h):
                z = model.step(z, a[:, :, t].reshape(b * kk, d)).float()
                if t == 0:
                    first_z = z
                if not high or t >= max(0, h - 3):
                    acc += mean_distance(z, goal) / (min(h, 3) if high else h)
            end.append(z.reshape(b, kk, *z.shape[1:]))
            first.append(first_z.reshape(b, kk, *z.shape[1:]))
            costs.append(acc.reshape(b, kk))
        return torch.cat(end, 1), torch.cat(first, 1), torch.cat(costs, 1)

    @torch.no_grad()
    def encode_raw(self, raw):
        b = len(raw)
        n = self.config["segments"]
        d = self.config["macro_steps"]
        aa = raw.reshape(b, n, d, 2)
        padded = F.pad(aa, (0, 0, 0, 25 - d))
        mask = torch.arange(25, device=raw.device)[None].expand(b * n, -1) < d
        return self.high_model.encode_actions(padded.flatten(0, 1), mask).reshape(
            b, n, 8
        )

    @torch.no_grad()
    def am_cost(self, source, first):
        b, k = first.shape[:2]
        src = self.am.encode_feasibility_frames(source)
        parts = []
        for start in range(0, k, 128):
            z = first[:, start : start + 128]
            kk = z.shape[1]
            dst = self.am.encode_feasibility_frames(z.flatten(0, 1))
            p, m, _ = self.am.score_feasibility_embeddings(
                src[:, None].expand(b, kk, -1).flatten(0, 1), dst
            )
            q, rho, u = risk(
                p,
                m,
                self.cal,
                torch.full((b * kk,), self.am_budget, device=source.device),
            )
            theta = self.cal["thresholds"][self.am_budget - 1]
            cost = -0.01 * q.log() + 0.01 * (theta - rho).clamp_min(0) / max(
                1 - theta, 0.001
            )
            parts.append(cost.reshape(b, kk))
        return self.config["am_scale"] * torch.cat(parts, 1)

    @torch.no_grad()
    def search(self, source, goal):
        b = len(source)
        k = 1000
        n = len(self.commands)
        candidates = self.structured(b, k)
        best_cost = None
        best_macro = None
        best_raw = None
        warm = (
            None
            if self.previous_raw is None
            else shift_primitive_plan(self.previous_raw, 6)
        )
        warm_macro = None if warm is None else self.encode_raw(warm)
        best_from_warm = torch.zeros(b, device=source.device, dtype=torch.bool)
        for iteration in range(2):
            ix = candidates.reshape(b, k, 5, 2)
            macro = self.codebook[ix[..., 0] * n + ix[..., 1]].clone()
            raw = self.commands[candidates].repeat_interleave(10, dim=2)
            from_warm = torch.zeros((b, k), device=source.device, dtype=torch.bool)
            if warm is not None:
                macro[:, 0] = warm_macro
                raw[:, 0] = warm
                from_warm[:, 0] = True
            if iteration == 1:
                macro[:, 1] = best_macro
                raw[:, 1] = best_raw
                from_warm[:, 1] = best_from_warm
            end, first, _ = self._rollout(
                source, macro, goal, self.high_model, high=True
            )
            pp = self.low_model.position(end.flatten(0, 1)).reshape(b, k, 2)
            gp = self.low_model.position(goal)[:, None]
            cost = (
                (pp - gp).square().sum(-1)
                + 0.01 * (end - goal[:, None]).square().flatten(2).mean(-1)
                + 0.001 * raw.square().mean((-2, -1))
            )
            if self.am is not None:
                cost += self.am_cost(source, first)
            index = cost.argmin(-1)
            batch = torch.arange(b, device=source.device)
            current = cost[batch, index]
            if best_cost is None:
                best_cost = current
                best_raw = raw[batch, index].clone()
                best_macro = macro[batch, index].clone()
                waypoint = first[batch, index].clone()
                best_from_warm = from_warm[batch, index].clone()
            else:
                improve = current < best_cost
                best_cost = torch.where(improve, current, best_cost)
                best_raw = torch.where(
                    improve[:, None, None], raw[batch, index], best_raw
                )
                best_macro = torch.where(
                    improve[:, None, None], macro[batch, index], best_macro
                )
                waypoint = torch.where(
                    improve[:, None, None, None], first[batch, index], waypoint
                )
                best_from_warm = torch.where(
                    improve, from_warm[batch, index], best_from_warm
                )
            if iteration == 0:
                elite_cost = cost.clone()
                if warm is not None:
                    elite_cost[:, 0] = torch.inf
                elite = elite_cost.topk(64, largest=False).indices
                elites = candidates.gather(1, elite[..., None].expand(-1, -1, 10))
                choices = torch.randint(
                    64, (b, 500), device=source.device, generator=self.generator
                )
                mutants = elites.gather(1, choices[..., None].expand(-1, -1, 10))
                mask = (
                    torch.rand(
                        mutants.shape, device=source.device, generator=self.generator
                    )
                    < 0.2
                )
                replacement = torch.randint(
                    n, mutants.shape, device=source.device, generator=self.generator
                )
                candidates = torch.cat(
                    [self.structured(b, 500), torch.where(mask, replacement, mutants)],
                    1,
                )
        self.previous_raw = best_raw.clone()
        self.trace.append(
            dict(
                warm_available=warm is not None,
                warm_selected=best_from_warm.cpu().tolist(),
            )
        )
        return (
            waypoint,
            dict(
                cost_min=best_cost,
                search_candidates=2000,
                waypoints=5,
                macro_steps=20,
                online_am=self.am is not None,
                warm_shift=6,
            ),
        )

    @torch.no_grad()
    def plan(self, source, goal, **unused):
        active = self.elapsed_steps < self.config["handoff"]
        if active:
            target, details = self.search(source, goal)
        else:
            target = goal
            details = dict(final_low_only=True)
        self.execution_trace.append(
            dict(
                step=self.elapsed_steps,
                high_enabled=active,
                waypoint_count=self.config["segments"] if active else 0,
                warm=self.trace[-1] if active else None,
            )
        )
        out = self.executor.plan(source, target)
        self.elapsed_steps += 6
        return PlanResult(out.actions, target, out.low, details)
