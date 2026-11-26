"""Hierarchical MPPI planning from encoded visual observations."""

from dataclasses import replace

import torch

from .models import AM, calibrated


def bounded(a):
    a = a.clamp(-1, 1)
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1.0)


class Planner:
    """Low-level MPPI over primitive actions toward a target feature map."""

    def __init__(
        self,
        model,
        iterations=3,
        candidates=500,
        temperature=0.0025,
        horizon=30,
        execute=10,
        seed=71843,
    ):
        self.model = model
        self.iterations = iterations
        self.candidates = candidates
        self.temperature = temperature
        self.horizon = horizon
        self.execute = execute
        self.generator = torch.Generator(
            device=next(model.parameters()).device
        ).manual_seed(seed)
        self.nominal = None

    @torch.no_grad()
    def cost(self, source, target, actions):
        """Visual position error plus 0.01 x feature error, averaged over the last five steps."""
        b, k, h, _ = actions.shape
        out = []
        goal_pos = self.model.position(target)
        for start in range(0, k, 64):
            a = actions[:, start : start + 64]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            goal = target[:, None].expand(b, kk, *target.shape[1:]).reshape_as(z)
            pgoal = goal_pos[:, None].expand(b, kk, 2).reshape(b * kk, 2)
            score = torch.zeros(b * kk, device=z.device)
            for t in range(h):
                z = self.model.step(z, a[:, :, t].reshape(b * kk, 2)).float()
                if t >= h - 5:
                    score = (
                        score
                        + (self.model.position(z) - pgoal).square().sum(-1)
                        + 0.01 * (z - goal).square().flatten(1).mean(-1)
                    )
            out.append(score.reshape(b, kk) / min(h, 5))
        return torch.cat(out, 1)

    def directed(self, source, target):
        """Straight-line action sequences toward the target's position readout."""
        delta = self.model.position(target) - self.model.position(source)
        length = delta.norm(dim=-1, keepdim=True)
        direction = delta / length.clamp_min(1e-8)
        remaining = (
            length / 0.015 - torch.arange(self.horizon, device=delta.device)[None, :]
        ).clamp(0, 1)
        return direction[:, None] * remaining[:, :, None]

    @torch.no_grad()
    def plan(self, source, target):
        b = source.shape[0]
        h = self.horizon
        k = self.candidates
        device = source.device
        if self.nominal is None:
            self.nominal = torch.zeros(b, h, 2, device=device)
        nominal = self.nominal.clone()
        direct = self.directed(source, target)
        directions = bounded(
            source.new_tensor(
                [
                    [0, 0],
                    [0, 1],
                    [0, -1],
                    [1, 0],
                    [-1, 0],
                    [1, 1],
                    [-1, 1],
                    [1, -1],
                    [-1, -1],
                ]
            )
        )
        for iteration in range(self.iterations):
            noise = torch.randn(
                (b, k, (h + 4) // 5, 2), device=device, generator=self.generator
            ) * (0.5 * 0.7**iteration)
            noise = noise.repeat_interleave(5, 2)[:, :, :h]
            samples = bounded(nominal[:, None] + noise)
            # Fixed proposals: the warm start, constant directions, and goal-directed lines.
            samples[:, 0] = nominal
            samples[:, 1:10] = directions[None, :, None, :]
            samples[:, 10] = direct
            for slot, scale in enumerate((0.5, 0.75, 1.25), 11):
                samples[:, slot] = bounded(direct * scale)
            costs = self.cost(source, target, samples)
            weights = torch.softmax(
                -(costs - costs.min(-1, keepdim=True).values) / self.temperature, -1
            )
            mean = bounded((weights[..., None, None] * samples).sum(1))
            ix = costs.argmin(-1)
            best = samples[torch.arange(b, device=device), ix]
            choices = torch.stack([mean, best], 1)
            choice_cost = self.cost(source, target, choices)
            nominal = choices[torch.arange(b, device=device), choice_cost.argmin(-1)]
        self.nominal = torch.cat(
            [nominal[:, self.execute :], torch.zeros_like(nominal[:, : self.execute])],
            1,
        )
        return nominal, {}


class HighLevelPlanner:
    """MPPI over four macro-actions; ExAbs adds AM scoring of the first waypoint."""

    def __init__(
        self,
        high,
        low,
        prior,
        seed,
        candidates=2000,
        temperature=0.0025,
        am=None,
        cal=None,
        project=False,
        aggregation="terminal",
        am_weight=0.01,
        barrier_weight=0.01,
        raw_disagreement=False,
        uncertainty_beta=1.0,
    ):
        self.high, self.low = high, low
        self.candidates, self.temperature = candidates, temperature
        self.mean = prior["mean"]
        self.factor = prior["factor"]
        self.minimum = prior["minimum"]
        self.maximum = prior["maximum"]
        self.generator = torch.Generator(device=self.mean.device).manual_seed(seed)
        self.nominal = None
        self.am = am
        self.cal = cal
        self.project = project
        self.aggregation = aggregation
        self.am_weight = am_weight
        self.barrier_weight = barrier_weight
        self.raw_disagreement = raw_disagreement
        self.uncertainty_beta = uncertainty_beta

    def project_codes(self, x):
        """Project macro codes onto the output geometry of the encoder's LayerNorm."""
        if not self.project:
            return x
        layer = self.high.macro.mu_ln
        w = layer.weight
        b = layer.bias
        if bool((w.abs() < 1e-6).any()):
            raise RuntimeError("Degenerate macro LayerNorm scale")
        y = (x - b) / w
        y = y - y.mean(-1, keepdim=True)
        y = y / y.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        return y * w + b

    @torch.no_grad()
    def cost(self, source, target, actions):
        b, k = actions.shape[:2]
        out = []
        gp = self.low.position(target)
        srcemb = self.am.frame(source) if self.am is not None else None
        for start in range(0, k, 64):
            a = actions[:, start : start + 64]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            goal = target[:, None].expand(b, kk, *target.shape[1:]).reshape_as(z)
            pg = gp[:, None].expand(b, kk, 2).reshape(b * kk, 2)
            costs = []
            for j in range(4):
                z = self.high.step(z, a[:, :, j].reshape(b * kk, 8)).float()
                if j == 0:
                    first = z
                if self.aggregation == "mean" or j == 3:
                    costs.append(
                        (self.low.position(z) - pg).square().sum(-1)
                        + 0.01 * (z - goal).square().flatten(1).mean(-1)
                    )
            base = torch.stack(costs).mean(0)
            if self.am is not None:
                se = srcemb[:, None].expand(b, kk, -1).reshape(b * kk, -1)
                logits = self.am.encoded(se, self.am.frame(first))
                # First-waypoint completion at the fixed reference budget B0 = 10.
                p = calibrated(AM.cdf(logits.mean(0))[:, 9], self.cal, budget=10)
                members = AM.cdf(logits)[:, :, 9]
                if not self.raw_disagreement:
                    members = calibrated(members, self.cal, budget=10)
                rho = (
                    p - self.uncertainty_beta * members.std(0, unbiased=False)
                ).clamp(0, 1)
                threshold = self.cal["threshold10"]
                penalty = -self.am_weight * rho.clamp_min(
                    1e-6
                ).log() + self.barrier_weight * (threshold - rho).clamp_min(0) / max(
                    1 - threshold, 0.001
                )
                base = base + penalty
            out.append(base.reshape(b, kk))
        return torch.cat(out, 1)

    @torch.no_grad()
    def plan(self, source, target):
        b = len(source)
        if self.nominal is None:
            self.nominal = self.project_codes(
                self.mean[None, None].expand(b, 4, 8).clone()
            )
        noise = torch.randn(
            b, self.candidates, 4, 8, generator=self.generator, device=source.device
        )
        samples = self.nominal[:, None] + noise @ self.factor.T
        samples = torch.maximum(torch.minimum(samples, self.maximum), self.minimum)
        samples = self.project_codes(samples)
        samples[:, 0] = self.nominal
        cost = self.cost(source, target, samples)
        if not bool(torch.isfinite(cost).all()):
            raise RuntimeError("Nonfinite macro cost")
        weights = torch.softmax(
            -(cost - cost.min(-1, keepdim=True).values) / self.temperature, -1
        )
        mean = self.project_codes((weights[..., None, None] * samples).sum(1))
        best = samples[torch.arange(b, device=source.device), cost.argmin(-1)]
        choices = torch.stack([mean, best], 1)
        cc = self.cost(source, target, choices)
        selected = choices[torch.arange(b, device=source.device), cc.argmin(-1)]
        waypoint = self.high.step(source, selected[:, 0]).float()
        self.nominal = torch.cat(
            [
                selected[:, 1:],
                self.project_codes(self.mean[None, None].expand(b, 1, 8)),
            ],
            1,
        )
        return waypoint, {}


class PETALPlanner(Planner):
    """Low-level planner whose goal term compares requested and predicted effects."""

    def __init__(
        self,
        low,
        petal,
        selector,
        coverage,
        seed,
        anchor=0.9,
        shortlist=128,
        temperature=0.0025,
        horizon=15,
        execute=10,
        candidates=500,
    ):
        super().__init__(
            low,
            iterations=1,
            candidates=candidates,
            horizon=horizon,
            execute=execute,
            seed=seed,
        )
        self.petal = petal
        self.selector = selector
        self.coverage = coverage
        self.shortlist = int(shortlist)
        self.temperature = float(temperature)
        self.petal.metric_config = replace(
            self.petal.metric_config, executor_anchor_weight=float(anchor)
        )

    @torch.no_grad()
    def cost(self, source, target, actions):
        b, k, h, _ = actions.shape
        ends = []
        for start in range(0, k, 64):
            a = actions[:, start : start + 64]
            kk = a.shape[1]
            z = (
                source[:, None]
                .expand(b, kk, *source.shape[1:])
                .reshape(b * kk, *source.shape[1:])
            )
            for t in range(h):
                z = self.model.step(z, a[:, :, t].reshape(b * kk, 2)).float()
            ends.append(z.reshape(b, kk, *source.shape[1:]))
        # One target-derived support is shared across the whole candidate pool.
        support = self.selector.hard_support(source, target, self.coverage)
        endpoints = torch.cat(ends, 1)
        duration = source.new_full((b,), float(h))
        effect_target = self.petal.build_target(
            source_feature=source,
            target_feature=target,
            target_support=support,
            requested_duration=duration,
        )
        anchors = self.petal.executor_anchor_costs(
            candidate_endpoints=endpoints, target=effect_target
        )
        ids = anchors.topk(min(self.shortlist, k), dim=1, largest=False).indices
        selected = endpoints[torch.arange(b, device=source.device)[:, None], ids]
        scored = self.petal.score_candidate_endpoints(
            source_features=source,
            candidate_endpoints=selected,
            target=effect_target,
            achieved_duration=duration,
            candidate_chunk_size=32,
        )
        return anchors.scatter(1, ids, scored.candidate_costs)
