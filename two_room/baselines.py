"""Single-level and hierarchical world-model control baselines."""

import time

import torch

from .planner import (
    Executor,
    LocalCandidates,
    Planner,
    PlanResult,
    bounded_actions,
    command_dictionary,
    dictionary_duration,
    mean_distance,
)


@torch.no_grad()
def baseline_dictionary(high, commands):
    """Encode pairs of ten-step commands into twenty-step macro-actions."""
    return dictionary_duration(high, commands, 20)


class SinglePlanner(LocalCandidates):
    """Thirty-step predictions with a ten-step receding-horizon execution prefix."""

    def __init__(self, model, *, seed=0):
        super().__init__(model, seed=seed)
        self.horizon = 30

    def candidates(self, source):
        b = source.shape[0]
        h = 30
        commands = command_dictionary(source.device)
        n = len(commands)
        if self.nominal_low is None:
            self.nominal_low = torch.zeros(b, h, 2, device=source.device)
        constant = commands[None, :, None].expand(b, n, h, 2)
        duration = torch.tensor([5, 10, 15, 20, 25], device=source.device)
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
            distance / 0.015 - torch.arange(30, device=source.device)[None]
        ).clamp(0, 1)
        return direction[:, None] * remaining[:, :, None]

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
            [selected[:, 10:], torch.zeros_like(selected[:, :10])], 1
        )
        elapsed = time.perf_counter() - begun
        self.calls.append(elapsed)
        self.records.append(
            dict(mean_rejected=int(reject.sum()), candidate_evaluations=501)
        )
        return PlanResult(selected, target, self.records[-1], {})


class HWMPlanner(Planner):
    """Fixed-duration high-level planning with the shared local controller."""

    def __init__(self, low, high, book, seed, config):
        self.low_model = low
        self.high_model = high
        self.codebook = book
        self.executor = Executor(low, "base", seed=seed)
        self.config = config
        self.chunk = 128
        self.elapsed_steps = 0
        self.previous_raw = None
        self.trace = []
        self.am = None
        device = next(low.parameters()).device
        self.commands = command_dictionary(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def search(self, source, goal):
        if self.config.get("reuse_high_plan", False):
            return super().search(source, goal)
        b = len(source)
        k = 1000
        n = len(self.commands)
        candidates = self.structured(b, k)
        best_cost = None
        best_commands = None
        best_waypoint = None
        for iteration in range(2):
            ix = candidates.reshape(b, k, 5, 2)
            macro = self.codebook[ix[..., 0] * n + ix[..., 1]]
            end, first, _ = self._rollout(
                source, macro, goal, self.high_model, high=True
            )
            p = self.low_model.position(end.flatten(0, 1)).reshape(b, k, 2)
            gp = self.low_model.position(goal)[:, None]
            cost = (
                (p - gp).square().sum(-1)
                + 0.01 * mean_distance(end, goal[:, None])
                + 0.001 * self.commands[candidates].square().mean((-2, -1))
            )
            ix = cost.argmin(-1)
            batch = torch.arange(b, device=source.device)
            current = cost[batch, ix]
            if best_cost is None:
                best_cost = current
                best_commands = candidates[batch, ix].clone()
                best_waypoint = first[batch, ix].clone()
            else:
                improve = current < best_cost
                best_cost = torch.where(improve, current, best_cost)
                best_commands = torch.where(
                    improve[:, None], candidates[batch, ix], best_commands
                )
                best_waypoint = torch.where(
                    improve[:, None, None, None], first[batch, ix], best_waypoint
                )
            if iteration == 0:
                elite = cost.topk(64, largest=False).indices
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
                candidates[:, 0] = best_commands
        return (
            best_waypoint,
            dict(
                cost_min=best_cost, search_candidates=2000, waypoints=5, macro_steps=20
            ),
        )

    @torch.no_grad()
    def plan(self, source, goal, **unused):
        if self.elapsed_steps < self.config["handoff"]:
            target, details = self.search(source, goal)
        else:
            target = goal
            details = dict(final_low_only=True)
        plan = self.executor.plan(source, target)
        self.elapsed_steps += 6
        return PlanResult(plan.actions, target, plan.low, details)
