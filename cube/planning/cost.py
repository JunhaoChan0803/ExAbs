from __future__ import annotations

import torch

from cube.models.components import AchievabilityModel
from cube.models.duration import DurationPosterior
from cube.models.petal import PETAL
from cube.planning.budget import remaining_budget_risk
from cube.planning.config import CostConfig, _current_latent


class PlanningCost(torch.nn.Module):
    """Add AM and PETAL terms to the hierarchical CEM costs of the world model.

    The duration posterior of the selected macro-action is kept with its
    subgoal, so the low-level PETAL target uses the matching durations.
    """

    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        config: CostConfig,
        duration_posterior: DurationPosterior,
        petal: PETAL | None = None,
        am: AchievabilityModel | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.duration_posterior = duration_posterior.eval().requires_grad_(False)
        self.petal = petal
        self.am = am
        self._episode_remaining_env_steps: int | None = None
        if config.method in {"petal", "exabs"} and petal is None:
            raise ValueError(f"{config.method} requires PETAL")
        if config.method in {"am", "exabs"} and am is None:
            raise ValueError(f"{config.method} requires AM")
        if petal is not None:
            petal.eval().requires_grad_(False)
        if am is not None:
            am.eval().requires_grad_(False)
        self._selected_duration_probability: torch.Tensor | None = None
        self._selected_subgoal_latent: torch.Tensor | None = None

    def set_episode_remaining_env_steps(self, value: int) -> None:
        """Set the remaining primitive budget before the next planner call."""
        if value <= 0:
            raise ValueError("The remaining episode budget must be positive")
        self._episode_remaining_env_steps = int(value)

    @property
    def latent_action_encoder(self):
        return self.base_model.latent_action_encoder

    def _infer_latent_action_dim(self) -> int:
        return int(self.base_model._infer_latent_action_dim())

    def encode(self, *args, **kwargs):
        return self.base_model.encode(*args, **kwargs)

    def encode_macro_actions(self, *args, **kwargs):
        return self.base_model.encode_macro_actions(*args, **kwargs)

    def encode_macro_actions_with_info(self, *args, **kwargs):
        return self.base_model.encode_macro_actions_with_info(*args, **kwargs)

    @torch.inference_mode()
    def rollout_high(self, z_init: torch.Tensor, macro_actions: torch.Tensor):
        posterior = self.duration_posterior(macro_actions[:, 0])
        self._selected_duration_probability = posterior.probabilities.detach()
        rollout = self.base_model.rollout_high(z_init, macro_actions)
        # Key the duration posterior by subgoal so solver batch slicing keeps alignment.
        self._selected_subgoal_latent = rollout[:, 0, 0].detach()
        return rollout

    def _duration_probability_for_low_target(
        self, target: torch.Tensor
    ) -> torch.Tensor:
        if self._selected_duration_probability is None:
            raise RuntimeError("rollout_high must run before the low-level cost")
        selected_subgoal = self._selected_subgoal_latent.to(target.device)
        probability = self._selected_duration_probability.to(target.device)
        exact_matches = (target[:, None] == selected_subgoal[None]).all(dim=-1)
        resolved = []
        for row in range(target.shape[0]):
            candidates = torch.nonzero(exact_matches[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise RuntimeError("Low-level subgoal does not match a selected macro")
            resolved.append(probability.index_select(0, candidates)[0])
        return torch.stack(resolved, dim=0)

    def rollout_low(self, *args, **kwargs):
        return self.base_model.rollout_low(*args, **kwargs)

    @torch.inference_mode()
    def get_cost_high(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Goal distance of the last waypoint, AM risk of the first, minus AM utility."""
        if self.config.method in {"world_model", "petal"}:
            return self.base_model.get_cost_high(info_dict, action_candidates)
        batch, samples, horizon, high_dim = action_candidates.shape
        device = action_candidates.device
        source_raw = info_dict["z_init"].to(device)
        goal_raw = info_dict["z_goal"].to(device)
        source = _current_latent(source_raw, samples=samples, name="z_init")
        goal = _current_latent(goal_raw, samples=samples, name="z_goal")
        expanded_source = self.base_model._expand_sample_dim(
            source_raw, target_samples=samples
        )
        expanded_goal = self.base_model._expand_sample_dim(
            goal_raw, target_samples=samples
        )
        latent_dim = self._infer_latent_action_dim()
        macros = action_candidates.reshape(
            batch, samples, horizon * (high_dim // latent_dim), latent_dim
        )
        predicted = self.base_model.rollout_high(expanded_source, macros)
        first_waypoint = predicted[:, :, 0]
        final_waypoint = predicted[:, :, -1]
        base_cost = (final_waypoint - expanded_goal).square().sum(-1)
        mask = torch.ones((batch, samples), dtype=torch.bool, device=device)
        am_output = self.am(source, first_waypoint, goal, candidate_mask=mask)
        duration = self.duration_posterior(macros[:, :, 0])
        if self._episode_remaining_env_steps is None:
            raise RuntimeError("Set the remaining episode budget before planning")
        risk = remaining_budget_risk(
            probability=duration.probabilities,
            cdf=am_output.mean_cdf,
            cdf_epistemic_std=am_output.cdf_epistemic_std,
            hazard_bin_steps=am_output.hazard_bin_steps,
            remaining_budget_env_steps=self._episode_remaining_env_steps,
            lcb_standard_deviations=self.config.am_lcb_standard_deviations,
        )
        result = base_cost + self.config.am_risk_weight * risk
        if self.config.am_value_weight:
            result = result - self.config.am_value_weight * am_output.mean_utility
        return result

    @torch.inference_mode()
    def get_cost_low(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Endpoint distance to the subgoal plus a bounded PETAL effect residual."""
        if self.config.method in {"world_model", "am"}:
            return self.base_model.get_cost_low(info_dict, action_candidates)
        batch, samples, horizon_blocks, _ = action_candidates.shape
        device = action_candidates.device
        z_hist = info_dict["z_hist"].to(device)
        z_target = info_dict["z_subgoal"].to(device)
        a_hist = info_dict.get("a_hist")
        source = _current_latent(z_hist, samples=samples, name="z_hist")
        target = _current_latent(z_target, samples=samples, name="z_subgoal")
        expanded_target = self.base_model._expand_sample_dim(
            z_target, target_samples=samples
        )
        predicted = self.base_model.rollout_low(
            z_hist,
            a_hist.to(device) if torch.is_tensor(a_hist) else None,
            action_candidates,
        )
        endpoints = predicted[:, :, -1]
        anchor_cost = (endpoints - expanded_target).square().sum(-1)
        mask = torch.ones((batch, samples), dtype=torch.bool, device=device)
        candidate_duration = torch.full(
            (batch, samples),
            horizon_blocks * self.config.low_action_block_env_steps,
            device=device,
            dtype=endpoints.dtype,
        )
        output = self.petal.score_existing_endpoints(
            source,
            target,
            endpoints,
            target_duration=self._duration_probability_for_low_target(target),
            candidate_duration=candidate_duration,
            candidate_mask=mask,
            anchor_cost=anchor_cost,
            alpha=self.config.petal_alpha,
            residual_clip=self.config.petal_residual_clip,
        )
        return output.anchored_cost

    def get_cost_flat(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.get_cost_flat(info_dict, action_candidates)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        if "z_goal" in info_dict:
            return self.get_cost_high(info_dict, action_candidates)
        return self.get_cost_low(info_dict, action_candidates)
