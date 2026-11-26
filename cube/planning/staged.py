# Adapted from Hi-LeWM: https://zenodo.org/records/21353240
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .policy import HierarchicalWorldModelPolicy, _to_torch


class StagedHierarchicalWorldModelPolicy(HierarchicalWorldModelPolicy):
    """Plan high-level waypoints once and track them in sequence.
    The low-level planner replans from observations at each control update.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        high_solver,
        low_solver,
        high_config,
        low_config,
        stage_duration_steps: int,
        clear_low_buffer_on_stage_change: bool = True,
        macro_replan_interval: int = 5,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(
            model=model,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_config,
            low_config=low_config,
            macro_replan_interval=macro_replan_interval,
            process=process,
            transform=transform,
            high_latent_bounds=high_latent_bounds,
        )
        if int(stage_duration_steps) <= 0:
            raise ValueError("stage_duration_steps must be > 0.")
        self.stage_duration_steps = int(stage_duration_steps)
        self.clear_low_buffer_on_stage_change = bool(clear_low_buffer_on_stage_change)
        self._stage_targets: torch.Tensor | None = None
        self._active_stage_idx: int = 0
        self._steps_total: int = 0

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self._stage_targets = None
        self._active_stage_idx = 0
        self._steps_total = 0

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        high_info = {
            "z_init": z_init,
            "z_goal": z_goal,
        }
        outputs = self.high_solver(high_info, init_action=None)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        full_h = int(self.high_cfg.horizon)
        high_plan = actions[:, :full_h]
        high_action_block = int(self.high_cfg.action_block)
        macro_seq = high_plan.reshape(
            z_init.size(0),
            full_h * high_action_block,
            self._latent_action_dim,
        )
        pred = self.model.rollout_high(z_init, macro_seq)
        if pred.ndim != 4 or pred.size(1) != 1:
            raise ValueError(
                "Expected rollout_high to return shape (B, 1, T, D) for staged planning, "
                f"got {tuple(pred.shape)}"
            )
        self._stage_targets = pred[:, 0]
        self._steps_since_high = 0
        self._set_active_stage(0)

    def _set_active_stage(self, stage_idx: int) -> None:
        if self._stage_targets is None:
            raise RuntimeError(
                "Stage targets must be initialized before setting a stage."
            )

        clamped_idx = max(0, min(int(stage_idx), int(self._stage_targets.size(1)) - 1))
        stage_changed = self._z_subgoal is None or clamped_idx != self._active_stage_idx
        self._active_stage_idx = clamped_idx
        self._z_subgoal = self._stage_targets[:, clamped_idx, :]

        if stage_changed:
            self._next_low_init = None
            if self.clear_low_buffer_on_stage_change:
                self._action_buffer.clear()

    def _sync_active_stage(self) -> None:
        if self._stage_targets is None:
            raise RuntimeError(
                "Stage targets must be initialized before syncing stages."
            )

        target_idx = min(
            self._steps_total // self.stage_duration_steps,
            int(self._stage_targets.size(1)) - 1,
        )
        self._set_active_stage(target_idx)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert self.env is not None, "Environment not set for policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(dict(info_dict))
        device = self._device()

        pixels = _to_torch(info_dict["pixels"], device)
        goal = _to_torch(info_dict["goal"], device)
        z_init = self._encode_pixels_last(pixels)
        z_goal = self._encode_pixels_last(goal)

        if self._stage_targets is None:
            self._plan_high(z_init=z_init, z_goal=z_goal)
        else:
            self._sync_active_stage()

        if len(self._action_buffer) == 0:
            self._plan_low(z_init=z_init)

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action_np = action.detach().cpu().numpy()

        if "action" in self.process:
            action_np = self.process["action"].inverse_transform(action_np)

        self._steps_total += 1
        self._steps_since_high += 1
        return action_np
