# Adapted from Hi-LeWM: https://zenodo.org/records/21353240
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import torch
from gymnasium.spaces import Box
from torchvision import tv_tensors


def _to_torch(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def _infer_latent_dim_from_model(model: torch.nn.Module) -> int:
    if hasattr(model, "_infer_latent_action_dim"):
        return int(model._infer_latent_action_dim())  # type: ignore[attr-defined]
    if hasattr(model, "latent_action_encoder"):
        latent_encoder = model.latent_action_encoder
        if hasattr(latent_encoder, "output_proj"):
            return int(latent_encoder.output_proj.out_features)
        if hasattr(latent_encoder, "latent_dim"):
            return int(latent_encoder.latent_dim)
    raise ValueError("Unable to infer latent action dimension from model.")


class HierarchicalWorldModelPolicy:
    """Two-level MPC policy: high-level latent CEM + low-level primitive CEM."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        high_solver,
        low_solver,
        high_config,
        low_config,
        macro_replan_interval: int = 5,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        high_latent_bounds: dict[str, np.ndarray] | None = None,
    ):
        self.type = "hierarchical_world_model"
        self.env = None

        self.model = model.eval()
        self.high_solver = high_solver
        self.low_solver = low_solver
        self.high_cfg = high_config
        self.low_cfg = low_config
        self.macro_replan_interval = int(macro_replan_interval)
        self.process = process or {}
        self.transform = transform or {}
        self.high_latent_bounds = high_latent_bounds

        self._action_buffer: deque[torch.Tensor] = deque(
            maxlen=self.flatten_receding_horizon_low
        )
        self._next_low_init: torch.Tensor | None = None
        self._next_high_init: torch.Tensor | None = None
        self._z_subgoal: torch.Tensor | None = None
        self._steps_since_high: int = 10**9
        self._latent_action_dim = _infer_latent_dim_from_model(self.model)
        self._low_grouped_action_dim: int | None = None

    @property
    def flatten_receding_horizon_low(self) -> int:
        return int(self.low_cfg.receding_horizon) * int(self.low_cfg.action_block)

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Apply preprocessors/transforms following WorldModelPolicy behavior."""
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray, np.generic))

            if k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}', got {type(v)}"
                    )
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])
                v = self.process[k].transform(v)
                v = v.reshape(shape)
                is_numpy = True

            if k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])

                if k.startswith("pixels") or k.startswith("goal"):
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])
                is_numpy = isinstance(v, (np.ndarray, np.generic))

            if (
                is_numpy
                and getattr(v, "dtype", None) is not None
                and v.dtype.kind not in "USO"
            ):
                v = torch.from_numpy(v)
            info_dict[k] = v

        return info_dict

    def _device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def _build_high_action_space(self, n_envs: int) -> Box:
        if self.high_latent_bounds is not None:
            low_1d = np.asarray(self.high_latent_bounds["low"], dtype=np.float32)
            high_1d = np.asarray(self.high_latent_bounds["high"], dtype=np.float32)
        else:
            low_1d = -np.ones((self._latent_action_dim,), dtype=np.float32)
            high_1d = np.ones((self._latent_action_dim,), dtype=np.float32)

        if low_1d.shape != high_1d.shape:
            raise ValueError("high_latent_bounds low/high must have the same shape")
        if low_1d.ndim != 1:
            raise ValueError("high_latent_bounds low/high must be 1D arrays")

        low = np.broadcast_to(low_1d[None, :], (n_envs, low_1d.shape[0])).copy()
        high = np.broadcast_to(high_1d[None, :], (n_envs, high_1d.shape[0])).copy()
        return Box(
            low=low, high=high, shape=(n_envs, low_1d.shape[0]), dtype=np.float32
        )

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = int(getattr(env, "num_envs", 1))

        self.low_solver.configure(
            action_space=env.action_space,
            n_envs=n_envs,
            config=self.low_cfg,
        )
        self._low_grouped_action_dim = int(self.low_solver.action_dim)

        high_action_space = self._build_high_action_space(n_envs)
        self.high_solver.configure(
            action_space=high_action_space,
            n_envs=n_envs,
            config=self.high_cfg,
        )

        self._action_buffer = deque(maxlen=self.flatten_receding_horizon_low)
        self._next_low_init = None
        self._next_high_init = None
        self._z_subgoal = None
        self._steps_since_high = 10**9

    @torch.inference_mode()
    def _encode_pixels_last(self, pixels: torch.Tensor) -> torch.Tensor:
        # Expect (B, T, C, H, W) or (B, C, H, W)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError(
                f"Unsupported pixel shape for encoding: {tuple(pixels.shape)}"
            )
        batch = {"pixels": pixels}
        out = self.model.encode(batch, encode_actions=False)
        return out["emb"][:, -1]

    @torch.inference_mode()
    def _plan_high(self, *, z_init: torch.Tensor, z_goal: torch.Tensor) -> None:
        high_info = {
            "z_init": z_init,
            "z_goal": z_goal,
        }
        # Keep warm-start tensors on CPU to avoid mixed-device issues in CEM internals.
        high_init_action = self._next_high_init
        if torch.is_tensor(high_init_action):
            high_init_action = high_init_action.detach().cpu()

        outputs = self.high_solver(high_info, init_action=high_init_action)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        keep_h = int(self.high_cfg.receding_horizon)
        high_plan = actions[:, :keep_h]
        self._next_high_init = (
            actions_solver[:, keep_h:].detach().cpu()
            if bool(getattr(self.high_cfg, "warm_start", True))
            else None
        )

        high_action_block = int(self.high_cfg.action_block)
        macro_seq = high_plan.reshape(
            z_init.size(0),
            keep_h * high_action_block,
            self._latent_action_dim,
        )
        pred = self.model.rollout_high(z_init, macro_seq)
        self._z_subgoal = pred[:, 0, 0, :]
        self._steps_since_high = 0

    @torch.inference_mode()
    def _plan_low(self, *, z_init: torch.Tensor) -> None:
        if self._z_subgoal is None:
            raise RuntimeError(
                "Low-level planning requested without a high-level subgoal."
            )
        if self._low_grouped_action_dim is None:
            raise RuntimeError(
                "Policy must be attached to env (set_env) before planning."
            )

        n_envs = z_init.size(0)
        a_hist = torch.zeros(
            (n_envs, 1, self._low_grouped_action_dim),
            device=z_init.device,
            dtype=z_init.dtype,
        )
        low_info = {
            "z_hist": z_init.unsqueeze(1),
            "a_hist": a_hist,
            "z_subgoal": self._z_subgoal,
        }

        # Keep warm-start tensors on CPU to avoid mixed-device issues in CEM internals.
        low_init_action = self._next_low_init
        if torch.is_tensor(low_init_action):
            low_init_action = low_init_action.detach().cpu()

        outputs = self.low_solver(low_info, init_action=low_init_action)
        actions_solver = outputs["actions"]
        if not torch.is_tensor(actions_solver):
            actions_solver = torch.as_tensor(actions_solver)
        actions = actions_solver.to(z_init.device)

        keep_h = int(self.low_cfg.receding_horizon)
        plan = actions[:, :keep_h]
        self._next_low_init = (
            actions_solver[:, keep_h:].detach().cpu()
            if bool(getattr(self.low_cfg, "warm_start", True))
            else None
        )

        # Expand grouped actions into primitive environment steps.
        plan = plan.reshape(self.env.num_envs, self.flatten_receding_horizon_low, -1)
        self._action_buffer.extend(plan.transpose(0, 1))

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

        need_high = (
            self._z_subgoal is None
            or self._steps_since_high >= self.macro_replan_interval
        )
        if need_high:
            self._plan_high(z_init=z_init, z_goal=z_goal)

        if len(self._action_buffer) == 0:
            self._plan_low(z_init=z_init)

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action_np = action.detach().cpu().numpy()

        if "action" in self.process:
            action_np = self.process["action"].inverse_transform(action_np)

        self._steps_since_high += 1
        return action_np
