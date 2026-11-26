# Adapted from Hi-LeWM: https://zenodo.org/records/21353240
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class EmpiricalMacroActionSolver:
    """High-level solver constrained to train-set macro-action sequences.

    Candidates are sampled as empirical_macro_sequence + residual. The returned
    action is always the best actual candidate observed, not the residual mean.
    """

    def __init__(
        self,
        *,
        model,
        macro_bank: np.ndarray | torch.Tensor,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        residual_scale: float = 0.1,
        min_residual_std: float = 1.0e-3,
        return_top_candidates: int = 8,
        stage_sampling: str = "sequence",
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.residual_scale = float(residual_scale)
        self.min_residual_std = float(min_residual_std)
        self.return_top_candidates = int(return_top_candidates)
        self.stage_sampling = str(stage_sampling)
        if self.stage_sampling not in {"sequence", "independent"}:
            raise ValueError("stage_sampling must be one of: sequence, independent.")
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.macro_bank = torch.as_tensor(macro_bank, dtype=torch.float32)

    def configure(self, *, action_space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        expected_shape = (int(config.horizon), self.action_dim)
        if (
            self.macro_bank.ndim != 3
            or tuple(self.macro_bank.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "Empirical macro bank must have shape (N, horizon, action_dim), "
                f"expected (*, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(self.macro_bank.shape)}."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * int(self._config.action_block)

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _sample_base_indices(
        self,
        *,
        current_bs: int,
        bank_size: int,
    ) -> torch.Tensor:
        index_shape = (
            (current_bs, self.num_samples, self.horizon)
            if self.stage_sampling == "independent"
            else (current_bs, self.num_samples)
        )
        return torch.randint(
            bank_size,
            index_shape,
            generator=self.torch_gen,
            device=self.device,
        )

    def _gather_base_actions(
        self, bank: torch.Tensor, base_idx: torch.Tensor
    ) -> torch.Tensor:
        if self.stage_sampling == "sequence":
            return bank[base_idx]
        stage_idx = torch.arange(self.horizon, device=self.device).view(
            1, 1, self.horizon
        )
        stage_idx = stage_idx.expand(base_idx.shape)
        return bank[base_idx, stage_idx]

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        import time

        del init_action
        start_time = time.time()
        bank = self.macro_bank.to(self.device)
        if bank.size(0) <= 0:
            raise ValueError("Empirical macro bank is empty.")

        outputs = {"costs": []}
        actions_out = torch.zeros(
            self.n_envs, self.horizon, self.action_dim, device=self.device
        )
        top_n = max(1, min(self.return_top_candidates, self.num_samples, self.topk))
        top_actions_out = torch.zeros(
            self.n_envs,
            top_n,
            self.horizon,
            self.action_dim,
            device=self.device,
        )
        top_costs_out = torch.full(
            (self.n_envs, top_n), float("inf"), device=self.device
        )

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            residual_mean = torch.zeros(
                current_bs,
                self.horizon,
                self.action_dim,
                device=self.device,
            )
            residual_std = torch.full_like(
                residual_mean, self.residual_scale * self.var_scale
            )

            expanded_infos = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    v_batch = v[start_idx:end_idx].to(self.device)
                    v_batch = v_batch.unsqueeze(1).expand(
                        current_bs, self.num_samples, *v_batch.shape[1:]
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = v[start_idx:end_idx]
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                else:
                    v_batch = v
                expanded_infos[k] = v_batch

            best_costs = torch.full(
                (current_bs, top_n), float("inf"), device=self.device
            )
            best_actions = torch.zeros(
                current_bs,
                top_n,
                self.horizon,
                self.action_dim,
                device=self.device,
            )

            for _ in range(self.n_steps):
                base_idx = self._sample_base_indices(
                    current_bs=current_bs,
                    bank_size=bank.size(0),
                )
                base = self._gather_base_actions(bank, base_idx)
                residual = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                residual = residual * residual_std.unsqueeze(
                    1
                ) + residual_mean.unsqueeze(1)
                residual[:, 0] = 0.0
                candidates = base + residual
                costs = self.model.get_cost(expanded_infos.copy(), candidates)

                _, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                elite_actions = candidates[batch_indices, topk_inds]
                elite_base = base[batch_indices, topk_inds]
                elite_residual = elite_actions - elite_base
                residual_mean = elite_residual.mean(dim=1)
                residual_std = elite_residual.std(dim=1, unbiased=False).clamp_min(
                    self.min_residual_std
                )

                combined_costs = torch.cat([best_costs, costs], dim=1)
                combined_actions = torch.cat([best_actions, candidates], dim=1)
                best_costs, best_inds = torch.topk(
                    combined_costs, k=top_n, dim=1, largest=False
                )
                gather_idx = best_inds[:, :, None, None].expand(
                    -1, -1, self.horizon, self.action_dim
                )
                best_actions = torch.gather(combined_actions, dim=1, index=gather_idx)

            actions_out[start_idx:end_idx] = best_actions[:, 0]
            top_actions_out[start_idx:end_idx] = best_actions
            top_costs_out[start_idx:end_idx] = best_costs
            outputs["costs"].extend(best_costs[:, 0].detach().cpu().tolist())

        outputs["actions"] = actions_out.detach().cpu()
        outputs["candidate_actions"] = top_actions_out.detach().cpu()
        outputs["candidate_costs"] = top_costs_out.detach().cpu()
        print(f"Empirical macro solve time: {time.time() - start_time:.4f} seconds")
        return outputs
