from __future__ import annotations

import torch

from cube.models.duration import DURATION_SUPPORT


def remaining_budget_risk(
    *,
    probability: torch.Tensor,
    cdf: torch.Tensor,
    cdf_epistemic_std: torch.Tensor,
    hazard_bin_steps: tuple[int, ...],
    remaining_budget_env_steps: int,
    lcb_standard_deviations: float,
    floor: float = 1.0e-6,
) -> torch.Tensor:
    """Duration-weighted -log of the AM lower bound on P(T <= min(d, B_remaining)).

    ``probability`` is the duration posterior over ``DURATION_SUPPORT`` [...,D];
    ``cdf`` and its member standard deviation are given on ``hazard_bin_steps``.
    Each budget reads the last hazard bin that does not exceed it.
    """
    device = cdf.device
    bins = torch.as_tensor(hazard_bin_steps, device=device, dtype=torch.long)
    support = torch.as_tensor(DURATION_SUPPORT, device=device, dtype=torch.long)
    remaining = torch.broadcast_to(
        torch.as_tensor(remaining_budget_env_steps, device=device).long(),
        probability.shape[:-1],
    )
    effective = torch.minimum(
        support.view(*([1] * remaining.ndim), -1), remaining[..., None]
    )
    indices = torch.searchsorted(bins, effective, right=True) - 1
    indices = indices.clamp(0, cdf.shape[-1] - 1)
    mean = torch.gather(
        cdf[..., None, :].expand(*cdf.shape[:-1], len(support), cdf.shape[-1]),
        -1,
        indices[..., None],
    ).squeeze(-1)
    std = torch.gather(
        cdf_epistemic_std[..., None, :].expand(
            *cdf.shape[:-1], len(support), cdf.shape[-1]
        ),
        -1,
        indices[..., None],
    ).squeeze(-1)
    lcb = (mean - float(lcb_standard_deviations) * std).clamp_min(float(floor))
    weights = probability.to(lcb.dtype)
    return (weights * -torch.log(lcb)).sum(dim=-1)
