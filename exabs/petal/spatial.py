"""Pooling and support moments on the normalized [-1, 1] feature grid."""

from __future__ import annotations

import torch


def weighted_pool(feature: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return (feature * support).sum(dim=(-2, -1))


def normalized_grid(height: int, width: int, device, dtype):
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def support_centroid(support: torch.Tensor) -> torch.Tensor:
    grid_x, grid_y = normalized_grid(
        support.shape[-2], support.shape[-1], support.device, support.dtype
    )
    center_x = (support.squeeze(-3) * grid_x).sum(dim=(-2, -1))
    center_y = (support.squeeze(-3) * grid_y).sum(dim=(-2, -1))
    return torch.stack((center_x, center_y), dim=-1)


def relative_support_statistics(
    support: torch.Tensor, energy: torch.Tensor
) -> torch.Tensor:
    """Change strength and second moments of the support around its centroid."""
    grid_x, grid_y = normalized_grid(
        support.shape[-2], support.shape[-1], support.device, support.dtype
    )
    scalar_support = support.squeeze(-3)
    center = support_centroid(support)
    offset_x = grid_x - center[..., 0, None, None]
    offset_y = grid_y - center[..., 1, None, None]
    var_x = (scalar_support * offset_x.square()).sum(dim=(-2, -1))
    var_y = (scalar_support * offset_y.square()).sum(dim=(-2, -1))
    covariance = (scalar_support * offset_x * offset_y).sum(dim=(-2, -1))
    strength = weighted_pool(energy, support).squeeze(-1)
    return torch.stack((strength, var_x, var_y, covariance), dim=-1)
