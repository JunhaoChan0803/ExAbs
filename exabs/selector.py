"""Action-grounded selection of the spatial support shared by a candidate pool."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionGroundedSelectorConfig:
    input_channels: int
    hidden_channels: int = 32
    temperature: float = 0.25
    epsilon: float = 1e-6


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _hard_topk(scores: torch.Tensor, coverage_fraction: float) -> torch.Tensor:
    spatial_size = scores.shape[-2] * scores.shape[-1]
    retained = max(
        1, min(spatial_size, int(math.ceil(coverage_fraction * spatial_size)))
    )
    indices = torch.topk(scores.flatten(start_dim=-2), k=retained, dim=-1).indices
    support = torch.zeros_like(scores.flatten(start_dim=-2))
    support.scatter_(-1, indices, 1.0)
    return support.reshape_as(scores)


class ActionGroundedSupportSelector(nn.Module):
    """Rank source/target locations by their usefulness for predicting actions."""

    def __init__(self, config: ActionGroundedSelectorConfig) -> None:
        super().__init__()
        self.config = config
        groups = _groups(config.hidden_channels)
        self.network = nn.Sequential(
            nn.Conv2d(3 * config.input_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, 1, 1),
        )

    def scores(
        self, source_feature: torch.Tensor, target_feature: torch.Tensor
    ) -> torch.Tensor:
        delta = target_feature - source_feature
        learned = self.network(
            torch.cat((source_feature, target_feature, delta), dim=1)
        )
        energy = delta.square().mean(dim=1, keepdim=True)
        normalized = energy / energy.amax(dim=(-2, -1), keepdim=True).clamp_min(
            self.config.epsilon
        )
        # A weak change prior breaks ties between equally scored locations.
        return learned + torch.log(normalized + self.config.epsilon)

    def hard_support(
        self,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        coverage_fraction: float,
    ) -> torch.Tensor:
        return _hard_topk(
            self.scores(source_feature, target_feature), coverage_fraction
        )

    def straight_through_support(
        self,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        coverage_fraction: float,
    ) -> torch.Tensor:
        scores = self.scores(source_feature, target_feature)
        hard = _hard_topk(scores, coverage_fraction)
        flattened = scores.flatten(start_dim=-2)
        retained = int(hard[0].sum().item())
        threshold = torch.topk(flattened, k=retained, dim=-1).values[..., -1:]
        soft = torch.sigmoid(
            (flattened - threshold.detach()) / float(self.config.temperature)
        ).reshape_as(scores)
        return hard + soft - soft.detach()
