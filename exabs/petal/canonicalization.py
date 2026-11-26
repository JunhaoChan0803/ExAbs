"""Centre a transition on its fixed target support and mask other locations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .types import CanonicalTransition, FixedTargetGrounding


class RelativeTransitionCanonicalizer(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        *,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        grounding: FixedTargetGrounding,
        requested_duration: torch.Tensor,
    ) -> CanonicalTransition:
        prefix = tuple(source_feature.shape[:-3])
        channels, height, width = source_feature.shape[-3:]
        count = 1
        for value in prefix:
            count *= value
        source = source_feature.reshape(count, channels, height, width)
        target = target_feature.reshape(count, channels, height, width)
        support = grounding.support.reshape(count, 1, height, width)
        reference = grounding.reference_xy.reshape(count, 2)

        theta = source.new_zeros((count, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = reference
        # Corner alignment matches the grid used for the support centroid.
        grid = F.affine_grid(theta, size=source.shape, align_corners=True)

        def resample(value):
            return F.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )

        source_centered = resample(source)
        target_centered = resample(target)
        support_centered = resample(support)
        support_centered = support_centered / support_centered.sum(
            dim=(-3, -2, -1), keepdim=True
        ).clamp_min(self.epsilon)
        active_support = support_centered / support_centered.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.epsilon)
        source_local = source_centered * active_support
        target_local = target_centered * active_support

        output_shape = prefix + (channels, height, width)
        return CanonicalTransition(
            source_local=source_local.reshape(output_shape),
            target_local=target_local.reshape(output_shape),
            delta_local=(target_local - source_local).reshape(output_shape),
            centered_support=support_centered.reshape(prefix + (1, height, width)),
            reference_xy=grounding.reference_xy,
            requested_duration=requested_duration,
        )
