"""Tensors passed between the PETAL canonicalizer, effect encoder and metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .spatial import support_centroid


@dataclass(frozen=True)
class FixedTargetGrounding:
    """Unit-mass support [...,1,H,W] and its centroid on the [-1, 1] grid.

    One grounding is derived from the requested target and shared by every
    candidate in the pool.
    """

    support: torch.Tensor
    reference_xy: torch.Tensor

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return tuple(self.support.shape[:-3])

    @classmethod
    def from_support(
        cls, support: torch.Tensor, *, epsilon: float = 1e-6
    ) -> "FixedTargetGrounding":
        mass = support.sum(dim=(-3, -2, -1), keepdim=True)
        if bool((mass <= 0).any()):
            raise ValueError("Every support map needs positive mass")
        normalized = support / mass.clamp_min(float(epsilon))
        return cls(support=normalized, reference_xy=support_centroid(normalized))

    def broadcast_to(self, batch_shape: Tuple[int, ...]) -> "FixedTargetGrounding":
        """Repeat the grounding along trailing candidate axes."""
        batch_shape = tuple(int(value) for value in batch_shape)
        extra = len(batch_shape) - len(self.batch_shape)
        spatial = tuple(self.support.shape[-3:])
        support = self.support.reshape(
            self.batch_shape + (1,) * extra + spatial
        ).expand(batch_shape + spatial)
        reference = self.reference_xy.reshape(
            self.batch_shape + (1,) * extra + (2,)
        ).expand(batch_shape + (2,))
        return FixedTargetGrounding(support=support, reference_xy=reference)


@dataclass(frozen=True)
class CanonicalTransition:
    """A transition re-centred on the target support."""

    source_local: torch.Tensor
    target_local: torch.Tensor
    delta_local: torch.Tensor
    centered_support: torch.Tensor
    reference_xy: torch.Tensor
    requested_duration: torch.Tensor


@dataclass(frozen=True)
class PETALTarget:
    """A requested transition together with its effect token."""

    grounding: FixedTargetGrounding
    effect_token: torch.Tensor
    target_feature: torch.Tensor
    requested_duration: torch.Tensor


@dataclass(frozen=True)
class CandidateEffectScores:
    """Effect tokens [B,K,D] and costs [B,K] for a candidate pool."""

    candidate_effects: torch.Tensor
    candidate_costs: torch.Tensor
