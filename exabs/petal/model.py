"""PETAL: compact effect encoder and target-conditioned effect metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .canonicalization import RelativeTransitionCanonicalizer
from .spatial import relative_support_statistics, weighted_pool
from .time_embedding import ScalarFourierEmbedding
from .types import (
    CandidateEffectScores,
    CanonicalTransition,
    FixedTargetGrounding,
    PETALTarget,
)


@dataclass(frozen=True)
class CompactEffectEncoderConfig:
    feature_channels: int
    effect_dim: int = 48
    hidden_channels: int = 64
    time_embedding_dim: int = 16
    maximum_duration: int = 64


@dataclass(frozen=True)
class EffectMetricConfig:
    effect_dim: int
    hidden_dim: int = 96
    time_embedding_dim: int = 16
    maximum_duration: int = 64
    minimum_scale: float = 0.05
    epsilon: float = 1e-6
    executor_anchor_weight: float = 0.0
    learned_z_limit: float = 2.0


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _check_duration(duration: torch.Tensor, maximum: int) -> None:
    # The Fourier embedding clamps its input, so out-of-range durations would
    # otherwise be encoded silently as the maximum.
    if bool(((duration <= 0) | (duration > float(maximum))).any()):
        raise ValueError(f"PETAL durations must lie in (0, {maximum}]")


class CompactEffectEncoder(nn.Module):
    """Encode a centred local transition; the absolute reference is not an input."""

    def __init__(self, config: CompactEffectEncoderConfig) -> None:
        super().__init__()
        self.config = config
        groups = _groups(config.hidden_channels)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                3 * config.feature_channels + 1,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(groups, config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(groups, config.hidden_channels),
            nn.SiLU(),
        )
        self.duration_embedding = ScalarFourierEmbedding(
            config.time_embedding_dim, config.maximum_duration
        )
        self.effect_head = nn.Sequential(
            nn.Linear(
                config.hidden_channels + 4 + config.time_embedding_dim,
                config.hidden_channels,
            ),
            nn.LayerNorm(config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, config.effect_dim),
            nn.LayerNorm(config.effect_dim),
        )

    def forward(self, transition: CanonicalTransition) -> torch.Tensor:
        prefix = tuple(transition.source_local.shape[:-3])
        channels, height, width = transition.source_local.shape[-3:]
        _check_duration(transition.requested_duration, self.config.maximum_duration)
        count = 1
        for value in prefix:
            count *= value
        support = transition.centered_support.reshape(count, 1, height, width)
        source = transition.source_local.reshape(count, channels, height, width)
        target = transition.target_local.reshape(count, channels, height, width)
        delta = transition.delta_local.reshape(count, channels, height, width)
        hidden = self.spatial_encoder(
            torch.cat((source, target, delta, support), dim=1)
        )
        pooled = weighted_pool(hidden, support)
        energy = delta.square().mean(dim=1, keepdim=True)
        statistics = relative_support_statistics(support, energy)
        duration = self.duration_embedding(
            transition.requested_duration.reshape(count)
        ).to(source.dtype)
        effect = self.effect_head(torch.cat((pooled, statistics, duration), dim=-1))
        return effect.reshape(prefix + (self.config.effect_dim,))


class LearnedEffectMetric(nn.Module):
    """Nonnegative distance with dimension weights conditioned on the target effect."""

    def __init__(self, config: EffectMetricConfig) -> None:
        super().__init__()
        self.config = config
        self.duration_embedding = ScalarFourierEmbedding(
            config.time_embedding_dim, config.maximum_duration
        )
        self.weight_network = nn.Sequential(
            nn.Linear(config.effect_dim + config.time_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.effect_dim + 1),
        )

    def forward(
        self,
        *,
        target_effect: torch.Tensor,
        achieved_effect: torch.Tensor,
        requested_duration: torch.Tensor,
    ) -> torch.Tensor:
        _check_duration(requested_duration, self.config.maximum_duration)
        duration = self.duration_embedding(requested_duration).to(target_effect.dtype)
        raw = self.weight_network(torch.cat((target_effect, duration), dim=-1))
        dimension_weights = F.softplus(raw[..., :-1])
        dimension_weights = dimension_weights / dimension_weights.mean(
            dim=-1, keepdim=True
        ).clamp_min(self.config.epsilon)
        scale = F.softplus(raw[..., -1]) + self.config.minimum_scale
        squared_difference = (target_effect - achieved_effect).square()
        return scale * (dimension_weights * squared_difference).mean(dim=-1)


class PETALCore(nn.Module):
    """Effect comparison for one frozen low-level candidate pool."""

    def __init__(
        self,
        *,
        encoder_config: CompactEffectEncoderConfig,
        metric_config: Optional[EffectMetricConfig] = None,
    ) -> None:
        super().__init__()
        if metric_config is None:
            metric_config = EffectMetricConfig(
                effect_dim=encoder_config.effect_dim,
                time_embedding_dim=encoder_config.time_embedding_dim,
                maximum_duration=encoder_config.maximum_duration,
            )
        self.encoder_config = encoder_config
        self.metric_config = metric_config
        self.canonicalizer = RelativeTransitionCanonicalizer()
        self.effect_encoder = CompactEffectEncoder(encoder_config)
        self.effect_metric = LearnedEffectMetric(metric_config)

    def build_target(
        self,
        *,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        target_support: torch.Tensor,
        requested_duration: torch.Tensor,
    ) -> PETALTarget:
        """Encode the requested transition once per query."""
        grounding = FixedTargetGrounding.from_support(target_support)
        transition = self.canonicalizer(
            source_feature=source_feature,
            target_feature=target_feature,
            grounding=grounding,
            requested_duration=requested_duration,
        )
        effect_token = self.effect_encoder(transition).to(source_feature.dtype)
        return PETALTarget(
            grounding=grounding,
            effect_token=effect_token,
            target_feature=target_feature,
            requested_duration=requested_duration,
        )

    def score_candidate_endpoints(
        self,
        *,
        source_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        target: PETALTarget,
        achieved_duration: torch.Tensor,
        candidate_chunk_size: Optional[int] = None,
    ) -> CandidateEffectScores:
        """Score predicted endpoints [B,K,C,H,W] against the target effect.

        ``achieved_duration`` is [B] or [B,K]. Chunking bounds activation memory.
        """
        batch, candidates, channels, height, width = candidate_endpoints.shape
        if achieved_duration.shape == (batch,):
            durations = achieved_duration[:, None].expand(batch, candidates)
        else:
            durations = achieved_duration
        chunk_size = min(candidate_chunk_size or candidates, candidates)

        effect_chunks = []
        cost_chunks = []
        for start in range(0, candidates, chunk_size):
            stop = min(start + chunk_size, candidates)
            current_count = stop - start
            expanded_source = source_features[:, None].expand(
                batch, current_count, channels, height, width
            )
            duration_chunk = durations[:, start:stop]
            achieved_transition = self.canonicalizer(
                source_feature=expanded_source,
                target_feature=candidate_endpoints[:, start:stop],
                grounding=target.grounding.broadcast_to((batch, current_count)),
                requested_duration=duration_chunk,
            )
            achieved_effect = self.effect_encoder(achieved_transition).to(
                source_features.dtype
            )
            target_effect = target.effect_token[:, None].expand_as(achieved_effect)
            cost = self.effect_metric(
                target_effect=target_effect,
                achieved_effect=achieved_effect,
                requested_duration=duration_chunk,
            )
            effect_chunks.append(achieved_effect)
            cost_chunks.append(cost)

        candidate_effects = torch.cat(effect_chunks, dim=1)
        learned_costs = torch.cat(cost_chunks, dim=1)
        candidate_costs = self._executor_anchored_cost(
            candidate_endpoints=candidate_endpoints,
            target=target,
            learned_costs=learned_costs,
        )
        return CandidateEffectScores(
            candidate_effects=candidate_effects, candidate_costs=candidate_costs
        )

    def _executor_anchored_cost(
        self,
        *,
        candidate_endpoints: torch.Tensor,
        target: PETALTarget,
        learned_costs: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the support anchor with learned costs matched to the anchor scale."""
        anchor_weight = float(self.metric_config.executor_anchor_weight)
        if anchor_weight == 0.0:
            return learned_costs
        anchor = self.executor_anchor_costs(
            candidate_endpoints=candidate_endpoints, target=target
        )
        epsilon = self.metric_config.epsilon
        learned_mean = learned_costs.mean(dim=1, keepdim=True)
        learned_std = learned_costs.std(dim=1, keepdim=True, unbiased=False).clamp_min(
            epsilon
        )
        anchor_mean = anchor.mean(dim=1, keepdim=True)
        anchor_std = anchor.std(dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)
        limit = self.metric_config.learned_z_limit
        normalized_learned = ((learned_costs - learned_mean) / learned_std).clamp(
            min=-limit, max=limit
        )
        scale_matched_learned = anchor_mean + anchor_std * normalized_learned
        return (
            anchor_weight * anchor + (1.0 - anchor_weight) * scale_matched_learned
        ).clamp_min(0.0)

    def executor_anchor_costs(
        self, *, candidate_endpoints: torch.Tensor, target: PETALTarget
    ) -> torch.Tensor:
        """Support-weighted endpoint MSE [B,K] to the target feature."""
        support = target.grounding.support
        normalized_support = support / support.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.metric_config.epsilon)
        per_token_mse = (
            (candidate_endpoints - target.target_feature[:, None]).square().mean(dim=2)
        )
        return (per_token_mse * normalized_support[:, None, 0]).sum(dim=(-2, -1))
