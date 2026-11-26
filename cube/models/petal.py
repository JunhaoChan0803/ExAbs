from __future__ import annotations

import math

import torch
from torch import nn

from cube.models.components import (
    CubeCLSAdapter,
    PETALOutput,
    PositiveDefiniteMetric,
)
from cube.models.config import ModelConfig
from cube.models.duration import DURATION_SUPPORT


def _candidate_mask(mask: torch.Tensor, batch: int, count: int) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool)
    if value.shape != (batch, count):
        raise ValueError(f"candidate mask must have shape {(batch, count)}")
    return value


class _DurationFeature(nn.Module):
    def __init__(self, width: int, maximum_duration: float) -> None:
        super().__init__()
        self.maximum_duration = float(maximum_duration)
        self.network = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )

    def forward(self, duration: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(duration, dtype=torch.float32)
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("PETAL durations must be positive and finite")
        x = value / self.maximum_duration
        features = torch.stack(
            (
                x,
                x.square(),
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.log1p(value) / math.log1p(self.maximum_duration),
            ),
            dim=-1,
        )
        return self.network(features)


class DurationAwareEffectEncoder(nn.Module):
    """Encode source-conditioned endpoint effects in a shared duration space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.source_adapter = CubeCLSAdapter(config.latent_dim, d)
        self.endpoint_adapter = CubeCLSAdapter(config.latent_dim, d)
        self.delta_adapter = CubeCLSAdapter(config.latent_dim, d)
        self.duration_adapter = _DurationFeature(d, config.maximum_duration_steps)
        self.network = nn.Sequential(
            nn.Linear(4 * d, config.petal_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.petal_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.petal_hidden_dim, config.effect_dim),
            nn.LayerNorm(config.effect_dim),
        )

    def forward(
        self,
        source: torch.Tensor,
        endpoint: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim != 2 or endpoint.shape != source.shape:
            raise ValueError("effect source/endpoint must align as [N,D]")
        if duration.shape != source.shape[:1]:
            raise ValueError("effect duration must have shape [N]")
        return self.network(
            torch.cat(
                (
                    self.source_adapter(source),
                    self.endpoint_adapter(endpoint),
                    self.delta_adapter(endpoint - source),
                    self.duration_adapter(duration.to(source.device)),
                ),
                dim=-1,
            )
        )


class PETAL(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        shared = DurationAwareEffectEncoder(self.config)
        self.target_effect_encoder = shared
        self.candidate_effect_encoder = shared
        self.metric = PositiveDefiniteMetric(
            self.config.effect_dim,
            self.config.petal_metric_rank,
            self.config.metric_diagonal_floor,
        )
        self.register_buffer("residual_center", torch.tensor(0.0))
        self.register_buffer("residual_scale", torch.tensor(1.0))
        self.register_buffer(
            "residual_statistics_ready", torch.tensor(False, dtype=torch.bool)
        )

    def set_residual_statistics(self, *, center: float, scale: float) -> None:
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("PETAL residual statistics are invalid")
        self.residual_center.fill_(center)
        self.residual_scale.fill_(scale)
        self.residual_statistics_ready.fill_(True)

    def _target_effect(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> torch.Tensor:
        batch = source.shape[0]
        if target_duration.shape == (batch,):
            return self.target_effect_encoder(source, target, target_duration)
        if target_duration.shape != (batch, len(DURATION_SUPPORT)):
            raise ValueError("target duration must be exact [B] or posterior [B,7]")
        if not torch.isfinite(target_duration).all() or (target_duration < 0).any():
            raise ValueError("target duration posterior is invalid")
        if not torch.allclose(
            target_duration.sum(-1),
            torch.ones(batch, device=target_duration.device),
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError("target duration posterior must sum to one")
        support = torch.tensor(
            DURATION_SUPPORT, device=source.device, dtype=source.dtype
        )
        count = len(DURATION_SUPPORT)
        encoded = self.target_effect_encoder(
            source[:, None].expand(-1, count, -1).reshape(batch * count, -1),
            target[:, None].expand(-1, count, -1).reshape(batch * count, -1),
            support[None].expand(batch, -1).reshape(-1),
        ).reshape(batch, count, -1)
        return (target_duration.to(encoded.dtype).unsqueeze(-1) * encoded).sum(dim=1)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        candidates: torch.Tensor,
        *,
        target_duration: torch.Tensor,
        candidate_duration: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PETALOutput:
        if source.ndim != 2 or target.shape != source.shape:
            raise ValueError("PETAL source/target must align as [B,D]")
        if (
            candidates.ndim != 3
            or candidates.shape[0] != source.shape[0]
            or candidates.shape[2] != source.shape[1]
        ):
            raise ValueError("PETAL candidates must have shape [B,K,D]")
        batch, count, width = candidates.shape
        mask = _candidate_mask(candidate_mask, batch, count).to(candidates.device)
        if candidate_duration.ndim == 0:
            candidate_duration = candidate_duration.expand(batch, count)
        elif candidate_duration.shape == (batch,):
            candidate_duration = candidate_duration[:, None].expand(-1, count)
        if candidate_duration.shape != (batch, count):
            raise ValueError("candidate duration must be scalar, [B], or [B,K]")
        target_effect = self._target_effect(source, target, target_duration)
        flat_source = (
            source[:, None].expand(-1, count, -1).reshape(batch * count, width)
        )
        candidate_effect = self.candidate_effect_encoder(
            flat_source,
            candidates.reshape(batch * count, width),
            candidate_duration.reshape(-1).to(candidates.device),
        ).reshape(batch, count, -1)
        effect_cost = self.metric(
            candidate_effect - target_effect[:, None]
        ).masked_fill(~mask, 0.0)
        return PETALOutput(
            target_effect=target_effect,
            candidate_effect=candidate_effect.masked_fill(~mask.unsqueeze(-1), 0.0),
            effect_cost=effect_cost,
        )

    def score_existing_endpoints(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        candidates: torch.Tensor,
        *,
        target_duration: torch.Tensor,
        candidate_duration: torch.Tensor,
        candidate_mask: torch.Tensor,
        anchor_cost: torch.Tensor,
        alpha: float,
        residual_clip: float = 3.0,
    ) -> PETALOutput:
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("PETAL alpha must be finite and non-negative")
        if not math.isfinite(residual_clip) or residual_clip <= 0:
            raise ValueError("PETAL residual clip must be positive")
        if anchor_cost.shape != candidates.shape[:2]:
            raise ValueError("PETAL anchor cost shape differs from candidates")
        output = self(
            source,
            target,
            candidates,
            target_duration=target_duration,
            candidate_duration=candidate_duration,
            candidate_mask=candidate_mask,
        )
        if alpha == 0:
            residual = torch.zeros_like(anchor_cost)
            anchored = anchor_cost
        else:
            if not bool(self.residual_statistics_ready.item()):
                raise RuntimeError("PETAL residual statistics were not fitted")
            normalized = (
                output.effect_cost - self.residual_center
            ) / self.residual_scale
            residual = alpha * normalized.clamp(-residual_clip, residual_clip)
            anchored = anchor_cost + residual
        output.bounded_residual = residual
        output.anchored_cost = anchored
        return output


__all__ = [
    "DurationAwareEffectEncoder",
    "PETAL",
]
