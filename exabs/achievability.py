"""Achievability Model (AM): waypoint completion as a discrete-time survival model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AchievabilityConfig:
    feature_channels: int
    maximum_budget: int = 32
    maximum_gap: int = 32
    frame_hidden_channels: Tuple[int, ...] = (48, 96, 96)
    spatial_pool_grid_size: int = 6
    embedding_dim: int = 192
    transformer_heads: int = 6
    transformer_layers: int = 3
    transformer_feedforward_dim: int = 512
    candidate_set_layers: int = 2
    candidate_set_feedforward_dim: int = 512
    bootstrap_heads: int = 3
    dropout: float = 0.10


@dataclass
class AchievabilityOutput:
    hazard_logits: torch.Tensor
    hit_cdf: torch.Tensor
    expected_hit_time: torch.Tensor
    predicted_log_residual: torch.Tensor
    predicted_progress: torch.Tensor
    waypoint_utility: torch.Tensor
    pair_embedding: torch.Tensor
    hazard_member_logits: torch.Tensor
    hit_cdf_members: torch.Tensor
    utility_members: torch.Tensor
    feasibility_uncertainty: torch.Tensor
    value_uncertainty: torch.Tensor


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FrameFeatureEncoder(nn.Module):
    """Compress one spatial feature map into an embedding."""

    def __init__(self, config: AchievabilityConfig) -> None:
        super().__init__()
        layers = []
        input_channels = config.feature_channels
        for index, output_channels in enumerate(config.frame_hidden_channels):
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=5 if index == 0 else 3,
                        stride=2,
                        padding=2 if index == 0 else 1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            input_channels = output_channels
        self.spatial = nn.Sequential(*layers)
        pool_grid = config.spatial_pool_grid_size
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
            nn.Linear(input_channels * pool_grid * pool_grid, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.project(self.spatial(features))


class _RelationEncoder(nn.Module):
    def __init__(self, config: AchievabilityConfig, roles: int) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.roles = nn.Parameter(torch.empty(roles + 1, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.embedding_dim),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.roles, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat((cls, tokens), dim=1) + self.roles.unsqueeze(0)
        return self.encoder(sequence)[:, 0]


class AchievabilityModel(nn.Module):
    """Completion CDFs of the executor and a goal-conditioned waypoint utility.

    The completion branch sees only the source and the waypoint; budgets index
    its CDF rather than entering the representation.
    """

    def __init__(self, config: AchievabilityConfig) -> None:
        super().__init__()
        self.config = config
        self.value_frame_encoder = FrameFeatureEncoder(config)
        self.feasibility_frame_encoder = FrameFeatureEncoder(config)
        self.feasibility_relation = _RelationEncoder(config, roles=2)
        self.value_relation = _RelationEncoder(config, roles=3)
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.candidate_set_feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set_encoder = nn.TransformerEncoder(
            candidate_layer,
            num_layers=config.candidate_set_layers,
            norm=nn.LayerNorm(config.embedding_dim),
            enable_nested_tensor=False,
        )
        self.candidate_set_norm = nn.LayerNorm(config.embedding_dim)
        self.hazard_heads = nn.ModuleList(
            nn.Linear(config.embedding_dim, config.maximum_budget)
            for _ in range(config.bootstrap_heads)
        )
        self.utility_heads = nn.ModuleList(
            nn.Linear(config.embedding_dim, 1) for _ in range(config.bootstrap_heads)
        )
        self.residual_head = nn.Linear(config.embedding_dim, 1)
        self.progress_head = nn.Linear(config.embedding_dim, 1)
        for head in self.hazard_heads:
            nn.init.constant_(head.bias, -2.5)

    @staticmethod
    def _member_cdf(logits: torch.Tensor) -> torch.Tensor:
        hazard = torch.sigmoid(logits)
        survival = torch.cumprod((1.0 - hazard).clamp_min(1e-7), dim=-1)
        return 1.0 - survival

    def encode_feasibility_frames(self, features: torch.Tensor) -> torch.Tensor:
        return self.feasibility_frame_encoder(features)

    def score_feasibility_embeddings(
        self, feasibility_source: torch.Tensor, feasibility_candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean CDF [N,B], member CDFs [M,N,B] and residual [N] for encoded pairs."""
        pair = self.feasibility_relation(
            torch.stack((feasibility_source, feasibility_candidate), dim=1)
        )
        member_logits = torch.stack([head(pair) for head in self.hazard_heads], dim=0)
        member_cdf = self._member_cdf(member_logits)
        mean_cdf = self._member_cdf(member_logits.mean(dim=0))
        residual = F.softplus(self.residual_head(pair).squeeze(-1))
        return mean_cdf, member_cdf, residual

    def forward(
        self,
        source_feature: torch.Tensor,
        candidate_features: torch.Tensor,
        goal_feature: torch.Tensor,
        valid_candidates: Optional[torch.Tensor] = None,
    ) -> AchievabilityOutput:
        """Score candidate waypoints [B,K,C,H,W] from a source and a goal [B,C,H,W]."""
        batch, candidates = candidate_features.shape[:2]
        flat_candidates = candidate_features.reshape(
            batch * candidates, *candidate_features.shape[2:]
        )
        value_source = self.value_frame_encoder(source_feature)
        feasibility_source = self.feasibility_frame_encoder(source_feature)
        value_goal = self.value_frame_encoder(goal_feature)
        value_candidates = self.value_frame_encoder(flat_candidates).reshape(
            batch, candidates, -1
        )
        feasibility_candidates = self.feasibility_frame_encoder(
            flat_candidates
        ).reshape(batch, candidates, -1)
        width = value_candidates.shape[-1]
        if valid_candidates is None:
            valid_candidates = torch.ones(
                (batch, candidates), dtype=torch.bool, device=value_candidates.device
            )
        else:
            valid_candidates = valid_candidates.to(
                device=value_candidates.device, dtype=torch.bool
            )

        source_for_candidates = value_source[:, None, :].expand(-1, candidates, -1)
        goal_for_candidates = value_goal[:, None, :].expand(-1, candidates, -1)
        value_tokens = torch.stack(
            (source_for_candidates, value_candidates, goal_for_candidates), dim=2
        ).reshape(batch * candidates, 3, width)
        value_pair = self.value_relation(value_tokens).reshape(batch, candidates, width)
        set_context = self.candidate_set_encoder(
            value_pair, src_key_padding_mask=~valid_candidates
        )
        value_pair = self.candidate_set_norm(value_pair + set_context)

        feasibility_source = feasibility_source[:, None, :].expand(-1, candidates, -1)
        feasibility_tokens = torch.stack(
            (feasibility_source, feasibility_candidates), dim=2
        ).reshape(batch * candidates, 2, width)
        feasibility_pair = self.feasibility_relation(feasibility_tokens).reshape(
            batch, candidates, width
        )
        hazard_member_logits = torch.stack(
            [head(feasibility_pair) for head in self.hazard_heads], dim=0
        )
        hit_cdf_members = self._member_cdf(hazard_member_logits)
        hazard_logits = hazard_member_logits.mean(dim=0)
        hit_cdf = self._member_cdf(hazard_logits)
        utility_members = torch.stack(
            [head(value_pair).squeeze(-1) for head in self.utility_heads], dim=0
        )
        return AchievabilityOutput(
            hazard_logits=hazard_logits,
            hit_cdf=hit_cdf,
            expected_hit_time=1.0 + (1.0 - hit_cdf).sum(dim=-1),
            predicted_log_residual=F.softplus(
                self.residual_head(feasibility_pair).squeeze(-1)
            ),
            predicted_progress=torch.tanh(self.progress_head(value_pair).squeeze(-1)),
            waypoint_utility=utility_members.mean(dim=0),
            pair_embedding=value_pair,
            hazard_member_logits=hazard_member_logits,
            hit_cdf_members=hit_cdf_members,
            utility_members=utility_members,
            feasibility_uncertainty=hit_cdf_members.std(dim=0, unbiased=False)[..., -1],
            value_uncertainty=utility_members.std(dim=0, unbiased=False),
        )
