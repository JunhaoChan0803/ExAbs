from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    ModelConfig,
)


def _require_shape(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if value.ndim != len(expected):
        raise ValueError(
            f"{name} must have {len(expected)} dimensions, got {value.shape}"
        )
    for axis, (actual, wanted) in enumerate(zip(value.shape, expected)):
        if wanted >= 0 and actual != wanted:
            raise ValueError(
                f"{name} axis {axis} must have size {wanted}, got {actual}"
            )


def _validate_candidate_mask(
    mask: torch.Tensor, batch: int, count: int
) -> torch.Tensor:
    _require_shape(mask, (batch, count), "candidate_mask")
    if mask.dtype is not torch.bool:
        raise TypeError("candidate_mask must have dtype torch.bool")
    if not torch.all(mask.any(dim=-1)):
        raise ValueError(
            "every candidate set must contain at least one valid candidate"
        )
    return mask


class CubeCLSAdapter(nn.Module):
    """Map a Cube CLS latent into the relation space of AM and PETAL."""

    def __init__(self, latent_dim: int, model_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Cube CLS input must end in {self.latent_dim}, got {latent.shape}"
            )
        if not torch.is_floating_point(latent):
            raise TypeError("Cube CLS input must be floating point")
        return self.network(latent)


class DurationAgnosticToken(nn.Module):
    """A learned relation-role token independent of physical duration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, config.duration_dim))
        self.network = nn.Sequential(
            nn.Linear(config.duration_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        nn.init.normal_(self.token, std=0.02)

    def forward(self, count: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("duration-agnostic token count must be positive")
        return self.network(self.token).expand(count, -1)


class RoleRelationTransformer(nn.Module):
    """Reason over a fixed set of explicitly role-labelled tokens."""

    def __init__(self, config: ModelConfig, token_count: int) -> None:
        super().__init__()
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        self.token_count = token_count
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.model_dim))
        self.role_embedding = nn.Parameter(
            torch.empty(1, token_count + 1, config.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.relation_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def forward(self, *role_tokens: torch.Tensor) -> torch.Tensor:
        if len(role_tokens) != self.token_count:
            raise ValueError(
                f"expected {self.token_count} role tokens, got {len(role_tokens)}"
            )
        reference_shape = role_tokens[0].shape
        if len(reference_shape) != 2:
            raise ValueError("relation role tokens must have shape [N, D]")
        if any(token.shape != reference_shape for token in role_tokens):
            raise ValueError("all relation role tokens must have the same shape")
        tokens = torch.stack(role_tokens, dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.role_embedding)
        return encoded[:, 0]


class CandidateSetTransformer(nn.Module):
    """Permutation-equivariant candidate context with an explicit set mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.candidate_set_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self, candidates: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        _require_shape(
            valid_mask,
            (candidates.shape[0], candidates.shape[1]),
            "candidate_mask",
        )
        contextual = self.encoder(
            candidates,
            src_key_padding_mask=~valid_mask,
        )
        return contextual.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


@dataclass
class AMFeasibilityMemberOutput:
    hazard_logits: torch.Tensor
    cdf: torch.Tensor
    terminal_residual: torch.Tensor


@dataclass
class AMMemberOutput(AMFeasibilityMemberOutput):
    progress: torch.Tensor
    utility: torch.Tensor


@dataclass
class AMFeasibilityEnsembleOutput:
    """Feasibility-only ensemble output with post-hoc budget queries."""

    member_hazard_logits: torch.Tensor
    member_cdf: torch.Tensor
    member_terminal_residual: torch.Tensor
    mean_cdf: torch.Tensor
    cdf_epistemic_std: torch.Tensor
    mean_terminal_residual: torch.Tensor
    terminal_residual_epistemic_std: torch.Tensor
    candidate_mask: torch.Tensor
    hazard_bin_steps: tuple[int, ...]

    def _budget_indices(self, budget_steps: int | float | torch.Tensor) -> torch.Tensor:
        target_shape = self.mean_cdf.shape[:-1]
        budget = torch.as_tensor(
            budget_steps,
            device=self.mean_cdf.device,
            dtype=self.mean_cdf.dtype,
        )
        if budget.ndim == 0:
            budget = budget.expand(target_shape)
        elif budget.shape == target_shape[:-1]:
            budget = budget.unsqueeze(-1).expand(target_shape)
        else:
            try:
                budget = torch.broadcast_to(budget, target_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"budget shape {budget.shape} is not broadcastable to {target_shape}"
                ) from error
        bins = torch.as_tensor(
            self.hazard_bin_steps,
            device=budget.device,
            dtype=budget.dtype,
        )
        matches = torch.isclose(budget.unsqueeze(-1), bins, rtol=0.0, atol=1.0e-6)
        if not torch.all(matches.any(dim=-1)):
            allowed = ", ".join(str(step) for step in self.hazard_bin_steps)
            raise ValueError(f"budget must exactly match a CDF bin: [{allowed}]")
        return matches.to(torch.int64).argmax(dim=-1, keepdim=True)


@dataclass
class AMEnsembleOutput(AMFeasibilityEnsembleOutput):
    """Full five-member feasibility and downstream-value output."""

    member_progress: torch.Tensor
    member_utility: torch.Tensor
    mean_progress: torch.Tensor
    mean_utility: torch.Tensor
    utility_epistemic_std: torch.Tensor


class _AchievabilityMember(nn.Module):
    """One fully independent member of the deep ensemble."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.model_dim

        # Feasibility owns all parameters below and never sees the goal.
        self.feasibility_adapter = CubeCLSAdapter(config.latent_dim, d)
        self.feasibility_duration_role = DurationAgnosticToken(config)
        self.executor_token = nn.Parameter(torch.empty(1, d))
        self.feasibility_relation = RoleRelationTransformer(config, token_count=4)
        self.hazard_head = nn.Linear(d, len(config.hazard_bin_steps))
        self.terminal_residual_head = nn.Linear(d, 1)

        # Value has a disjoint adapter/relation/set-context graph.
        self.value_adapter = CubeCLSAdapter(config.latent_dim, d)
        self.value_duration_role = DurationAgnosticToken(config)
        self.value_relation = RoleRelationTransformer(config, token_count=4)
        self.candidate_set = CandidateSetTransformer(config)
        self.progress_head = nn.Linear(d, 1)
        self.utility_head = nn.Linear(d, 1)
        nn.init.normal_(self.executor_token, std=0.02)

    def forward_feasibility(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        *,
        candidate_mask: torch.Tensor,
    ) -> AMFeasibilityMemberOutput:
        batch, count, _ = candidates.shape
        flat_source = (
            source[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
        )
        flat_candidate = candidates.reshape(batch * count, -1)

        feasibility = self.feasibility_relation(
            self.feasibility_adapter(flat_source),
            self.feasibility_adapter(flat_candidate),
            self.feasibility_duration_role(batch * count),
            self.executor_token.expand(batch * count, -1),
        ).reshape(batch, count, -1)
        hazard_logits = self.hazard_head(feasibility)
        hazards = torch.sigmoid(hazard_logits)
        cdf = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        terminal_residual = F.softplus(
            self.terminal_residual_head(feasibility).squeeze(-1)
        )

        invalid = ~candidate_mask
        return AMFeasibilityMemberOutput(
            hazard_logits=hazard_logits.masked_fill(invalid.unsqueeze(-1), 0.0),
            cdf=cdf.masked_fill(invalid.unsqueeze(-1), 0.0),
            terminal_residual=terminal_residual.masked_fill(invalid, 0.0),
        )

    def forward(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        goal: torch.Tensor,
        *,
        candidate_mask: torch.Tensor,
    ) -> AMMemberOutput:
        batch, count, _ = candidates.shape
        flat_source = (
            source[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
        )
        flat_candidate = candidates.reshape(batch * count, -1)
        flat_goal = goal[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
        feasibility = self.forward_feasibility(
            source, candidates, candidate_mask=candidate_mask
        )

        value_relation = self.value_relation(
            self.value_adapter(flat_source),
            self.value_adapter(flat_candidate),
            self.value_adapter(flat_goal),
            self.value_duration_role(batch * count),
        ).reshape(batch, count, -1)
        contextual_value = self.candidate_set(value_relation, candidate_mask)
        progress = self.progress_head(contextual_value).squeeze(-1)
        utility = self.utility_head(contextual_value).squeeze(-1)

        invalid = ~candidate_mask
        return AMMemberOutput(
            hazard_logits=feasibility.hazard_logits,
            cdf=feasibility.cdf,
            terminal_residual=feasibility.terminal_residual,
            progress=progress.masked_fill(invalid, 0.0),
            utility=utility.masked_fill(invalid, 0.0),
        )


class AchievabilityModel(nn.Module):
    """Five independent AMs with separate completion and value pathways."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.members = nn.ModuleList(
            _AchievabilityMember(self.config)
            for _ in range(self.config.ensemble_members)
        )

    def forward(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        goal: torch.Tensor,
        *,
        candidate_mask: torch.Tensor,
    ) -> AMEnsembleOutput:
        _require_shape(source, (-1, self.config.latent_dim), "source")
        batch = source.shape[0]
        _require_shape(candidates, (batch, -1, self.config.latent_dim), "candidates")
        count = candidates.shape[1]
        _require_shape(goal, (batch, self.config.latent_dim), "goal")
        candidate_mask = _validate_candidate_mask(candidate_mask, batch, count)
        if candidates.device != source.device or goal.device != source.device:
            raise ValueError("source, candidates, and goal must be on the same device")

        outputs = [
            member(
                source,
                candidates,
                goal,
                candidate_mask=candidate_mask,
            )
            for member in self.members
        ]
        hazard_logits = torch.stack([item.hazard_logits for item in outputs])
        cdf = torch.stack([item.cdf for item in outputs])
        residual = torch.stack([item.terminal_residual for item in outputs])
        progress = torch.stack([item.progress for item in outputs])
        utility = torch.stack([item.utility for item in outputs])
        return AMEnsembleOutput(
            member_hazard_logits=hazard_logits,
            member_cdf=cdf,
            member_terminal_residual=residual,
            member_progress=progress,
            member_utility=utility,
            mean_cdf=cdf.mean(dim=0),
            cdf_epistemic_std=cdf.std(dim=0, unbiased=False),
            mean_terminal_residual=residual.mean(dim=0),
            terminal_residual_epistemic_std=residual.std(dim=0, unbiased=False),
            mean_progress=progress.mean(dim=0),
            mean_utility=utility.mean(dim=0),
            utility_epistemic_std=utility.std(dim=0, unbiased=False),
            candidate_mask=candidate_mask,
            hazard_bin_steps=self.config.hazard_bin_steps,
        )

    def forward_feasibility(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        *,
        candidate_mask: torch.Tensor,
    ) -> AMFeasibilityEnsembleOutput:
        """Run feasibility scoring without the goal-conditioned value branch."""

        _require_shape(source, (-1, self.config.latent_dim), "source")
        batch = source.shape[0]
        _require_shape(candidates, (batch, -1, self.config.latent_dim), "candidates")
        count = candidates.shape[1]
        candidate_mask = _validate_candidate_mask(candidate_mask, batch, count)
        if candidates.device != source.device:
            raise ValueError("source and candidates must be on the same device")
        outputs = [
            member.forward_feasibility(
                source, candidates, candidate_mask=candidate_mask
            )
            for member in self.members
        ]
        hazard_logits = torch.stack([item.hazard_logits for item in outputs])
        cdf = torch.stack([item.cdf for item in outputs])
        residual = torch.stack([item.terminal_residual for item in outputs])
        return AMFeasibilityEnsembleOutput(
            member_hazard_logits=hazard_logits,
            member_cdf=cdf,
            member_terminal_residual=residual,
            mean_cdf=cdf.mean(dim=0),
            cdf_epistemic_std=cdf.std(dim=0, unbiased=False),
            mean_terminal_residual=residual.mean(dim=0),
            terminal_residual_epistemic_std=residual.std(dim=0, unbiased=False),
            candidate_mask=candidate_mask,
            hazard_bin_steps=self.config.hazard_bin_steps,
        )


class PositiveDefiniteMetric(nn.Module):
    """Learn ``diag(positive) + U U^T`` without constructing a dense matrix."""

    def __init__(self, effect_dim: int, rank: int, diagonal_floor: float) -> None:
        super().__init__()
        self.diagonal_floor = float(diagonal_floor)
        self.raw_diagonal = nn.Parameter(torch.zeros(effect_dim))
        self.factor = nn.Parameter(torch.empty(effect_dim, rank))
        nn.init.normal_(self.factor, std=0.02 / math.sqrt(max(rank, 1)))

    @property
    def diagonal(self) -> torch.Tensor:
        return F.softplus(self.raw_diagonal) + self.diagonal_floor

    def matrix(self) -> torch.Tensor:
        return torch.diag(self.diagonal) + self.factor @ self.factor.transpose(0, 1)

    def forward(self, difference: torch.Tensor) -> torch.Tensor:
        diagonal_term = (difference.square() * self.diagonal).sum(dim=-1)
        low_rank_term = (difference @ self.factor).square().sum(dim=-1)
        return diagonal_term + low_rank_term


@dataclass
class PETALOutput:
    target_effect: torch.Tensor
    candidate_effect: torch.Tensor
    effect_cost: torch.Tensor
    bounded_residual: torch.Tensor | None = None
    anchored_cost: torch.Tensor | None = None
