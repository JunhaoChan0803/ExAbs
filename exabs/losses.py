"""Training objectives shared by the AM and PETAL learners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch.nn import functional as F


def discrete_survival_nll(
    hazard_logits: torch.Tensor,
    *,
    first_hit_step: torch.Tensor,
    reached: torch.Tensor,
    valid_candidates: torch.Tensor,
) -> torch.Tensor:
    """Hazard logits [B,K,H]; first hits in 1..H, or -1 for attempts censored at H."""
    horizon = hazard_logits.shape[-1]
    hit = first_hit_step.to(torch.long)
    success = reached.to(torch.bool)
    valid = valid_candidates.to(torch.bool)
    step = torch.arange(1, horizon + 1, device=hazard_logits.device)
    at_risk = step.view(1, 1, -1) <= torch.where(
        success, hit, torch.full_like(hit, horizon)
    ).unsqueeze(-1)
    event = success.unsqueeze(-1) & (step.view(1, 1, -1) == hit.unsqueeze(-1))
    element = F.binary_cross_entropy_with_logits(
        hazard_logits, event.to(hazard_logits.dtype), reduction="none"
    )
    mask = at_risk & valid.unsqueeze(-1)
    denominator = mask.sum().clamp_min(1)
    return (element * mask.to(element.dtype)).sum() / denominator


def masked_standardized_mse(student, teacher, valid_mask):
    """Mean over queries of the MSE between per-query standardized costs."""
    rows = []
    epsilon = torch.finfo(student.dtype).eps
    for index in range(student.shape[0]):
        valid = valid_mask[index]
        student_row = student[index, valid]
        teacher_row = teacher[index, valid]
        if student_row.numel() <= 1:
            rows.append(student_row.new_zeros(()))
            continue
        student_row = (student_row - student_row.mean()) / student_row.std(
            unbiased=False
        ).clamp_min(epsilon)
        teacher_row = (teacher_row - teacher_row.mean()) / teacher_row.std(
            unbiased=False
        ).clamp_min(epsilon)
        rows.append(F.mse_loss(student_row, teacher_row))
    return torch.stack(rows).mean()


class PlannerSelectionKind(str, Enum):
    MPPI_SOFT = "mppi_soft"
    CEM_ELITE = "cem_elite"


@dataclass(frozen=True)
class PlannerSelectionBatch:
    """Teacher decisions over a shared candidate pool.

    Actions are [B,K,T,A]; costs, probabilities and masks are [B,K]; the
    nominal sequence and the teacher's control update are [B,T,A].
    """

    kind: PlannerSelectionKind
    candidate_actions: torch.Tensor
    teacher_costs: torch.Tensor
    teacher_probabilities: torch.Tensor
    candidate_valid_mask: torch.Tensor
    nominal_actions: torch.Tensor
    teacher_control_update: torch.Tensor
    elite_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.kind is PlannerSelectionKind.CEM_ELITE and self.elite_mask is None:
            raise ValueError("CEM supervision requires an elite mask")


@dataclass(frozen=True)
class SelectionDistillationWeights:
    probability_kl: float = 1.0
    control_update: float = 1.0
    elite_ranking: float = 0.25
    elite_distribution: float = 1.0
    elite_fraction: float = 0.10
    elite_temperature: float = 1.0
    epsilon: float = 1e-8


@dataclass(frozen=True)
class SelectionDistillationLoss:
    total: torch.Tensor
    probability_kl: torch.Tensor
    control_update: torch.Tensor
    elite_ranking: torch.Tensor
    elite_distribution: torch.Tensor
    student_probabilities: torch.Tensor
    student_control_update: torch.Tensor


def probabilities_from_costs(
    costs: torch.Tensor, *, valid_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = -costs / float(temperature)
    logits = logits.masked_fill(~valid_mask, -torch.inf)
    return torch.softmax(logits, dim=-1)


def _teacher_elites(teacher_costs, valid_mask, explicit_elites, elite_fraction, index):
    valid_indices = torch.nonzero(valid_mask[index], as_tuple=False).flatten()
    if explicit_elites is None:
        elite_count = max(1, int(round(elite_fraction * len(valid_indices))))
        order = torch.argsort(teacher_costs[index, valid_indices])
        elite_indices = valid_indices[order[:elite_count]]
    else:
        elite_indices = torch.nonzero(
            explicit_elites[index] & valid_mask[index], as_tuple=False
        ).flatten()
    return valid_indices, elite_indices


def _elite_ranking_loss(
    *, teacher_costs, student_costs, valid_mask, explicit_elites, elite_fraction
):
    losses = []
    for index in range(teacher_costs.shape[0]):
        _, elite_indices = _teacher_elites(
            teacher_costs, valid_mask, explicit_elites, elite_fraction, index
        )
        non_elite_mask = valid_mask[index].clone()
        non_elite_mask[elite_indices] = False
        non_elite_indices = torch.nonzero(non_elite_mask, as_tuple=False).flatten()
        if len(elite_indices) == 0 or len(non_elite_indices) == 0:
            losses.append(student_costs[index].new_zeros(()))
            continue
        elite = student_costs[index, elite_indices]
        non_elite = student_costs[index, non_elite_indices]
        losses.append(F.softplus(elite[:, None] - non_elite[None]).mean())
    return torch.stack(losses).mean()


def _elite_distribution_loss(
    *,
    teacher_costs,
    student_costs,
    valid_mask,
    explicit_elites,
    elite_fraction,
    temperature,
    epsilon,
):
    """KL from a uniform distribution over teacher elites to standardized student costs."""
    rows = []
    for index in range(teacher_costs.shape[0]):
        valid_indices, elite_indices = _teacher_elites(
            teacher_costs, valid_mask, explicit_elites, elite_fraction, index
        )
        valid_cost = student_costs[index, valid_indices]
        standardized = (valid_cost - valid_cost.mean()) / valid_cost.std(
            unbiased=False
        ).clamp_min(epsilon)
        log_probability = torch.log_softmax(-standardized / temperature, dim=-1)
        target = torch.isin(valid_indices, elite_indices).to(log_probability.dtype)
        target = target / target.sum()
        target_log = torch.where(target > 0, torch.log(target.clamp_min(epsilon)), 0.0)
        rows.append((target * (target_log - log_probability)).sum())
    return torch.stack(rows).mean()


def selection_distillation_loss(
    *,
    batch: PlannerSelectionBatch,
    student_costs: torch.Tensor,
    temperature: float,
    weights: SelectionDistillationWeights = SelectionDistillationWeights(),
) -> SelectionDistillationLoss:
    """Match teacher probabilities, the full control update, and elite ordering."""
    student_probability = probabilities_from_costs(
        student_costs, valid_mask=batch.candidate_valid_mask, temperature=temperature
    )
    teacher = batch.teacher_probabilities.clamp_min(weights.epsilon)
    student = student_probability.clamp_min(weights.epsilon)
    kl_per_query = torch.where(
        batch.candidate_valid_mask,
        batch.teacher_probabilities * (torch.log(teacher) - torch.log(student)),
        torch.zeros_like(student),
    ).sum(dim=-1)
    probability_kl = kl_per_query.mean()

    deviations = batch.candidate_actions - batch.nominal_actions[:, None]
    student_update = (student_probability[:, :, None, None] * deviations).sum(dim=1)
    control_update = F.mse_loss(student_update, batch.teacher_control_update)
    elite_ranking = _elite_ranking_loss(
        teacher_costs=batch.teacher_costs,
        student_costs=student_costs,
        valid_mask=batch.candidate_valid_mask,
        explicit_elites=batch.elite_mask,
        elite_fraction=float(weights.elite_fraction),
    )
    elite_distribution = _elite_distribution_loss(
        teacher_costs=batch.teacher_costs,
        student_costs=student_costs,
        valid_mask=batch.candidate_valid_mask,
        explicit_elites=batch.elite_mask,
        elite_fraction=float(weights.elite_fraction),
        temperature=float(weights.elite_temperature),
        epsilon=float(weights.epsilon),
    )
    weighted_kl = probability_kl * float(weights.probability_kl)
    weighted_update = control_update * float(weights.control_update)
    weighted_ranking = elite_ranking * float(weights.elite_ranking)
    weighted_elite_distribution = elite_distribution * float(weights.elite_distribution)
    return SelectionDistillationLoss(
        total=(
            weighted_kl
            + weighted_update
            + weighted_ranking
            + weighted_elite_distribution
        ),
        probability_kl=weighted_kl,
        control_update=weighted_update,
        elite_ranking=weighted_ranking,
        elite_distribution=weighted_elite_distribution,
        student_probabilities=student_probability,
        student_control_update=student_update,
    )


@dataclass(frozen=True)
class EffectAntiCollapseConfig:
    """VICReg-style regularization of the effect space."""

    variance_weight: float = 0.10
    covariance_weight: float = 0.01
    target_standard_deviation: float = 1.0
    epsilon: float = 1e-4


@dataclass(frozen=True)
class EffectAntiCollapseLoss:
    total: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    sample_count: int


def effect_anti_collapse_loss(
    effects: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    config: EffectAntiCollapseConfig = EffectAntiCollapseConfig(),
) -> EffectAntiCollapseLoss:
    """Regularize effects [...,D]; ``valid_mask`` covers the leading sample axes."""
    flattened = effects.reshape(-1, effects.shape[-1])
    if valid_mask is not None:
        flattened = flattened[valid_mask.reshape(-1)]
    sample_count = int(flattened.shape[0])
    active = config.variance_weight > 0 or config.covariance_weight > 0
    if active and sample_count < 2:
        raise ValueError("Effect regularization needs at least two valid effects")
    zero = effects.new_zeros(())
    if not active:
        return EffectAntiCollapseLoss(
            total=zero, variance=zero, covariance=zero, sample_count=sample_count
        )

    centered = flattened - flattened.mean(dim=0, keepdim=True)
    if config.variance_weight > 0:
        standard_deviation = torch.sqrt(
            centered.var(dim=0, unbiased=True) + float(config.epsilon)
        )
        variance = torch.relu(
            float(config.target_standard_deviation) - standard_deviation
        ).mean()
    else:
        variance = zero
    if config.covariance_weight > 0 and flattened.shape[-1] > 1:
        covariance_matrix = centered.transpose(0, 1).matmul(centered)
        covariance_matrix = covariance_matrix / float(sample_count - 1)
        diagonal = torch.diagonal(covariance_matrix)
        off_diagonal_square_sum = (
            covariance_matrix.square().sum() - diagonal.square().sum()
        )
        covariance = off_diagonal_square_sum / float(
            flattened.shape[-1] * (flattened.shape[-1] - 1)
        )
    else:
        covariance = zero
    weighted_variance = variance * float(config.variance_weight)
    weighted_covariance = covariance * float(config.covariance_weight)
    return EffectAntiCollapseLoss(
        total=weighted_variance + weighted_covariance,
        variance=weighted_variance,
        covariance=weighted_covariance,
        sample_count=sample_count,
    )
