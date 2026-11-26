"""AM, PETAL, and L2 objectives for prepared trajectory and execution batches."""

import torch
from torch import nn
from torch.nn import functional as F

from exabs.losses import (
    PlannerSelectionBatch,
    PlannerSelectionKind,
    discrete_survival_nll,
    effect_anti_collapse_loss,
    masked_standardized_mse,
    selection_distillation_loss,
)

from .models import AM, petal_model


def head_mask(groups, head):
    """Each ensemble head is supervised on a fixed two-thirds of the source groups."""
    return (17 * groups + 11 * head) % 3 != 0


def objective(logits, hit, groups):
    agg = logits.mean(0)
    valid = torch.ones((len(hit), 1), device=hit.device, dtype=torch.bool)
    nll = discrete_survival_nll(
        agg[:, None],
        first_hit_step=hit[:, None],
        reached=(hit >= 1)[:, None],
        valid_candidates=valid,
    )
    y = (
        (hit[:, None] >= 1)
        & (hit[:, None] <= torch.arange(1, 31, device=hit.device)[None])
    ).float()
    brier = (AM.cdf(agg) - y).square().mean()
    terms = []
    for i in range(3):
        terms.append(
            discrete_survival_nll(
                logits[i, :, None],
                first_hit_step=hit[:, None],
                reached=(hit >= 1)[:, None],
                valid_candidates=head_mask(groups, i)[:, None],
            )
        )
    return 1.25 * nll + 0.4 * brier + 0.3 * torch.stack(terms).mean()


def am_loss(logits, h, g, strata, cfg):
    """Survival and Brier terms plus same-gap ranking at the 10-step budget."""
    loss = objective(logits, h, g)
    prob = AM.cdf(logits.mean(0))[:, 9]
    y = (h >= 1) & (h <= 10)
    loss = loss + cfg["am_brier10"] * (prob - y.float()).square().mean()
    terms = []
    logit = torch.logit(prob.clamp(1e-6, 1 - 1e-6))
    for v in torch.unique(strata):
        pos = logit[(strata == v) & y]
        neg = logit[(strata == v) & ~y]
        if len(pos) and len(neg):
            terms.append(F.softplus(-(pos[:, None] - neg[None])).mean())
    if terms:
        loss = loss + cfg["am_ranking"] * torch.stack(terms).mean()
    return loss


def high_loss(pred, target, low):
    """Fit predicted endpoints to frozen observed feature targets."""
    target = target.detach()
    p = low.position(pred.flatten(0, 1).float())
    with torch.no_grad():
        q = low.position(target.flatten(0, 1))
    return (pred.float() - target).square().mean() + 25 * (p - q).square().mean()


def petal_cost(petal, source, target, endpoints, support):
    """PETAL costs over the full candidate pool, plus target and candidate effects."""
    duration = source.new_full((len(source),), 15.0)
    effect_target = petal.build_target(
        source_feature=source,
        target_feature=target,
        target_support=support,
        requested_duration=duration,
    )
    result = petal.score_candidate_endpoints(
        source_features=source,
        candidate_endpoints=endpoints,
        target=effect_target,
        achieved_duration=duration,
        candidate_chunk_size=32,
    )
    tokens = torch.cat(
        [effect_target.effect_token[:, None], result.candidate_effects], 1
    ).float()
    return result.candidate_costs, tokens


class PETALTraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.petal = petal_model()

    def forward(self, batch, support):
        source, target, endpoints, actions, costs, weights, nominal, update = batch
        valid = torch.ones_like(costs, dtype=torch.bool)
        selection = PlannerSelectionBatch(
            kind=PlannerSelectionKind.MPPI_SOFT,
            candidate_actions=actions,
            teacher_costs=costs,
            teacher_probabilities=weights,
            candidate_valid_mask=valid,
            nominal_actions=nominal,
            teacher_control_update=update,
        )
        student, tokens = petal_cost(self.petal, source, target, endpoints, support)
        selected = selection_distillation_loss(
            batch=selection, student_costs=student.float(), temperature=0.0025
        )
        terminal = (endpoints - target[:, None]).square().flatten(2).mean(-1)
        calibration = 0.1 * masked_standardized_mse(student.float(), terminal, valid)
        total = selected.total + calibration + effect_anti_collapse_loss(tokens).total
        return (
            total,
            torch.stack(
                [selected.probability_kl.detach(), selected.control_update.detach()]
            ),
        )


def optimizer_step(model, optimizer, loss, max_gradient_norm=5.0):
    """Apply one update to the selected trainable component."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite training gradient")
    optimizer.step()
    return float(norm)
