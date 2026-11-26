"""Two-room training objectives for prepared feature, action, and execution batches."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from exabs.losses import (
    PlannerSelectionBatch,
    PlannerSelectionKind,
    discrete_survival_nll,
    effect_anti_collapse_loss,
    masked_standardized_mse,
    selection_distillation_loss,
)


def exact_partition(edges, segments=5, minimum=10, maximum=25):
    """Maximize AM edge utility over a fixed-span, fixed-count partition.

    Edges [T+1,T+1] use -inf for invalid transitions; returns boundaries and utility.
    """
    end = edges.shape[0] - 1
    value = np.full((segments + 1, end + 1), -np.inf)
    previous = np.full((segments + 1, end + 1), -1, int)
    value[0, 0] = 0.0
    for s in range(1, segments + 1):
        for j in range(s * minimum, min(end, s * maximum) + 1):
            for gap in range(minimum, maximum + 1):
                i = j - gap
                if i < 0:
                    continue
                score = value[s - 1, i] + edges[i, j]
                if score > value[s, j]:
                    value[s, j] = score
                    previous[s, j] = i
    if not np.isfinite(value[segments, end]):
        raise RuntimeError("No feasible finite AM partition")
    out = [end]
    j = end
    for s in range(segments, 0, -1):
        j = int(previous[s, j])
        out.append(j)
    return np.asarray(out[::-1]), float(value[segments, end])


def high_loss(high, low, batch, weight):
    """Feature, motion-region, rollout, and weighted position losses for L2.

    Batch: z [B,N+1,16,43,43], actions [B,N,25,2], mask [B,N,25], roi [B,N,43,43].
    L1 stays frozen. Returns (total, feature_and_motion_loss, position_loss).
    """
    z, a, mask, roi = batch
    b, n = a.shape[:2]
    source, target = z[:, :-1], z[:, 1:]
    macro = high.encode_actions(a.flatten(0, 1), mask.flatten(0, 1)).reshape(b, n, 8)
    teacher = high.step(source.flatten(0, 1), macro.flatten(0, 1)).reshape_as(target)
    error = (teacher.float() - target).square().mean(2)
    glob = error.mean()
    motion = (error * roi).sum() / roi.sum().clamp_min(1.0)
    with torch.no_grad():
        tp = low.position(target.flatten(0, 1)).reshape(b, n, 2)
    pos = (low.position(teacher.flatten(0, 1)).reshape(b, n, 2) - tp).square().mean()
    pred = z[:, 0]
    rolled = 0.0
    rp = 0.0
    for j in range(n):
        pred = high.step(pred, macro[:, j])
        rolled += F.mse_loss(pred.float(), target[:, j]) / n
        rp += (low.position(pred) - tp[:, j]).square().mean() / n
    base = glob + 2 * motion + 0.25 * rolled
    position = pos + rp
    return base + weight * position, base, position


def head_mask(group_index: torch.Tensor, member: int) -> torch.Tensor:
    """Each ensemble head is supervised on a fixed two-thirds of the source groups."""
    return (group_index.to(torch.long) * 17 + member * 11) % 3 != 0


def hazard_loss(logits, members, hit, groups):
    hit = hit[:, None]
    valid = torch.ones_like(hit, dtype=torch.bool)
    reached = hit >= 1
    survival = discrete_survival_nll(
        logits[:, None].float(),
        first_hit_step=hit,
        reached=reached,
        valid_candidates=valid,
    )
    cdf = 1 - torch.cumprod((1 - torch.sigmoid(logits.float())).clamp_min(1e-7), -1)
    times = torch.arange(1, 26, device=logits.device)[None]
    truth = ((hit >= 1) & (hit <= times)).float()
    brier = (cdf - truth).square().mean()
    terms = []
    for m in range(3):
        mask = head_mask(groups, m)[:, None]
        if mask.any():
            terms.append(
                discrete_survival_nll(
                    members[m, :, None].float(),
                    first_hit_step=hit,
                    reached=reached,
                    valid_candidates=mask,
                )
            )
    bootstrap = torch.stack(terms).mean() if terms else survival * 0
    return (
        1.25 * survival + 0.4 * brier + 0.3 * bootstrap,
        dict(survival=float(survival.detach()), brier=float(brier.detach())),
    )


def am_forward(model, source, target):
    """Mean and per-head hazard logits of the completion branch."""
    a = model.encode_feasibility_frames(source)
    b = model.encode_feasibility_frames(target)
    pair = model.feasibility_relation(torch.stack([a, b], 1))
    logits = torch.stack([head(pair) for head in model.hazard_heads])
    return logits.mean(0), logits


def am_loss(model, source, target, first_hit, groups):
    """Survival likelihood, integrated Brier loss, and grouped ensemble loss.

    source/target: [B,16,43,43]; first_hit: [B], values 1..25 or -1 for
    a censored attempt; groups: [B] stable integer trajectory-group IDs.
    """
    logits, members = am_forward(model, source.detach(), target.detach())
    return hazard_loss(logits, members, first_hit, groups)


def petal_loss(model, selector, batch, *, temperature=0.0025):
    """Distill decisions over a fixed candidate pool.

    Source/target: [B,C,H,W]; endpoints: [B,K,C,H,W]; actions: [B,K,15,2].
    Costs/probabilities: [B,K]; nominal/update: [B,15,2].
    """
    device = next(model.parameters()).device
    values = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in batch.items()
    }
    source, target, endpoints = (
        values[key] for key in ("source", "target", "endpoints")
    )
    b = len(source)
    valid = torch.ones_like(values["costs"], dtype=torch.bool)
    selection = PlannerSelectionBatch(
        kind=PlannerSelectionKind.MPPI_SOFT,
        candidate_actions=values["actions"],
        teacher_costs=values["costs"],
        teacher_probabilities=values["probabilities"],
        candidate_valid_mask=valid,
        nominal_actions=values["nominal"],
        teacher_control_update=values["update"],
    )
    with torch.no_grad():
        support = selector.hard_support(source, target, 0.2)
    duration = source.new_full((b,), 15.0)
    context = model.build_target(
        source_feature=source,
        target_feature=target,
        target_support=support,
        requested_duration=duration,
    )
    model.base_costs = values["costs"]
    try:
        scores = model.score_candidate_endpoints(
            source_features=source,
            candidate_endpoints=endpoints,
            target=context,
            achieved_duration=duration,
            candidate_chunk_size=16,
        )
    finally:
        model.base_costs = None
    selection_loss = selection_distillation_loss(
        batch=selection,
        student_costs=scores.candidate_costs.float(),
        temperature=temperature,
    )
    terminal_cost = (endpoints - target[:, None]).square().mean(dim=(-3, -2, -1))
    calibration = 0.1 * masked_standardized_mse(
        scores.candidate_costs.float(), terminal_cost, valid
    )
    tokens = torch.cat(
        [context.effect_token[:, None], scores.candidate_effects], dim=1
    ).float()
    token_mask = torch.cat(
        [torch.ones(b, 1, device=device, dtype=torch.bool), valid], dim=1
    )
    regularization = effect_anti_collapse_loss(tokens, valid_mask=token_mask)
    total = selection_loss.total + calibration + regularization.total
    return total, dict(
        selection=selection_loss.total.detach(),
        calibration=calibration.detach(),
        regularization=regularization.total.detach(),
    )


def configure_training(models, component):
    """Unfreeze one component for training and keep all others frozen."""
    if component not in {"high", "am", "petal"}:
        raise ValueError("Choose high, am, or petal")
    for model in models.values():
        model.eval().requires_grad_(False)
    selected = models[component]
    selected.train()
    if component == "am":
        selected.feasibility_relation.local.requires_grad_(True)
        selected.feasibility_relation.project.requires_grad_(True)
        for head in selected.hazard_heads:
            head.correction.requires_grad_(True)
    else:
        selected.requires_grad_(True)
    return selected


def make_optimizer(model, component, *, learning_rate):
    """Create the optimizer for one training stage with an explicit step size."""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Call configure_training before creating the optimizer")
    if component == "high":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if component not in {"am", "petal"}:
        raise ValueError("Choose high, am, or petal")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)


def optimizer_step(model, optimizer, loss, *, max_gradient_norm=5.0):
    """Backpropagate, clip the gradient, and apply one optimizer update."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite training gradient")
    optimizer.step()
    return float(norm.detach())
