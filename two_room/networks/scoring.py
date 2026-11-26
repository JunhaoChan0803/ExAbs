"""Calibrated AM risk and the Two-room PETAL score on the executor path cost."""

import torch

from exabs.petal import PETALCore


def calibrated(p, cal):
    a = torch.as_tensor(cal["alpha"], device=p.device, dtype=p.dtype)
    b = torch.as_tensor(cal["beta"], device=p.device, dtype=p.dtype)
    return torch.sigmoid(a * torch.logit(p.clamp(1e-6, 1 - 1e-6)) + b)


def risk(p, m, cal, budget, seg=False):
    cp = calibrated(p, cal)
    cm = calibrated(m, cal)
    idx = budget.long() - 1
    mean = cp.gather(-1, idx[..., None])[..., 0]
    members = (cm if seg else m).gather(
        -1, idx[None, ..., None].expand(m.shape[0], *idx.shape, 1)
    )[..., 0]
    u = members.std(0, unbiased=False)
    rho = (mean - u).clamp(0, 1)
    return rho.clamp_min(1e-6), rho, u


class AnchoredPETAL(PETALCore):
    """Anchor the learned effect distance on the executor's path cost.

    Callers set ``base_costs`` [B,K] for the candidate pool before scoring.
    """

    base_costs = None

    def _executor_anchored_cost(self, *, candidate_endpoints, target, learned_costs):
        base = self.base_costs
        if base is None or base.shape != learned_costs.shape:
            raise RuntimeError("Set base_costs for the candidate pool before scoring")
        return aligned_fusion(base, learned_costs)


def aligned_fusion(base, learned, weight=0.1):
    z = (
        (learned - learned.mean(1, keepdim=True))
        / learned.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    ).clamp(-2, 2)
    rescaled = (
        base.mean(1, keepdim=True)
        + base.std(1, keepdim=True, unbiased=False).clamp_min(1e-6) * z
    )
    return ((1 - weight) * base + weight * rescaled).clamp_min(0)
