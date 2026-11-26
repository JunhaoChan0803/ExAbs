"""Calibrate AM probabilities on controller outcomes from reserved layouts."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def fit_calibration(probabilities, uncertainty, hits, uncertainty_weight=0.25):
    # Preserve the optimizer's finite-difference steps in the loss calculation.
    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    h = np.asarray(hits)
    if (
        p.ndim != 2
        or p.shape[1] != 30
        or u.shape != p.shape
        or h.shape != (len(p),)
        or not len(p)
    ):
        raise ValueError(
            "Expected nonempty AM probabilities [N,30], uncertainty and hit times"
        )
    y = (h[:, None] >= 1) & (h[:, None] <= np.arange(1, 31)[None])
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    alpha, beta = [], []
    for budget in range(30):

        def loss(v, budget=budget):
            z = x[:, budget] * np.exp(v[0]) + v[1]
            return np.mean(np.logaddexp(0, z) - y[:, budget] * z)

        result = minimize(
            loss, [0.0, 0.0], method="L-BFGS-B", bounds=[(-4, 4), (-15, 15)]
        )
        if not result.success:
            raise RuntimeError(f"AM calibration failed: {result.message}")
        alpha.append(float(np.exp(result.x[0])))
        beta.append(float(result.x[1]))
    q = expit(alpha[9] * x[:, 9] + beta[9])
    rho = np.clip(q - uncertainty_weight * u[:, 9], 0, 1)
    # Threshold: the largest descending-score prefix (at least 64 records) whose
    # one-sided 95% Wilson upper bound on the failure rate is at most 0.2. If no
    # prefix qualifies, the threshold lies above every score and the barrier term
    # of the planner applies to all candidates.
    order = np.argsort(-rho, kind="stable")
    bad = (~y[order, 9]).cumsum()
    threshold, accepted = 1.000001, 0
    z = 1.6448536269514722
    for count in range(64, len(order) + 1):
        if count < len(order) and rho[order[count - 1]] == rho[order[count]]:
            continue
        rate = bad[count - 1] / count
        upper = (
            rate
            + z * z / (2 * count)
            + z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
        ) / (1 + z * z / count)
        if upper <= 0.2:
            threshold, accepted = float(rho[order[count - 1]]), count
    return dict(
        alpha=alpha,
        beta=beta,
        threshold10=threshold,
        accepted10=accepted,
        online_uncertainty_beta=uncertainty_weight,
    )
