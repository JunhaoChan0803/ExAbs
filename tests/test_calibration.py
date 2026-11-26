import numpy as np

from counter_flow.calibration import fit_calibration


def _records(seed=11, n=512):
    rng = np.random.default_rng(seed)
    probabilities = np.tile(np.linspace(0.02, 0.4, 30), (n, 1))
    hits = np.where(rng.random(n) < 0.8, 5, -1)
    return probabilities, np.zeros_like(probabilities), hits


def test_float32_matches_float64():
    p, u, h = _records()
    a = fit_calibration(p.astype(np.float32), u.astype(np.float32), h)
    b = fit_calibration(p, u, h)
    np.testing.assert_allclose(a["alpha"], b["alpha"], rtol=1e-4)
    np.testing.assert_allclose(a["beta"], b["beta"], rtol=1e-4, atol=1e-4)


def test_miscalibrated_scores_are_corrected():
    p, u, h = _records()
    result = fit_calibration(p.astype(np.float32), u.astype(np.float32), h)
    assert abs(result["alpha"][9] - 1.0) > 0.1
    assert abs(result["beta"][9]) > 0.1
