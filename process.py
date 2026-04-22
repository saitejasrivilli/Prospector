"""
process.py — Synthetic noisy manufacturing process function.

Simulates a black-box process (e.g., semiconductor fab) where:
  - Inputs: 3–5 "recipe" parameters (e.g., temperature, pressure, time, flow, power)
  - Output: a noisy scalar "metrology" reading (e.g., etch rate, film thickness)

The true underlying function is a sum of Gaussians (multimodal) so the
optimizer has to work past local optima — just like real fab processes.
"""

import numpy as np


# ── Domain ────────────────────────────────────────────────────────────────────
# Each parameter lives in [0, 1] (normalised units).
# In a real setting you'd scale from physical units (°C, Torr, sccm …).
N_PARAMS = 5
BOUNDS = np.array([[0.0, 1.0]] * N_PARAMS)   # shape (N_PARAMS, 2)

# True optima (planted so we can benchmark convergence)
_PEAKS = np.array([
    [0.2, 0.8, 0.5, 0.3, 0.7],   # global optimum
    [0.7, 0.3, 0.2, 0.8, 0.4],   # local optimum (slightly lower)
    [0.5, 0.5, 0.8, 0.1, 0.9],   # another local peak
])
_WEIGHTS = np.array([1.0, 0.80, 0.65])        # peak heights
_WIDTHS  = np.array([0.18, 0.15, 0.12])       # Gaussian widths


def true_function(x: np.ndarray) -> float:
    """
    Noiseless ground-truth objective (never seen by the optimizer).

    Parameters
    ----------
    x : array-like, shape (N_PARAMS,)
        Normalised recipe parameters in [0, 1].

    Returns
    -------
    float
        Scalar output (higher is better — think yield or target thickness).
    """
    x = np.asarray(x, dtype=float)
    value = 0.0
    for peak, weight, width in zip(_PEAKS, _WEIGHTS, _WIDTHS):
        diff   = x - peak
        value += weight * np.exp(-np.dot(diff, diff) / (2 * width ** 2))
    return float(value)


def observe(x: np.ndarray, noise_std: float = 0.05, rng: np.random.Generator | None = None) -> float:
    """
    Noisy observation — what the metrology tool actually returns.

    Parameters
    ----------
    x         : array-like, shape (N_PARAMS,)
    noise_std : standard deviation of Gaussian measurement noise
    rng       : optional numpy Generator for reproducibility

    Returns
    -------
    float
        Noisy scalar output.
    """
    if rng is None:
        rng = np.random.default_rng()
    return true_function(x) + rng.normal(0.0, noise_std)


def random_sample(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Draw *n* uniformly random points from the parameter space.

    Returns
    -------
    np.ndarray, shape (n, N_PARAMS)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n, N_PARAMS))


GLOBAL_OPTIMUM_X   = _PEAKS[0]
GLOBAL_OPTIMUM_VAL = true_function(_PEAKS[0])
