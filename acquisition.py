"""
acquisition.py — Acquisition functions for Bayesian Optimization.

Implements Expected Improvement (EI) from scratch using the analytic formula.

Why EI?
  EI(x) = E[max(f(x) - f*, 0)]

  where f* is the current best observed value.  It naturally balances:
    - Exploitation: points where the mean is high.
    - Exploration:  points where uncertainty is high.

  Closed-form solution (GP posterior is Gaussian):
    Let  Z = (mu(x) - f* - xi) / std(x)
    EI(x) = (mu(x) - f* - xi) * Phi(Z) + std(x) * phi(Z)

  where Phi is the standard-normal CDF and phi is its PDF.
  xi (xi) is a small jitter (default 0.01) that nudges toward exploration.

Reference: Mockus (1975); Jones et al. (1998) "Efficient Global Optimization".
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(
    X: np.ndarray,
    surrogate,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Compute Expected Improvement at each candidate point.

    Parameters
    ----------
    X         : shape (n_candidates, n_params) — points to evaluate EI at
    surrogate : fitted GaussianProcessSurrogate
    y_best    : best *observed* objective value so far  (f*)
    xi        : exploration–exploitation trade-off (higher → more exploration)

    Returns
    -------
    ei : shape (n_candidates,) — EI value at each candidate point
    """
    mu, std = surrogate.predict(X)

    # Avoid division by zero at already-observed points (std ≈ 0)
    std = np.maximum(std, 1e-9)

    # Core EI formula — ~10 lines of math
    improvement = mu - y_best - xi
    Z  = improvement / std
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)

    # EI is non-negative by definition; clip numerical noise
    ei = np.maximum(ei, 0.0)
    return ei


def upper_confidence_bound(
    X: np.ndarray,
    surrogate,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Upper Confidence Bound — alternative acquisition function.

    UCB(x) = mu(x) + sqrt(beta) * std(x)

    Higher beta → more exploration.
    Included for comparison; the optimizer uses EI by default.
    """
    mu, std = surrogate.predict(X)
    return mu + np.sqrt(beta) * std


def propose_next_point(
    surrogate,
    y_best: float,
    bounds: np.ndarray,
    n_restarts: int = 20,
    n_random: int = 10_000,
    xi: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Find the next point to evaluate by maximising EI.

    Strategy: random grid search over a large candidate set, then
    L-BFGS-B local optimisation from the top-k candidates.
    This avoids getting stuck in local EI maxima.

    Parameters
    ----------
    surrogate  : fitted GaussianProcessSurrogate
    y_best     : best observed value so far
    bounds     : shape (n_params, 2)
    n_restarts : number of local optimisation starts
    n_random   : size of initial random candidate pool
    xi         : EI exploration parameter
    rng        : optional numpy Generator

    Returns
    -------
    x_next : shape (n_params,) — recommended next experiment point
    """
    from scipy.optimize import minimize

    if rng is None:
        rng = np.random.default_rng()

    n_params = bounds.shape[0]

    # ── Step 1: coarse random search ─────────────────────────────────────────
    X_random = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_random, n_params))
    ei_random = expected_improvement(X_random, surrogate, y_best, xi=xi)

    # Pick top candidates as local optimisation seeds
    top_idx   = np.argsort(ei_random)[-n_restarts:]
    X_seeds   = X_random[top_idx]

    # ── Step 2: local optimisation (maximise EI = minimise −EI) ──────────────
    best_ei   = -np.inf
    best_x    = X_seeds[0]

    for x0 in X_seeds:
        result = minimize(
            fun=lambda x: -expected_improvement(x.reshape(1, -1), surrogate, y_best, xi=xi)[0],
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if result.success and -result.fun > best_ei:
            best_ei = -result.fun
            best_x  = result.x

    return np.clip(best_x, bounds[:, 0], bounds[:, 1])
