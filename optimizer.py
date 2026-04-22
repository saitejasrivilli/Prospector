"""
optimizer.py — Bayesian Optimization loop.

Orchestrates the surrogate model + acquisition function into a full
Bayesian Optimization (BO) routine and provides a parallel Random Search
baseline for benchmarking.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from surrogate import GaussianProcessSurrogate
from acquisition import propose_next_point
from process import observe, BOUNDS, N_PARAMS


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Stores the full history of an optimization run."""
    X_observed: np.ndarray          # shape (n_iter, n_params)
    y_observed: np.ndarray          # shape (n_iter,)
    best_x:     np.ndarray          # best input found
    best_y:     float               # best output found
    best_so_far: np.ndarray         # running best at each iteration
    method:     str = "bayesian"    # "bayesian" | "random"

    @property
    def n_iterations(self) -> int:
        return len(self.y_observed)


# ── Bayesian Optimization ─────────────────────────────────────────────────────

def bayesian_optimize(
    n_initial: int = 5,
    n_iterations: int = 45,
    noise_std: float = 0.05,
    xi: float = 0.01,
    seed: int = 42,
) -> OptimizationResult:
    """
    Run Bayesian Optimization on the synthetic process.

    Parameters
    ----------
    n_initial   : number of random initial observations (warm-up)
    n_iterations: number of BO acquisition steps
    noise_std   : measurement noise std (passed to observe())
    xi          : EI exploration parameter
    seed        : random seed for reproducibility

    Returns
    -------
    OptimizationResult
    """
    rng       = np.random.default_rng(seed)
    surrogate = GaussianProcessSurrogate(noise_std=noise_std, random_state=seed)

    # ── Warm-up: random initial design ───────────────────────────────────────
    X_init = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_initial, N_PARAMS))
    y_init = np.array([observe(x, noise_std=noise_std, rng=rng) for x in X_init])

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    best_so_far = [float(np.max(y_obs))]

    # ── BO loop ───────────────────────────────────────────────────────────────
    for i in range(n_iterations):
        # 1. Fit surrogate on all observations so far
        surrogate.fit(X_obs, y_obs)

        # 2. Propose next point via EI maximisation
        y_best = float(np.max(y_obs))
        x_next = propose_next_point(
            surrogate=surrogate,
            y_best=y_best,
            bounds=BOUNDS,
            xi=xi,
            rng=rng,
        )

        # 3. Evaluate (expensive "experiment")
        y_next = observe(x_next, noise_std=noise_std, rng=rng)

        # 4. Append to dataset
        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

        best_so_far.append(float(np.max(y_obs)))

        print(
            f"  BO iter {i+1:3d}/{n_iterations} | "
            f"EI-proposed y={y_next:.4f} | best={best_so_far[-1]:.4f}"
        )

    best_idx = int(np.argmax(y_obs))
    return OptimizationResult(
        X_observed  = X_obs,
        y_observed  = y_obs,
        best_x      = X_obs[best_idx],
        best_y      = float(y_obs[best_idx]),
        best_so_far = np.array(best_so_far),
        method      = "bayesian",
    )


# ── Random Search baseline ────────────────────────────────────────────────────

def random_search(
    n_total: int = 50,
    noise_std: float = 0.05,
    seed: int = 42,
) -> OptimizationResult:
    """
    Pure random search baseline — same budget as BO, no model.

    Parameters
    ----------
    n_total   : total number of random experiments
    noise_std : measurement noise std
    seed      : random seed

    Returns
    -------
    OptimizationResult
    """
    rng = np.random.default_rng(seed)

    X_obs = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_total, N_PARAMS))
    y_obs = np.array([observe(x, noise_std=noise_std, rng=rng) for x in X_obs])

    best_so_far = np.maximum.accumulate(y_obs)

    best_idx = int(np.argmax(y_obs))
    return OptimizationResult(
        X_observed  = X_obs,
        y_observed  = y_obs,
        best_x      = X_obs[best_idx],
        best_y      = float(y_obs[best_idx]),
        best_so_far = best_so_far,
        method      = "random",
    )
