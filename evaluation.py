"""
evaluation.py — Benchmark and visualise optimization results.

Compares Bayesian Optimization vs Random Search and produces:
  1. Convergence plot (best-so-far vs iterations)
  2. EI surface slice (2D projection of acquisition landscape)
  3. GP fit quality summary
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from process import true_function, GLOBAL_OPTIMUM_VAL, BOUNDS, N_PARAMS
from surrogate import GaussianProcessSurrogate
from acquisition import expected_improvement
from optimizer import OptimizationResult


# ── Convergence metric ────────────────────────────────────────────────────────

def iterations_to_threshold(best_so_far: np.ndarray, threshold: float) -> int | None:
    """Return the iteration index at which best_so_far first crosses threshold."""
    crossings = np.where(best_so_far >= threshold)[0]
    return int(crossings[0]) if len(crossings) > 0 else None


def summarise(result: OptimizationResult, label: str = "") -> None:
    """Print a short text summary of a run."""
    print(f"\n{'─'*55}")
    print(f"  {label or result.method.upper()} SUMMARY")
    print(f"{'─'*55}")
    print(f"  Total experiments      : {result.n_iterations}")
    print(f"  Best observed value    : {result.best_y:.4f}")
    print(f"  Global optimum value   : {GLOBAL_OPTIMUM_VAL:.4f}")
    gap = GLOBAL_OPTIMUM_VAL - result.best_y
    print(f"  Gap to global optimum  : {gap:.4f}")
    print(f"  Best input parameters  : {np.round(result.best_x, 3)}")

    thresh = 0.85 * GLOBAL_OPTIMUM_VAL
    it     = iterations_to_threshold(result.best_so_far, thresh)
    if it is not None:
        print(f"  Iterations to 85% opt  : {it}")
    else:
        print(f"  Never reached 85% opt threshold")
    print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(
    bo_result:   OptimizationResult,
    rand_result: OptimizationResult,
    save_path:   str | Path = "convergence.png",
) -> None:
    """
    Plot best-so-far vs number of experiments for BO and random search.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    n_bo   = len(bo_result.best_so_far)
    n_rand = len(rand_result.best_so_far)

    ax.plot(range(n_bo),   bo_result.best_so_far,   color="#00d4aa", lw=2.5,
            label="Bayesian Optimization (GP + EI)", zorder=3)
    ax.plot(range(n_rand), rand_result.best_so_far, color="#ff6b6b", lw=2.5,
            ls="--", label="Random Search baseline", zorder=3)

    ax.axhline(GLOBAL_OPTIMUM_VAL, color="white", lw=1, ls=":", alpha=0.5,
               label=f"Global optimum ({GLOBAL_OPTIMUM_VAL:.3f})")

    # Annotate speedup
    thresh = 0.85 * GLOBAL_OPTIMUM_VAL
    it_bo   = iterations_to_threshold(bo_result.best_so_far,   thresh)
    it_rand = iterations_to_threshold(rand_result.best_so_far, thresh)
    if it_bo and it_rand:
        speedup = it_rand / it_bo
        ax.annotate(
            f"BO reaches 85% optimum\n{speedup:.1f}× faster than random",
            xy=(it_bo, thresh), xytext=(it_bo + 3, thresh - 0.07),
            color="#00d4aa", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#00d4aa", lw=1.5),
        )

    ax.set_xlabel("Number of experiments", color="white", fontsize=12)
    ax.set_ylabel("Best observed output", color="white", fontsize=12)
    ax.set_title("Bayesian Optimization vs Random Search\nConvergence on Synthetic Manufacturing Process",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1e2130", edgecolor="#444", labelcolor="white", fontsize=10)
    ax.grid(alpha=0.15, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {save_path}")


def plot_gp_slice(
    bo_result: OptimizationResult,
    param_i:   int = 0,
    param_j:   int = 1,
    save_path: str | Path = "gp_slice.png",
    resolution: int = 60,
) -> None:
    """
    2-D slice of the GP posterior (mean + uncertainty) after the BO run,
    fixing all other parameters at their optimum values.
    Shows where the GP thinks the optimum is and how confident it is.
    """
    # Fit final surrogate
    surrogate = GaussianProcessSurrogate(noise_std=0.05)
    surrogate.fit(bo_result.X_observed, bo_result.y_observed)

    # Build grid in the 2-D slice
    g = np.linspace(0, 1, resolution)
    G1, G2 = np.meshgrid(g, g)
    base = bo_result.best_x.copy()   # fix other dims at best point

    grid_pts = np.tile(base, (resolution * resolution, 1))
    grid_pts[:, param_i] = G1.ravel()
    grid_pts[:, param_j] = G2.ravel()

    mu, std = surrogate.predict(grid_pts)
    MU  = mu.reshape(resolution, resolution)
    STD = std.reshape(resolution, resolution)

    # True function on the same slice (for comparison)
    true_vals = np.array([
        true_function(grid_pts[k]) for k in range(len(grid_pts))
    ]).reshape(resolution, resolution)

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    titles = ["GP Posterior Mean", "GP Uncertainty (std)", "Ground Truth"]
    datas  = [MU, STD, true_vals]
    cmaps  = ["viridis", "plasma", "viridis"]

    for idx, (title, data, cmap) in enumerate(zip(titles, datas, cmaps)):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#0f1117")
        im = ax.contourf(G1, G2, data, levels=25, cmap=cmap)
        plt.colorbar(im, ax=ax).ax.tick_params(colors="white", labelsize=8)

        # Scatter observed points
        ax.scatter(
            bo_result.X_observed[:, param_i],
            bo_result.X_observed[:, param_j],
            c="white", s=18, zorder=5, alpha=0.7, label="Observations"
        )
        # Mark best
        ax.scatter(
            bo_result.best_x[param_i], bo_result.best_x[param_j],
            marker="*", c="#ffdd00", s=200, zorder=6, label="Best found"
        )

        ax.set_xlabel(f"Param {param_i}", color="white", fontsize=10)
        ax.set_ylabel(f"Param {param_j}", color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        if idx == 0:
            ax.legend(facecolor="#1e2130", edgecolor="#444", labelcolor="white", fontsize=8)

    fig.suptitle(f"GP Posterior Slice — params {param_i} vs {param_j}",
                 color="white", fontsize=13, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {save_path}")


def plot_ei_trajectory(
    bo_result: OptimizationResult,
    save_path: str | Path = "ei_trajectory.png",
) -> None:
    """
    Show the EI value of each selected point over the BO run.
    Illustrates how EI starts high (lots of uncertainty) and decays as
    the surrogate becomes confident near the optimum.
    """
    n_init = 5    # warm-up points had no EI selection
    n_bo   = bo_result.n_iterations - n_init

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Recompute EI at each selected point (retrospectively)
    surrogate = GaussianProcessSurrogate(noise_std=0.05)
    ei_vals   = []
    X, y      = bo_result.X_observed[:n_init], bo_result.y_observed[:n_init]

    for i in range(n_bo):
        surrogate.fit(X, y)
        x_sel = bo_result.X_observed[n_init + i]
        ei    = expected_improvement(x_sel.reshape(1, -1), surrogate, float(np.max(y)))[0]
        ei_vals.append(ei)
        X = np.vstack([X, x_sel])
        y = np.append(y, bo_result.y_observed[n_init + i])

    ax.bar(range(n_bo), ei_vals, color="#00d4aa", alpha=0.75, width=0.8)
    ax.set_xlabel("BO Iteration", color="white", fontsize=12)
    ax.set_ylabel("Expected Improvement at Selected Point", color="white", fontsize=12)
    ax.set_title("EI Decay Over BO Iterations\n(high EI early = exploration; low EI late = exploitation near optimum)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(axis="y", alpha=0.15, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {save_path}")
