"""
main.py — Run the full Bayesian Optimization benchmark.

Usage:
    python main.py
    python main.py --n-iter 30 --seed 7
"""

import argparse
from pathlib import Path

from optimizer import bayesian_optimize, random_search
from evaluation import summarise, plot_convergence, plot_gp_slice, plot_ei_trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian Process Optimizer")
    p.add_argument("--n-init",  type=int,   default=5,    help="# random warm-up experiments")
    p.add_argument("--n-iter",  type=int,   default=45,   help="# BO acquisition iterations")
    p.add_argument("--noise",   type=float, default=0.05, help="observation noise std")
    p.add_argument("--xi",      type=float, default=0.01, help="EI exploration parameter")
    p.add_argument("--seed",    type=int,   default=42,   help="random seed")
    p.add_argument("--out-dir", type=Path,  default=Path("."), help="directory for saved plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_total = args.n_init + args.n_iter

    print("=" * 55)
    print("  BAYESIAN PROCESS OPTIMIZER")
    print("=" * 55)

    # ── Bayesian Optimization ─────────────────────────────────────────────────
    print(f"\n[1/2] Running Bayesian Optimization ({n_total} total experiments) …")
    bo_result = bayesian_optimize(
        n_initial   = args.n_init,
        n_iterations= args.n_iter,
        noise_std   = args.noise,
        xi          = args.xi,
        seed        = args.seed,
    )
    summarise(bo_result, label="Bayesian Optimization")

    # ── Random Search ─────────────────────────────────────────────────────────
    print(f"[2/2] Running Random Search ({n_total} total experiments) …")
    rand_result = random_search(
        n_total   = n_total,
        noise_std = args.noise,
        seed      = args.seed,
    )
    summarise(rand_result, label="Random Search (baseline)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_convergence(bo_result, rand_result,
                     save_path=args.out_dir / "convergence.png")
    plot_gp_slice(bo_result,
                  save_path=args.out_dir / "gp_slice.png")
    plot_ei_trajectory(bo_result,
                       save_path=args.out_dir / "ei_trajectory.png")

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
