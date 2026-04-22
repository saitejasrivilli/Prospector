"""
surrogate.py — Gaussian Process surrogate model.

Wraps scikit-learn's GaussianProcessRegressor with a clean interface
used by the Bayesian optimization loop.

Why a GP?
  - Gives a predictive mean AND uncertainty (std) at every point.
  - Uncertainty is what drives the acquisition function to explore.
  - Data-efficient: works well with O(10–100) observations.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class GaussianProcessSurrogate:
    """
    Thin wrapper around sklearn's GPR.

    Kernel choice — ConstantKernel * Matern(nu=2.5):
      - Matern-2.5 is the standard BO kernel: twice-differentiable,
        slightly less smooth than RBF, which better matches real processes.
      - ConstantKernel lets the GP scale its amplitude automatically.
    """

    def __init__(self, noise_std: float = 0.05, random_state: int = 42):
        self.noise_std    = noise_std
        self.random_state = random_state
        self._build_gp()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_gp(self) -> None:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=0.3,
            length_scale_bounds=(1e-2, 10.0),
            nu=2.5,
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_std ** 2,   # observation noise variance
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=self.random_state,
        )
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """
        Fit the GP on observed (X, y) pairs.

        Parameters
        ----------
        X : shape (n_obs, n_params)
        y : shape (n_obs,)
        """
        self.gp.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return predictive mean and standard deviation.

        Parameters
        ----------
        X : shape (n_points, n_params)

        Returns
        -------
        mu  : shape (n_points,)  — posterior mean
        std : shape (n_points,)  — posterior std (sqrt of variance)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        mu, std = self.gp.predict(X, return_std=True)
        return mu, std

    @property
    def fitted(self) -> bool:
        return self._fitted

    def calibration_rmse(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute RMSE of the surrogate's mean predictions on held-out data.
        Useful for benchmarking uncertainty calibration.
        """
        mu, _ = self.predict(X_test)
        return float(np.sqrt(np.mean((mu - y_test) ** 2)))
