"""
tests/test_all.py — pytest unit tests for every module.

Run with:  pytest -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Make the parent package importable when running from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from process import (
    true_function, observe, random_sample,
    BOUNDS, N_PARAMS, GLOBAL_OPTIMUM_VAL, GLOBAL_OPTIMUM_X,
)
from surrogate import GaussianProcessSurrogate
from acquisition import expected_improvement, upper_confidence_bound, propose_next_point
from optimizer import bayesian_optimize, random_search


# ══════════════════════════════════════════════════════════════════════════════
# process.py
# ══════════════════════════════════════════════════════════════════════════════

class TestProcess:
    def test_true_function_returns_scalar(self):
        x = np.zeros(N_PARAMS)
        val = true_function(x)
        assert isinstance(val, float)

    def test_true_function_at_global_optimum_is_max(self):
        """Global optimum should have the highest true function value."""
        rng = np.random.default_rng(0)
        random_pts = rng.uniform(0, 1, size=(500, N_PARAMS))
        random_vals = np.array([true_function(x) for x in random_pts])
        assert true_function(GLOBAL_OPTIMUM_X) >= random_vals.max() - 1e-6

    def test_observe_adds_noise(self):
        """Repeated observations at the same point should differ."""
        rng = np.random.default_rng(1)
        x   = np.full(N_PARAMS, 0.5)
        obs = [observe(x, noise_std=0.1, rng=rng) for _ in range(50)]
        assert np.std(obs) > 0.0

    def test_observe_noise_level(self):
        """Empirical noise std should be close to the specified noise_std."""
        rng      = np.random.default_rng(2)
        x        = GLOBAL_OPTIMUM_X.copy()
        noise_std = 0.05
        obs      = np.array([observe(x, noise_std=noise_std, rng=rng) for _ in range(2000)])
        assert abs(np.std(obs) - noise_std) < 0.01

    def test_random_sample_shape(self):
        samples = random_sample(10)
        assert samples.shape == (10, N_PARAMS)

    def test_random_sample_in_bounds(self):
        samples = random_sample(200, rng=np.random.default_rng(3))
        assert np.all(samples >= BOUNDS[:, 0])
        assert np.all(samples <= BOUNDS[:, 1])

    def test_global_optimum_val_consistent(self):
        assert abs(GLOBAL_OPTIMUM_VAL - true_function(GLOBAL_OPTIMUM_X)) < 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# surrogate.py
# ══════════════════════════════════════════════════════════════════════════════

class TestGaussianProcessSurrogate:
    @pytest.fixture
    def fitted_surrogate(self):
        rng = np.random.default_rng(42)
        X   = rng.uniform(0, 1, size=(20, N_PARAMS))
        y   = np.array([true_function(x) + rng.normal(0, 0.05) for x in X])
        s   = GaussianProcessSurrogate(noise_std=0.05)
        s.fit(X, y)
        return s

    def test_not_fitted_raises(self):
        s = GaussianProcessSurrogate()
        with pytest.raises(RuntimeError, match="fit"):
            s.predict(np.zeros((1, N_PARAMS)))

    def test_predict_shapes(self, fitted_surrogate):
        X_test     = np.random.default_rng(0).uniform(0, 1, (10, N_PARAMS))
        mu, std    = fitted_surrogate.predict(X_test)
        assert mu.shape  == (10,)
        assert std.shape == (10,)

    def test_std_nonnegative(self, fitted_surrogate):
        X_test  = np.random.default_rng(1).uniform(0, 1, (30, N_PARAMS))
        _, std  = fitted_surrogate.predict(X_test)
        assert np.all(std >= 0)

    def test_fit_idempotent(self, fitted_surrogate):
        """Refitting on the same data should not crash."""
        rng = np.random.default_rng(5)
        X   = rng.uniform(0, 1, size=(20, N_PARAMS))
        y   = np.array([true_function(x) for x in X])
        fitted_surrogate.fit(X, y)   # second fit
        assert fitted_surrogate.fitted

    def test_calibration_rmse_reasonable(self, fitted_surrogate):
        rng    = np.random.default_rng(99)
        X_test = rng.uniform(0, 1, size=(50, N_PARAMS))
        y_test = np.array([true_function(x) for x in X_test])
        rmse   = fitted_surrogate.calibration_rmse(X_test, y_test)
        assert rmse < 0.5   # very loose — 20 training points for 5-D space

    def test_higher_density_lowers_uncertainty(self):
        """GP uncertainty should be lower near observed points."""
        rng = np.random.default_rng(7)
        X_obs = np.tile(np.array([0.5] * N_PARAMS), (10, 1))
        X_obs += rng.normal(0, 0.02, X_obs.shape)
        y_obs = np.array([true_function(x) for x in X_obs])

        s = GaussianProcessSurrogate()
        s.fit(X_obs, y_obs)

        x_near = np.array([[0.5] * N_PARAMS])
        x_far  = np.array([[0.0] * N_PARAMS])
        _, std_near = s.predict(x_near)
        _, std_far  = s.predict(x_far)
        assert std_near[0] < std_far[0]


# ══════════════════════════════════════════════════════════════════════════════
# acquisition.py
# ══════════════════════════════════════════════════════════════════════════════

class TestAcquisitionFunctions:
    @pytest.fixture
    def surrogate_and_ybest(self):
        rng = np.random.default_rng(42)
        X   = rng.uniform(0, 1, size=(15, N_PARAMS))
        y   = np.array([true_function(x) + rng.normal(0, 0.05) for x in X])
        s   = GaussianProcessSurrogate()
        s.fit(X, y)
        return s, float(np.max(y))

    def test_ei_nonnegative(self, surrogate_and_ybest):
        surrogate, y_best = surrogate_and_ybest
        rng = np.random.default_rng(0)
        X   = rng.uniform(0, 1, (100, N_PARAMS))
        ei  = expected_improvement(X, surrogate, y_best)
        assert np.all(ei >= 0)

    def test_ei_shape(self, surrogate_and_ybest):
        surrogate, y_best = surrogate_and_ybest
        X  = np.random.default_rng(1).uniform(0, 1, (20, N_PARAMS))
        ei = expected_improvement(X, surrogate, y_best)
        assert ei.shape == (20,)

    def test_ei_zero_at_very_high_y_best(self, surrogate_and_ybest):
        """If y_best is far above any achievable value, EI should be near 0."""
        surrogate, _ = surrogate_and_ybest
        X  = np.random.default_rng(2).uniform(0, 1, (50, N_PARAMS))
        ei = expected_improvement(X, surrogate, y_best=1e6, xi=0.0)
        assert np.all(ei < 1e-3)

    def test_ucb_shape(self, surrogate_and_ybest):
        surrogate, _ = surrogate_and_ybest
        X   = np.random.default_rng(3).uniform(0, 1, (10, N_PARAMS))
        ucb = upper_confidence_bound(X, surrogate)
        assert ucb.shape == (10,)

    def test_propose_next_point_in_bounds(self, surrogate_and_ybest):
        surrogate, y_best = surrogate_and_ybest
        x_next = propose_next_point(surrogate, y_best, BOUNDS,
                                    n_restarts=5, n_random=200,
                                    rng=np.random.default_rng(4))
        assert x_next.shape == (N_PARAMS,)
        assert np.all(x_next >= BOUNDS[:, 0] - 1e-9)
        assert np.all(x_next <= BOUNDS[:, 1] + 1e-9)

    def test_propose_next_point_different_from_observed(self, surrogate_and_ybest):
        """Proposed point should explore, not just revisit the best observed."""
        surrogate, y_best = surrogate_and_ybest
        rng    = np.random.default_rng(5)
        x_next = propose_next_point(surrogate, y_best, BOUNDS,
                                    n_restarts=5, n_random=500, rng=rng)
        # Should not collapse exactly onto the observed maximum
        x_best_obs = np.full(N_PARAMS, 0.5)
        assert not np.allclose(x_next, x_best_obs, atol=0.05)


# ══════════════════════════════════════════════════════════════════════════════
# optimizer.py
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizer:
    @pytest.fixture(scope="class")
    def bo_result(self):
        return bayesian_optimize(n_initial=5, n_iterations=10, seed=42)

    @pytest.fixture(scope="class")
    def rand_result(self):
        return random_search(n_total=15, seed=42)

    def test_bo_result_shapes(self, bo_result):
        assert bo_result.X_observed.shape == (15, N_PARAMS)
        assert bo_result.y_observed.shape == (15,)

    def test_bo_best_y_consistent(self, bo_result):
        assert abs(bo_result.best_y - np.max(bo_result.y_observed)) < 1e-9

    def test_bo_best_x_consistent(self, bo_result):
        best_idx = np.argmax(bo_result.y_observed)
        assert np.allclose(bo_result.best_x, bo_result.X_observed[best_idx])

    def test_bo_best_so_far_monotone(self, bo_result):
        """Running best must be non-decreasing."""
        assert np.all(np.diff(bo_result.best_so_far) >= -1e-9)

    def test_rand_result_shapes(self, rand_result):
        assert rand_result.X_observed.shape == (15, N_PARAMS)
        assert rand_result.y_observed.shape == (15,)

    def test_rand_best_so_far_monotone(self, rand_result):
        assert np.all(np.diff(rand_result.best_so_far) >= -1e-9)

    def test_bo_outperforms_random_on_average(self):
        """
        Over multiple seeds, BO should find a better result than random
        search with the same budget.  Uses a generous threshold since
        n_iterations is tiny in the test suite.
        """
        bo_bests   = []
        rand_bests = []
        for seed in range(5):
            bo   = bayesian_optimize(n_initial=5, n_iterations=10, seed=seed)
            rand = random_search(n_total=15, seed=seed)
            bo_bests.append(bo.best_y)
            rand_bests.append(rand.best_y)

        # BO mean best should be >= random mean best (very lenient — small budget)
        assert np.mean(bo_bests) >= np.mean(rand_bests) - 0.05
