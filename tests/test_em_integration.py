"""Integration tests for EM algorithm on DFSV models."""

import numpy as np
import pytest

from bellman_filter_dfsv.core.models import DFSVParamsDataclass
from bellman_filter_dfsv.core.models.simulation import simulate_DFSV
from bellman_filter_dfsv.core.optimization.em import EMHistory, EMOptimizer


def create_test_params(N: int = 5, K: int = 1) -> DFSVParamsDataclass:
    """Create well-behaved test parameters."""
    np.random.seed(42)
    lambda_r = np.random.randn(N, K) * 0.5
    lambda_r[0, :] = np.abs(lambda_r[0, :]) + 0.1

    return DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=np.eye(K) * 0.8,
        Phi_h=np.eye(K) * 0.95,
        mu=np.zeros(K) - 1.0,
        Q_h=np.eye(K) * 0.1,
        sigma2=np.ones(N) * 0.1,
    )


class TestEMOptimizerIntegration:
    """Integration tests for full EM algorithm."""

    def test_single_em_iteration_runs_without_error(self):
        """Verify one complete EM iteration executes."""
        N, K, T = 5, 1, 50
        params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(params, T=T, seed=123)

        em = EMOptimizer(N=N, K=K, max_iters=1, verbose=False)
        fitted_params, history = em.fit(returns, params)

        assert isinstance(fitted_params, DFSVParamsDataclass)
        assert isinstance(history, EMHistory)
        assert len(history.log_likelihoods) == 1
        assert np.isfinite(history.log_likelihoods[0])

    def test_em_log_likelihood_is_finite(self):
        """Verify log-likelihood is finite throughout EM."""
        N, K, T = 5, 1, 100
        params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(params, T=T, seed=456)

        em = EMOptimizer(N=N, K=K, max_iters=5, verbose=False)
        _, history = em.fit(returns, params)

        for ll in history.log_likelihoods:
            assert np.isfinite(ll), f"Non-finite log-likelihood: {ll}"

    def test_em_log_likelihood_non_decreasing(self):
        """EM guarantee: log-likelihood should not decrease significantly.

        Note: With Gaussian approximation (BIF), small decreases can occur
        since we're not optimizing the exact likelihood. We allow small
        violations (< 5 log-likelihood units).
        """
        N, K, T = 5, 1, 100
        params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(params, T=T, seed=789)

        em = EMOptimizer(N=N, K=K, max_iters=10, tol=1e-8, verbose=False)
        _, history = em.fit(returns, params)

        lls = history.log_likelihoods
        for i in range(1, len(lls)):
            decrease = lls[i - 1] - lls[i]
            assert decrease < 5.0, (
                f"Log-likelihood decreased significantly at iter {i}: "
                f"{lls[i - 1]:.4f} -> {lls[i]:.4f}"
            )

    def test_em_returns_valid_parameters(self):
        """Verify fitted parameters satisfy constraints."""
        N, K, T = 5, 1, 100
        params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(params, T=T, seed=101)

        em = EMOptimizer(N=N, K=K, max_iters=5, verbose=False)
        fitted, _ = em.fit(returns, params)

        assert fitted.lambda_r.shape == (N, K)
        assert fitted.sigma2.shape == (N,)
        assert np.all(fitted.sigma2 > 0), "sigma2 must be positive"

        phi_f_diag = np.diag(fitted.Phi_f)
        assert np.all(np.abs(phi_f_diag) < 1), "Phi_f eigenvalues must be < 1"

        phi_h_diag = np.diag(fitted.Phi_h)
        assert np.all(np.abs(phi_h_diag) < 1), "Phi_h eigenvalues must be < 1"

        Q_h_diag = np.diag(fitted.Q_h)
        assert np.all(Q_h_diag > 0), "Q_h diagonal must be positive"

    def test_em_with_two_factors(self):
        """Verify EM works with K > 1."""
        N, K, T = 8, 2, 100
        params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(params, T=T, seed=202)

        em = EMOptimizer(N=N, K=K, max_iters=3, verbose=False)
        fitted, history = em.fit(returns, params)

        assert fitted.lambda_r.shape == (N, K)
        assert fitted.Phi_f.shape == (K, K)
        assert fitted.mu.shape == (K,)
        assert len(history.log_likelihoods) >= 1


class TestEMParameterRecovery:
    """Test that EM can approximately recover true parameters."""

    @pytest.mark.slow
    def test_recovers_lambda_approximately(self):
        """With enough data, lambda_r should be close to true value."""
        N, K, T = 5, 1, 500
        true_params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(true_params, T=T, seed=303)

        em = EMOptimizer(N=N, K=K, max_iters=20, tol=1e-4, verbose=False)
        fitted, history = em.fit(returns, true_params)

        lambda_error = np.mean(np.abs(fitted.lambda_r - true_params.lambda_r))
        assert lambda_error < 0.5, f"Lambda recovery error too high: {lambda_error}"

    @pytest.mark.slow
    def test_recovers_phi_h_approximately(self):
        """Phi_h should be recovered reasonably well."""
        N, K, T = 5, 1, 500
        true_params = create_test_params(N, K)
        returns, _, _ = simulate_DFSV(true_params, T=T, seed=404)

        em = EMOptimizer(N=N, K=K, max_iters=20, tol=1e-4, verbose=False)
        fitted, _ = em.fit(returns, true_params)

        phi_h_error = np.abs(np.diag(fitted.Phi_h) - np.diag(true_params.Phi_h)).mean()
        assert phi_h_error < 0.2, f"Phi_h recovery error too high: {phi_h_error}"
