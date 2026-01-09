"""
Comprehensive tests for EM algorithm components.

Tests are organized by component:
1. EMSufficientStats - dataclass operations
2. E-step - sufficient statistics computation
3. M-step - closed-form parameter updates
4. EMOptimizer - full EM iteration
"""

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


class TestEMSufficientStats:
    """Tests for EMSufficientStats dataclass."""

    def test_zeros_creates_correct_shapes(self):
        """Given N=5, K=2, T=100, zeros() returns correctly shaped arrays."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 5, 2, 100
        stats = EMSufficientStats.zeros(N, K, T)

        # Observation stats
        assert stats.sum_r_f.shape == (N, K)
        assert stats.sum_f_f.shape == (K, K)
        assert stats.sum_r_r_diag.shape == (N,)

        # Factor dynamics stats
        assert stats.sum_f_fprev.shape == (K, K)
        assert stats.sum_fprev_fprev.shape == (K, K)
        assert stats.sum_exp_neg_h.shape == (K,)
        assert stats.sum_exp_neg_h_f_fprev_diag.shape == (K,)
        assert stats.sum_exp_neg_h_fprev_sq.shape == (K,)

        # Log-vol stats
        assert stats.sum_h.shape == (K,)
        assert stats.sum_hprev.shape == (K,)
        assert stats.sum_h_h.shape == (K, K)
        assert stats.sum_h_hprev.shape == (K, K)
        assert stats.sum_hprev_hprev.shape == (K, K)

        # Count
        assert stats.T == T

    def test_zeros_all_values_are_zero(self):
        """Given zeros(), all array values should be 0.0."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        stats = EMSufficientStats.zeros(N=3, K=2, T=50)

        assert jnp.allclose(stats.sum_r_f, 0.0)
        assert jnp.allclose(stats.sum_f_f, 0.0)
        assert jnp.allclose(stats.sum_r_r_diag, 0.0)
        assert jnp.allclose(stats.sum_f_fprev, 0.0)
        assert jnp.allclose(stats.sum_exp_neg_h, 0.0)
        assert jnp.allclose(stats.sum_h, 0.0)
        assert jnp.allclose(stats.sum_h_h, 0.0)

    def test_addition_combines_stats_elementwise(self):
        """Given two stats, addition sums all arrays element-wise."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 4, 2, 100

        stats1 = EMSufficientStats(
            sum_r_f=jnp.ones((N, K)),
            sum_f_f=jnp.ones((K, K)) * 2,
            sum_r_r_diag=jnp.ones(N) * 3,
            sum_f_fprev=jnp.ones((K, K)),
            sum_fprev_fprev=jnp.ones((K, K)),
            sum_exp_neg_h=jnp.ones(K),
            sum_exp_neg_h_f_fprev_diag=jnp.ones(K),
            sum_exp_neg_h_fprev_sq=jnp.ones(K),
            sum_h=jnp.ones(K) * 4,
            sum_hprev=jnp.ones(K),
            sum_h_h=jnp.ones((K, K)),
            sum_h_hprev=jnp.ones((K, K)),
            sum_hprev_hprev=jnp.ones((K, K)),
            T=T,
        )

        stats2 = EMSufficientStats(
            sum_r_f=jnp.ones((N, K)) * 10,
            sum_f_f=jnp.ones((K, K)) * 20,
            sum_r_r_diag=jnp.ones(N) * 30,
            sum_f_fprev=jnp.ones((K, K)) * 10,
            sum_fprev_fprev=jnp.ones((K, K)) * 10,
            sum_exp_neg_h=jnp.ones(K) * 10,
            sum_exp_neg_h_f_fprev_diag=jnp.ones(K) * 10,
            sum_exp_neg_h_fprev_sq=jnp.ones(K) * 10,
            sum_h=jnp.ones(K) * 40,
            sum_hprev=jnp.ones(K) * 10,
            sum_h_h=jnp.ones((K, K)) * 10,
            sum_h_hprev=jnp.ones((K, K)) * 10,
            sum_hprev_hprev=jnp.ones((K, K)) * 10,
            T=T,
        )

        combined = stats1 + stats2

        assert jnp.allclose(combined.sum_r_f, 11.0)
        assert jnp.allclose(combined.sum_f_f, 22.0)
        assert jnp.allclose(combined.sum_r_r_diag, 33.0)
        assert jnp.allclose(combined.sum_h, 44.0)
        assert combined.T == T

    def test_is_valid_jax_pytree(self):
        """EMSufficientStats should work with JAX pytree operations."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        stats = EMSufficientStats.zeros(N=3, K=2, T=50)

        # Test tree_map (multiply all by 2)
        doubled = jax.tree_util.tree_map(lambda x: x * 2, stats)
        assert jnp.allclose(doubled.sum_r_f, 0.0)  # 0 * 2 = 0

        # Test with non-zero values
        stats_nonzero = EMSufficientStats(
            sum_r_f=jnp.ones((3, 2)),
            sum_f_f=jnp.ones((2, 2)),
            sum_r_r_diag=jnp.ones(3),
            sum_f_fprev=jnp.ones((2, 2)),
            sum_fprev_fprev=jnp.ones((2, 2)),
            sum_exp_neg_h=jnp.ones(2),
            sum_exp_neg_h_f_fprev_diag=jnp.ones(2),
            sum_exp_neg_h_fprev_sq=jnp.ones(2),
            sum_h=jnp.ones(2),
            sum_hprev=jnp.ones(2),
            sum_h_h=jnp.ones((2, 2)),
            sum_h_hprev=jnp.ones((2, 2)),
            sum_hprev_hprev=jnp.ones((2, 2)),
            T=50,
        )
        doubled_nonzero = jax.tree_util.tree_map(lambda x: x * 2, stats_nonzero)
        assert jnp.allclose(doubled_nonzero.sum_r_f, 2.0)

    def test_zeros_plus_zeros_equals_zeros(self):
        """Edge case: 0 + 0 = 0 for all fields."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        z1 = EMSufficientStats.zeros(N=4, K=2, T=100)
        z2 = EMSufficientStats.zeros(N=4, K=2, T=100)

        combined = z1 + z2

        assert jnp.allclose(combined.sum_r_f, 0.0)
        assert jnp.allclose(combined.sum_f_f, 0.0)
        assert jnp.allclose(combined.sum_h, 0.0)


class TestSmoothedMoments:
    """Tests for SmoothedMoments and SmoothedLagMoments."""

    def test_smoothed_moments_is_named_tuple(self):
        """SmoothedMoments should be a NamedTuple with correct fields."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import SmoothedMoments

        K = 2
        moments = SmoothedMoments(
            f_mean=jnp.zeros(K),
            h_mean=jnp.zeros(K),
            P_ff=jnp.eye(K),
            P_hh=jnp.eye(K),
            P_fh=jnp.zeros((K, K)),
        )

        assert moments.f_mean.shape == (K,)
        assert moments.P_ff.shape == (K, K)

    def test_smoothed_lag_moments_has_lag_covariances(self):
        """SmoothedLagMoments stores lag-1 cross-covariances."""
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            SmoothedLagMoments,
        )

        K = 2
        lag = SmoothedLagMoments(
            P_ff_lag=jnp.eye(K) * 0.5,
            P_hh_lag=jnp.eye(K) * 0.3,
        )

        assert lag.P_ff_lag.shape == (K, K)
        assert lag.P_hh_lag.shape == (K, K)


class TestComputeExpNegH:
    """Tests for E[exp(-h)] computation from Gaussian posterior."""

    def test_known_analytical_result(self):
        """
        Given h ~ N(μ, σ²), E[exp(-h)] = exp(-μ + σ²/2).

        Test with known values:
        - μ = 0, σ² = 0 => E[exp(-h)] = exp(0) = 1.0
        - μ = 1, σ² = 0 => E[exp(-h)] = exp(-1) ≈ 0.368
        - μ = 0, σ² = 2 => E[exp(-h)] = exp(1) ≈ 2.718
        """
        from bellman_filter_dfsv.core.optimization._em_estep import compute_exp_neg_h

        # Case 1: μ=0, σ²=0
        result = compute_exp_neg_h(h_mean=jnp.array([0.0]), h_var=jnp.array([0.0]))
        assert jnp.allclose(result, 1.0, atol=1e-10)

        # Case 2: μ=1, σ²=0
        result = compute_exp_neg_h(h_mean=jnp.array([1.0]), h_var=jnp.array([0.0]))
        assert jnp.allclose(result, jnp.exp(-1.0), atol=1e-10)

        # Case 3: μ=0, σ²=2
        result = compute_exp_neg_h(h_mean=jnp.array([0.0]), h_var=jnp.array([2.0]))
        assert jnp.allclose(result, jnp.exp(1.0), atol=1e-10)

        # Case 4: μ=-1, σ²=1
        result = compute_exp_neg_h(h_mean=jnp.array([-1.0]), h_var=jnp.array([1.0]))
        expected = jnp.exp(1.0 + 0.5)  # exp(-(-1) + 1/2)
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_monte_carlo_verification(self):
        """Verify formula against Monte Carlo sampling."""
        from bellman_filter_dfsv.core.optimization._em_estep import compute_exp_neg_h

        key = jr.PRNGKey(42)
        mu = -0.5
        var = 0.3
        n_samples = 100_000

        # Monte Carlo estimate
        samples = jr.normal(key, (n_samples,)) * jnp.sqrt(var) + mu
        mc_estimate = jnp.mean(jnp.exp(-samples))

        # Analytical
        analytical = compute_exp_neg_h(h_mean=jnp.array([mu]), h_var=jnp.array([var]))[
            0
        ]

        assert jnp.allclose(mc_estimate, analytical, rtol=0.02)

    def test_variance_capping_prevents_explosion(self):
        """Large variance should be capped to prevent numerical overflow."""
        from bellman_filter_dfsv.core.optimization._em_estep import compute_exp_neg_h

        # Without capping, var=100 would give exp(50) ≈ 5e21
        result = compute_exp_neg_h(
            h_mean=jnp.array([0.0]), h_var=jnp.array([100.0]), max_var=4.0
        )

        # With capping at 4.0, we get exp(2.0) ≈ 7.39
        expected = jnp.exp(2.0)
        assert jnp.allclose(result, expected, atol=1e-6)
        assert jnp.isfinite(result).all()

    def test_vectorized_over_k_factors(self):
        """Should handle K > 1 factors."""
        from bellman_filter_dfsv.core.optimization._em_estep import compute_exp_neg_h

        K = 3
        h_mean = jnp.array([-1.0, 0.0, 1.0])
        h_var = jnp.array([0.5, 1.0, 0.2])

        result = compute_exp_neg_h(h_mean, h_var)

        assert result.shape == (K,)
        expected = jnp.exp(-h_mean + 0.5 * h_var)
        assert jnp.allclose(result, expected, atol=1e-10)


class TestMStepLambdaR:
    """Tests for factor loadings M-step update."""

    def test_recovers_true_lambda_with_known_factors(self):
        """
        Given perfect knowledge of factors (no uncertainty),
        λ_r should be recovered exactly via OLS.

        Model: r_t = λ_r @ f_t + e_t
        True λ_r = [[0.8], [0.6], [0.9]] for N=3, K=1
        """
        from bellman_filter_dfsv.core.optimization._em_mstep import update_lambda_r
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 3, 1, 500
        true_lambda = jnp.array([[0.8], [0.6], [0.9]])

        key = jr.PRNGKey(123)
        key, k1, k2 = jr.split(key, 3)

        # Simulate factors and observations
        f = jr.normal(k1, (T, K))  # T x K
        noise = jr.normal(k2, (T, N)) * 0.1
        r = f @ true_lambda.T + noise  # T x N

        # Compute sufficient stats as if we had perfect E-step
        # E[f_t] = f_t, E[f_t f_t'] = f_t @ f_t' (no covariance added)
        sum_r_f = jnp.sum(r[:, :, None] * f[:, None, :], axis=0)  # N x K
        sum_f_f = jnp.sum(f[:, :, None] * f[:, None, :], axis=0)  # K x K

        stats = EMSufficientStats(
            sum_r_f=sum_r_f,
            sum_f_f=sum_f_f,
            sum_r_r_diag=jnp.sum(r**2, axis=0),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            T=T,
        )

        lambda_est = update_lambda_r(stats)

        assert jnp.allclose(lambda_est, true_lambda, atol=0.05)

    def test_handles_k_greater_than_one(self):
        """λ_r update should work for K > 1 factors."""
        from bellman_filter_dfsv.core.optimization._em_mstep import update_lambda_r
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 5, 3, 1000
        key = jr.PRNGKey(456)
        true_lambda = jr.normal(key, (N, K)) * 0.5

        key, k1, k2 = jr.split(key, 3)
        f = jr.normal(k1, (T, K))
        noise = jr.normal(k2, (T, N)) * 0.05
        r = f @ true_lambda.T + noise

        sum_r_f = jnp.sum(r[:, :, None] * f[:, None, :], axis=0)
        sum_f_f = jnp.sum(f[:, :, None] * f[:, None, :], axis=0)

        stats = EMSufficientStats(
            sum_r_f=sum_r_f,
            sum_f_f=sum_f_f,
            sum_r_r_diag=jnp.sum(r**2, axis=0),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            T=T,
        )

        lambda_est = update_lambda_r(stats)

        assert lambda_est.shape == (N, K)
        assert jnp.allclose(lambda_est, true_lambda, atol=0.05)


class TestMStepSigma2:
    """Tests for idiosyncratic variance M-step update."""

    def test_recovers_true_sigma2(self):
        """
        Given λ_r and factors, σ² = (1/T) Σ E[(r - λf)²].

        With known factors and λ, this should recover true σ².
        """
        from bellman_filter_dfsv.core.optimization._em_mstep import update_sigma2
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 4, 1, 1000
        true_lambda = jnp.array([[0.7], [0.5], [0.8], [0.6]])
        true_sigma2 = jnp.array([0.1, 0.2, 0.15, 0.25])

        key = jr.PRNGKey(789)
        key, k1, k2 = jr.split(key, 3)

        f = jr.normal(k1, (T, K))
        noise_scales = jnp.sqrt(true_sigma2)
        noise = jr.normal(k2, (T, N)) * noise_scales
        r = f @ true_lambda.T + noise

        sum_r_f = jnp.sum(r[:, :, None] * f[:, None, :], axis=0)
        sum_f_f = jnp.sum(f[:, :, None] * f[:, None, :], axis=0)
        sum_r_r_diag = jnp.sum(r**2, axis=0)

        stats = EMSufficientStats(
            sum_r_f=sum_r_f,
            sum_f_f=sum_f_f,
            sum_r_r_diag=sum_r_r_diag,
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            T=T,
        )

        sigma2_est = update_sigma2(stats, true_lambda)

        assert sigma2_est.shape == (N,)
        assert jnp.allclose(sigma2_est, true_sigma2, rtol=0.15)

    def test_enforces_positivity(self):
        """σ² should always be positive (clipped if necessary)."""
        from bellman_filter_dfsv.core.optimization._em_mstep import update_sigma2
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        N, K, T = 2, 1, 10

        # Construct pathological stats that would give negative variance
        stats = EMSufficientStats(
            sum_r_f=jnp.ones((N, K)) * 100,
            sum_f_f=jnp.ones((K, K)),
            sum_r_r_diag=jnp.ones(N),  # Very small r²
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            T=T,
        )

        lambda_r = jnp.ones((N, K))
        sigma2_est = update_sigma2(stats, lambda_r)

        assert (sigma2_est > 0).all()


class TestMStepPhiH:
    """Tests for log-vol AR coefficient M-step."""

    def test_recovers_ar_coefficient(self):
        """
        Given h_t = μ + φ_h (h_{t-1} - μ) + η_t,
        φ_h = Σ E[(h_t - μ)(h_{t-1} - μ)] / Σ E[(h_{t-1} - μ)²]
        """
        from bellman_filter_dfsv.core.optimization._em_mstep import update_Phi_h
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        K, T = 1, 1000
        true_phi_h = 0.95
        true_mu = -1.0
        true_q_h = 0.1

        key = jr.PRNGKey(111)

        # Simulate AR(1) process
        h = jnp.zeros(T)
        h = h.at[0].set(true_mu)
        for t in range(1, T):
            key, subkey = jr.split(key)
            h = h.at[t].set(
                true_mu
                + true_phi_h * (h[t - 1] - true_mu)
                + jr.normal(subkey) * jnp.sqrt(true_q_h)
            )

        # Compute RAW sufficient stats (update_Phi_h centers internally using mu)
        sum_h_hprev_raw = jnp.sum(h[1:] * h[:-1])
        sum_hprev_hprev_raw = jnp.sum(h[:-1] ** 2)

        stats = EMSufficientStats(
            sum_r_f=jnp.zeros((1, K)),
            sum_f_f=jnp.zeros((K, K)),
            sum_r_r_diag=jnp.zeros(1),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.array([jnp.sum(h[1:])]),
            sum_hprev=jnp.array([jnp.sum(h[:-1])]),
            sum_h_h=jnp.array([[jnp.sum(h[1:] ** 2)]]),
            sum_h_hprev=jnp.array([[sum_h_hprev_raw]]),
            sum_hprev_hprev=jnp.array([[sum_hprev_hprev_raw]]),
            T=T,
        )

        phi_h_est = update_Phi_h(stats, jnp.array([true_mu]))

        assert phi_h_est.shape == (K, K)
        assert jnp.allclose(jnp.diag(phi_h_est), true_phi_h, atol=0.05)


class TestMStepMu:
    def test_recovers_mu_from_ar_process(self):
        from bellman_filter_dfsv.core.optimization._em_mstep import update_mu
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        K, T = 1, 2000
        true_phi_h = 0.9
        true_mu = -1.5
        true_q_h = 0.15

        key = jr.PRNGKey(222)

        h = jnp.zeros(T)
        h = h.at[0].set(true_mu)
        for t in range(1, T):
            key, subkey = jr.split(key)
            h = h.at[t].set(
                true_mu
                + true_phi_h * (h[t - 1] - true_mu)
                + jr.normal(subkey) * jnp.sqrt(true_q_h)
            )

        stats = EMSufficientStats(
            sum_r_f=jnp.zeros((1, K)),
            sum_f_f=jnp.zeros((K, K)),
            sum_r_r_diag=jnp.zeros(1),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.array([jnp.sum(h[1:])]),
            sum_hprev=jnp.array([jnp.sum(h[:-1])]),
            sum_h_h=jnp.array([[jnp.sum(h[1:] ** 2)]]),
            sum_h_hprev=jnp.array([[jnp.sum(h[1:] * h[:-1])]]),
            sum_hprev_hprev=jnp.array([[jnp.sum(h[:-1] ** 2)]]),
            T=T,
        )

        Phi_h = jnp.array([[true_phi_h]])
        mu_est = update_mu(stats, Phi_h)

        assert mu_est.shape == (K,)
        assert jnp.allclose(mu_est, true_mu, atol=0.1)


class TestMStepQh:
    def test_recovers_innovation_variance(self):
        from bellman_filter_dfsv.core.optimization._em_mstep import update_Q_h
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        K, T = 1, 2000
        true_phi_h = 0.9
        true_mu = -1.0
        true_q_h = 0.2

        key = jr.PRNGKey(333)

        h = jnp.zeros(T)
        h = h.at[0].set(true_mu)
        for t in range(1, T):
            key, subkey = jr.split(key)
            h = h.at[t].set(
                true_mu
                + true_phi_h * (h[t - 1] - true_mu)
                + jr.normal(subkey) * jnp.sqrt(true_q_h)
            )

        stats = EMSufficientStats(
            sum_r_f=jnp.zeros((1, K)),
            sum_f_f=jnp.zeros((K, K)),
            sum_r_r_diag=jnp.zeros(1),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            sum_h=jnp.array([jnp.sum(h[1:])]),
            sum_hprev=jnp.array([jnp.sum(h[:-1])]),
            sum_h_h=jnp.array([[jnp.sum(h[1:] ** 2)]]),
            sum_h_hprev=jnp.array([[jnp.sum(h[1:] * h[:-1])]]),
            sum_hprev_hprev=jnp.array([[jnp.sum(h[:-1] ** 2)]]),
            T=T,
        )

        mu = jnp.array([true_mu])
        Phi_h = jnp.array([[true_phi_h]])
        Q_h_est = update_Q_h(stats, mu, Phi_h)

        assert Q_h_est.shape == (K, K)
        assert jnp.allclose(jnp.diag(Q_h_est), true_q_h, rtol=0.15)


class TestMStepPhiF:
    def test_recovers_factor_ar_coefficient(self):
        from bellman_filter_dfsv.core.optimization._em_mstep import update_Phi_f
        from bellman_filter_dfsv.core.optimization._em_suffstats import (
            EMSufficientStats,
        )

        K, T = 1, 1000
        true_phi_f = 0.85

        key = jr.PRNGKey(444)

        f = jnp.zeros(T)
        for t in range(1, T):
            key, subkey = jr.split(key)
            f = f.at[t].set(true_phi_f * f[t - 1] + jr.normal(subkey))

        w = jnp.ones(T - 1)
        sum_w_f_fprev = jnp.sum(w * f[1:] * f[:-1])
        sum_w_fprev_sq = jnp.sum(w * f[:-1] ** 2)

        stats = EMSufficientStats(
            sum_r_f=jnp.zeros((1, K)),
            sum_f_f=jnp.zeros((K, K)),
            sum_r_r_diag=jnp.zeros(1),
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.ones(K) * (T - 1),
            sum_exp_neg_h_f_fprev_diag=jnp.array([sum_w_f_fprev]),
            sum_exp_neg_h_fprev_sq=jnp.array([sum_w_fprev_sq]),
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            T=T,
        )

        Phi_f_est = update_Phi_f(stats)

        assert Phi_f_est.shape == (K, K)
        assert jnp.allclose(jnp.diag(Phi_f_est), true_phi_f, atol=0.05)
