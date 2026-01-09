"""Comprehensive tests for fit_mle and fit_em (v2 architecture).

Feature: v2-architecture-migration
Property 2: Test Suite Consolidation and Coverage
Target Coverage: estimation.py from 16% → 80%+
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import dfsv_params_strategy
from hypothesis import given, settings

from bellman_filter_dfsv import BellmanFilter, DFSVParams, fit_em, fit_mle
from bellman_filter_dfsv.estimation import (
    constrain_params_default,
    m_step,
    unconstrain_params_default,
    update_lambda_r,
    update_mu,
    update_Phi_f,
    update_Phi_h,
    update_Q_h,
    update_sigma2,
)

jax.config.update("jax_enable_x64", True)


class TestParameterTransformations:
    def test_constrain_params_default_shapes(self, simple_params):
        unconstrained = unconstrain_params_default(simple_params)
        constrained = constrain_params_default(unconstrained)

        assert constrained.lambda_r.shape == simple_params.lambda_r.shape
        assert constrained.Phi_f.shape == simple_params.Phi_f.shape
        assert constrained.Phi_h.shape == simple_params.Phi_h.shape
        assert constrained.mu.shape == simple_params.mu.shape
        assert constrained.sigma2.shape == simple_params.sigma2.shape
        assert constrained.Q_h.shape == simple_params.Q_h.shape

    def test_constrain_enforces_stability(self, simple_params):
        unconstrained = unconstrain_params_default(simple_params)
        constrained = constrain_params_default(unconstrained)

        phi_f_eigenvalues = jnp.linalg.eigvals(constrained.Phi_f)
        phi_h_eigenvalues = jnp.linalg.eigvals(constrained.Phi_h)

        assert jnp.all(jnp.abs(phi_f_eigenvalues) < 1.0)
        assert jnp.all(jnp.abs(phi_h_eigenvalues) < 1.0)

    def test_constrain_enforces_positive_variances(self, simple_params):
        unconstrained = unconstrain_params_default(simple_params)
        constrained = constrain_params_default(unconstrained)

        assert jnp.all(constrained.sigma2 > 0)
        assert jnp.all(jnp.linalg.eigvalsh(constrained.Q_h) > 0)

    def test_roundtrip_transformation_approximate(self, simple_params):
        unconstrained = unconstrain_params_default(simple_params)
        roundtrip = constrain_params_default(unconstrained)

        assert jnp.allclose(roundtrip.lambda_r, simple_params.lambda_r, atol=1e-3)
        assert jnp.allclose(roundtrip.Phi_f, simple_params.Phi_f, atol=1e-3)
        assert jnp.allclose(roundtrip.Phi_h, simple_params.Phi_h, atol=1e-3)
        assert jnp.allclose(roundtrip.mu, simple_params.mu, atol=1e-3)
        assert jnp.allclose(roundtrip.sigma2, simple_params.sigma2, atol=1e-3)


class TestFitMLE:
    def test_fit_mle_runs_without_error(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, history = fit_mle(
            simple_params, observations, num_steps=5, verbose=False
        )

        assert isinstance(fitted_params, DFSVParams)
        assert len(history) == 5

    def test_fit_mle_loss_decreases(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        _, history = fit_mle(simple_params, observations, num_steps=10, verbose=False)

        assert history[-1] < history[0]

    def test_fit_mle_preserves_parameter_shapes(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, _ = fit_mle(
            simple_params, observations, num_steps=5, verbose=False
        )

        assert fitted_params.lambda_r.shape == simple_params.lambda_r.shape
        assert fitted_params.Phi_f.shape == simple_params.Phi_f.shape
        assert fitted_params.Phi_h.shape == simple_params.Phi_h.shape
        assert fitted_params.mu.shape == simple_params.mu.shape
        assert fitted_params.sigma2.shape == simple_params.sigma2.shape
        assert fitted_params.Q_h.shape == simple_params.Q_h.shape

    def test_fit_mle_respects_constraints(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, _ = fit_mle(
            simple_params, observations, num_steps=10, verbose=False
        )

        assert jnp.all(fitted_params.sigma2 > 0)
        assert jnp.all(jnp.linalg.eigvalsh(fitted_params.Q_h) > 0)

        phi_f_eigenvalues = jnp.linalg.eigvals(fitted_params.Phi_f)
        phi_h_eigenvalues = jnp.linalg.eigvals(fitted_params.Phi_h)

        assert jnp.all(jnp.abs(phi_f_eigenvalues) < 1.0)
        assert jnp.all(jnp.abs(phi_h_eigenvalues) < 1.0)

    def test_fit_mle_custom_optimizer(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        custom_optimizer = optax.sgd(learning_rate=0.01)

        fitted_params, history = fit_mle(
            simple_params,
            observations,
            num_steps=5,
            optimizer=custom_optimizer,
            verbose=False,
        )

        assert isinstance(fitted_params, DFSVParams)
        assert len(history) == 5

    def test_fit_mle_different_learning_rates(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        for lr in [0.001, 0.01, 0.1]:
            fitted_params, history = fit_mle(
                simple_params,
                observations,
                learning_rate=lr,
                num_steps=5,
                verbose=False,
            )

            assert isinstance(fitted_params, DFSVParams)
            assert len(history) == 5

    def test_fit_mle_improves_log_likelihood(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        bf_init = BellmanFilter(simple_params)
        initial_ll = bf_init.filter(observations).log_likelihood

        fitted_params, _ = fit_mle(
            simple_params, observations, num_steps=20, verbose=False
        )

        bf_fitted = BellmanFilter(fitted_params)
        fitted_ll = bf_fitted.filter(observations).log_likelihood

        assert fitted_ll >= initial_ll

    @given(params=dfsv_params_strategy(N=3, K=1))
    @settings(max_examples=5, deadline=20000)
    @pytest.mark.property
    def test_property_fit_mle_always_returns_valid_params(self, params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (20, 3)) * 0.1

        fitted_params, history = fit_mle(
            params, observations, num_steps=3, verbose=False
        )

        assert jnp.all(jnp.isfinite(fitted_params.lambda_r))
        assert jnp.all(jnp.isfinite(fitted_params.Phi_f))
        assert jnp.all(jnp.isfinite(fitted_params.Phi_h))
        assert jnp.all(jnp.isfinite(fitted_params.mu))
        assert jnp.all(jnp.isfinite(fitted_params.sigma2))
        assert jnp.all(jnp.isfinite(fitted_params.Q_h))


class TestFitEM:
    def test_fit_em_runs_without_error(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, history = fit_em(
            observations,
            simple_params,
            num_particles=50,
            num_trajectories=10,
            max_iters=2,
            verbose=False,
        )

        assert isinstance(fitted_params, DFSVParams)
        assert len(history) == 2

    def test_fit_em_preserves_parameter_shapes(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, _ = fit_em(
            observations,
            simple_params,
            num_particles=50,
            num_trajectories=10,
            max_iters=2,
            verbose=False,
        )

        assert fitted_params.lambda_r.shape == simple_params.lambda_r.shape
        assert fitted_params.Phi_f.shape == simple_params.Phi_f.shape
        assert fitted_params.Phi_h.shape == simple_params.Phi_h.shape
        assert fitted_params.mu.shape == simple_params.mu.shape
        assert fitted_params.sigma2.shape == simple_params.sigma2.shape
        assert fitted_params.Q_h.shape == simple_params.Q_h.shape

    def test_fit_em_respects_constraints(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        fitted_params, _ = fit_em(
            observations,
            simple_params,
            num_particles=50,
            num_trajectories=10,
            max_iters=2,
            verbose=False,
        )

        assert jnp.all(fitted_params.sigma2 > 0)
        assert jnp.all(jnp.linalg.eigvalsh(fitted_params.Q_h) > 0)


class TestMStepFunctions:
    @pytest.fixture
    def mock_suffstats(self):
        from bellman_filter_dfsv.types import EMSufficientStats

        K = 1
        N = 3
        T = 30

        return EMSufficientStats(
            sum_r_f=jnp.ones((N, K)) * 10.0,
            sum_f_f=jnp.eye(K) * 20.0,
            sum_r_r_diag=jnp.ones(N) * 50.0,
            sum_f_fprev=jnp.eye(K) * 15.0,
            sum_fprev_fprev=jnp.eye(K) * 20.0,
            sum_exp_neg_h=jnp.ones(K) * 25.0,
            sum_exp_neg_h_f_fprev_diag=jnp.ones(K) * 12.0,
            sum_exp_neg_h_fprev_sq=jnp.ones(K) * 18.0,
            sum_h=jnp.ones(K) * (-30.0),
            sum_hprev=jnp.ones(K) * (-29.0),
            sum_h_h=jnp.eye(K) * 40.0,
            sum_h_hprev=jnp.eye(K) * 38.0,
            sum_hprev_hprev=jnp.eye(K) * 39.0,
            T=T,
        )

    def test_update_lambda_r(self, mock_suffstats):
        lambda_r = update_lambda_r(mock_suffstats)

        N = 3
        K = 1
        assert lambda_r.shape == (N, K)
        assert jnp.all(jnp.isfinite(lambda_r))

    def test_update_sigma2(self, mock_suffstats):
        lambda_r = update_lambda_r(mock_suffstats)
        sigma2 = update_sigma2(mock_suffstats, lambda_r)

        N = 3
        assert sigma2.shape == (N,)
        assert jnp.all(sigma2 > 0)
        assert jnp.all(jnp.isfinite(sigma2))

    def test_update_Phi_f(self, mock_suffstats):
        Phi_f = update_Phi_f(mock_suffstats)

        K = 1
        assert Phi_f.shape == (K, K)
        assert jnp.all(jnp.abs(jnp.diag(Phi_f)) < 1.0)

    def test_update_Phi_h(self, mock_suffstats):
        mu = jnp.array([-1.0])
        Phi_h = update_Phi_h(mock_suffstats, mu)

        K = 1
        assert Phi_h.shape == (K, K)
        assert jnp.all(jnp.abs(jnp.diag(Phi_h)) < 1.0)

    def test_update_mu(self, mock_suffstats):
        Phi_h = jnp.array([[0.95]])
        mu = update_mu(mock_suffstats, Phi_h)

        K = 1
        assert mu.shape == (K,)
        assert jnp.all(jnp.isfinite(mu))

    def test_update_Q_h(self, mock_suffstats):
        mu = jnp.array([-1.0])
        Phi_h = jnp.array([[0.95]])
        Q_h = update_Q_h(mock_suffstats, mu, Phi_h)

        K = 1
        assert Q_h.shape == (K, K)
        assert jnp.all(jnp.linalg.eigvalsh(Q_h) > 0)

    def test_m_step_integration(self, mock_suffstats):
        lambda_r, sigma2, Phi_f, mu, Phi_h, Q_h = m_step(mock_suffstats)

        K = 1
        N = 3

        assert lambda_r.shape == (N, K)
        assert sigma2.shape == (N,)
        assert Phi_f.shape == (K, K)
        assert mu.shape == (K,)
        assert Phi_h.shape == (K, K)
        assert Q_h.shape == (K, K)

        assert jnp.all(jnp.isfinite(lambda_r))
        assert jnp.all(sigma2 > 0)
        assert jnp.all(jnp.abs(jnp.diag(Phi_f)) < 1.0)
        assert jnp.all(jnp.abs(jnp.diag(Phi_h)) < 1.0)
        assert jnp.all(jnp.linalg.eigvalsh(Q_h) > 0)
