"""Comprehensive tests for BellmanFilter and ParticleFilter (v2 architecture).

Feature: v2-architecture-migration
Property 2: Test Suite Consolidation and Coverage
Target Coverage: filters.py from 55% → 90%+
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import dfsv_params_strategy
from hypothesis import given, settings
from hypothesis import strategies as st

from bellman_filter_dfsv import BellmanFilter, DFSVParams, ParticleFilter

jax.config.update("jax_enable_x64", True)


class TestBellmanFilter:
    def test_initialization_shapes_single_factor(self, simple_params):
        bf = BellmanFilter(simple_params)
        state = bf.initialize()

        K = 1
        expected_dim = 2 * K

        assert state.mean.shape == (expected_dim,)
        assert state.info.shape == (expected_dim, expected_dim)

        assert jnp.all(jnp.isfinite(state.mean))
        assert jnp.all(jnp.isfinite(state.info))

    def test_initialization_shapes_multi_factor(self, multi_factor_params):
        bf = BellmanFilter(multi_factor_params)
        state = bf.initialize()

        K = 2
        expected_dim = 2 * K

        assert state.mean.shape == (expected_dim,)
        assert state.info.shape == (expected_dim, expected_dim)

    def test_initialization_information_matrix_is_positive_definite(
        self, simple_params
    ):
        bf = BellmanFilter(simple_params)
        state = bf.initialize()

        eigenvalues = jnp.linalg.eigvalsh(state.info)
        assert jnp.all(eigenvalues > 0), "Information matrix must be positive definite"

    def test_initialization_mean_uses_prior(self, simple_params):
        bf = BellmanFilter(simple_params)
        state = bf.initialize()

        K = 1
        f0 = state.mean[:K]
        h0 = state.mean[K:]

        assert jnp.allclose(f0, 0.0), "Factor initial mean should be zero"
        assert jnp.allclose(h0, simple_params.mu), (
            "Log-vol should start at long-run mean"
        )

    def test_filter_output_shapes(self, simple_params):
        bf = BellmanFilter(simple_params)
        T, N = 50, 3
        observations = jnp.zeros((T, N))

        result = bf.filter(observations)

        K = 1
        expected_dim = 2 * K

        assert result.means.shape == (T, expected_dim)
        assert result.infos.shape == (T, expected_dim, expected_dim)
        assert result.log_likelihood.shape == ()

    def test_filter_log_likelihood_is_finite(self, simple_params):
        bf = BellmanFilter(simple_params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (50, 3)) * 0.1

        result = bf.filter(observations)

        assert jnp.isfinite(result.log_likelihood)

    def test_filter_information_matrices_remain_positive_definite(self, simple_params):
        bf = BellmanFilter(simple_params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (20, 3)) * 0.1

        result = bf.filter(observations)

        for t in range(result.infos.shape[0]):
            eigenvalues = jnp.linalg.eigvalsh(result.infos[t])
            assert jnp.all(eigenvalues > 0), (
                f"Info matrix at t={t} is not positive definite"
            )

    def test_filter_is_jit_compatible(self, simple_params):
        bf = BellmanFilter(simple_params)

        @jax.jit
        def run_filter(obs):
            return bf.filter(obs)

        observations = jnp.zeros((10, 3))
        result = run_filter(observations)

        assert result.means.shape == (10, 2)

    def test_filter_zero_observations_converge_to_prior(self, simple_params):
        bf = BellmanFilter(simple_params)
        observations = jnp.zeros((100, 3))

        result = bf.filter(observations)

        K = 1
        final_f_mean = result.means[-1, :K]

        assert jnp.abs(final_f_mean[0]) < 0.5

    def test_smooth_state_conversion(self, simple_params):
        bf = BellmanFilter(simple_params)
        observations = jnp.zeros((10, 3))

        result = bf.filter(observations)

        mean, cov = bf.smooth_state(result.means[0], result.infos[0])

        assert mean.shape == result.means[0].shape
        assert cov.shape == result.infos[0].shape

        eigenvalues = jnp.linalg.eigvalsh(cov)
        assert jnp.all(eigenvalues > 0), (
            "Converted covariance must be positive definite"
        )

    def test_filter_handles_different_dimensions(self):
        for N in [2, 3, 5]:
            for K in [1, 2, 3]:
                params = DFSVParams(
                    lambda_r=jnp.ones((N, K)) * 0.8,
                    Phi_f=jnp.eye(K) * 0.7,
                    Phi_h=jnp.eye(K) * 0.95,
                    mu=jnp.ones(K) * (-1.0),
                    sigma2=jnp.ones(N) * 0.3,
                    Q_h=jnp.eye(K) * 0.01,
                )

                bf = BellmanFilter(params)
                observations = jnp.zeros((10, N))
                result = bf.filter(observations)

                assert result.means.shape == (10, 2 * K)
                assert jnp.isfinite(result.log_likelihood)

    @given(
        params=dfsv_params_strategy(N=3, K=1), T=st.integers(min_value=5, max_value=30)
    )
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.property
    def test_property_filter_output_shapes_always_correct(self, params, T):
        bf = BellmanFilter(params)
        observations = jnp.zeros((T, params.lambda_r.shape[0]))

        result = bf.filter(observations)

        K = params.lambda_r.shape[1]
        expected_dim = 2 * K

        assert result.means.shape == (T, expected_dim)
        assert result.infos.shape == (T, expected_dim, expected_dim)

    @given(params=dfsv_params_strategy(N=3, K=1))
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.property
    def test_property_log_likelihood_is_always_finite(self, params):
        bf = BellmanFilter(params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (15, params.lambda_r.shape[0])) * 0.1

        result = bf.filter(observations)

        assert jnp.isfinite(result.log_likelihood)

    @given(params=dfsv_params_strategy(N=3, K=1))
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.property
    def test_property_information_matrices_always_positive_definite(self, params):
        bf = BellmanFilter(params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (10, params.lambda_r.shape[0])) * 0.1

        result = bf.filter(observations)

        for t in range(result.infos.shape[0]):
            eigenvalues = jnp.linalg.eigvalsh(result.infos[t])
            assert jnp.all(eigenvalues > 0)


class TestParticleFilter:
    def test_initialization(self, simple_params):
        pf = ParticleFilter(simple_params, num_particles=100, seed=42)

        assert pf.num_particles == 100
        assert pf.seed == 42
        assert pf.resample_threshold_frac == 0.5

    def test_filter_output_shapes(self, simple_params):
        pf = ParticleFilter(simple_params, num_particles=100, seed=42)
        T, N = 50, 3
        observations = jnp.zeros((T, N))

        result = pf.filter(observations)

        K = 1
        expected_dim = 2 * K

        assert result.means.shape == (T, expected_dim)
        assert result.covs.shape == (T, expected_dim, expected_dim)
        assert result.log_likelihood.shape == ()

    def test_filter_log_likelihood_is_finite(self, simple_params):
        pf = ParticleFilter(simple_params, num_particles=100, seed=42)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (50, 3)) * 0.1

        result = pf.filter(observations)

        assert jnp.isfinite(result.log_likelihood)

    def test_filter_covariances_are_positive_definite(self, simple_params):
        pf = ParticleFilter(simple_params, num_particles=100, seed=42)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (20, 3)) * 0.1

        result = pf.filter(observations)

        for t in range(result.covs.shape[0]):
            eigenvalues = jnp.linalg.eigvalsh(result.covs[t])
            assert jnp.all(eigenvalues >= 0), f"Covariance at t={t} is not PSD"

    def test_filter_is_jit_compatible(self, simple_params):
        pf = ParticleFilter(simple_params, num_particles=100, seed=42)

        @jax.jit
        def run_filter(obs):
            return pf.filter(obs)

        observations = jnp.zeros((10, 3))
        result = run_filter(observations)

        assert result.means.shape == (10, 2)

    def test_filter_different_particle_counts(self, simple_params):
        observations = jnp.zeros((10, 3))

        for num_particles in [50, 100, 500]:
            pf = ParticleFilter(simple_params, num_particles=num_particles, seed=42)
            result = pf.filter(observations)

            assert jnp.isfinite(result.log_likelihood)
            assert result.means.shape == (10, 2)

    def test_filter_different_resampling_thresholds(self, simple_params):
        observations = jnp.zeros((10, 3))

        for threshold in [0.3, 0.5, 0.7]:
            pf = ParticleFilter(
                simple_params,
                num_particles=100,
                resample_threshold_frac=threshold,
                seed=42,
            )
            result = pf.filter(observations)

            assert jnp.isfinite(result.log_likelihood)

    def test_filter_handles_different_dimensions(self):
        for N in [2, 3]:
            for K in [1, 2]:
                params = DFSVParams(
                    lambda_r=jnp.ones((N, K)) * 0.8,
                    Phi_f=jnp.eye(K) * 0.7,
                    Phi_h=jnp.eye(K) * 0.95,
                    mu=jnp.ones(K) * (-1.0),
                    sigma2=jnp.ones(N) * 0.3,
                    Q_h=jnp.eye(K) * 0.01,
                )

                pf = ParticleFilter(params, num_particles=100, seed=42)
                observations = jnp.zeros((10, N))
                result = pf.filter(observations)

                assert result.means.shape == (10, 2 * K)
                assert jnp.isfinite(result.log_likelihood)

    def test_filter_reproducibility_with_same_seed(self, simple_params):
        observations = jnp.zeros((10, 3))

        pf1 = ParticleFilter(simple_params, num_particles=100, seed=42)
        result1 = pf1.filter(observations)

        pf2 = ParticleFilter(simple_params, num_particles=100, seed=42)
        result2 = pf2.filter(observations)

        assert jnp.allclose(result1.log_likelihood, result2.log_likelihood)
        assert jnp.allclose(result1.means, result2.means)

    @given(
        params=dfsv_params_strategy(N=3, K=1), T=st.integers(min_value=5, max_value=30)
    )
    @settings(max_examples=10, deadline=10000)
    @pytest.mark.property
    def test_property_pf_output_shapes_always_correct(self, params, T):
        pf = ParticleFilter(params, num_particles=50, seed=42)
        observations = jnp.zeros((T, params.lambda_r.shape[0]))

        result = pf.filter(observations)

        K = params.lambda_r.shape[1]
        expected_dim = 2 * K

        assert result.means.shape == (T, expected_dim)
        assert result.covs.shape == (T, expected_dim, expected_dim)

    @given(params=dfsv_params_strategy(N=3, K=1))
    @settings(max_examples=10, deadline=10000)
    @pytest.mark.property
    def test_property_pf_log_likelihood_is_always_finite(self, params):
        pf = ParticleFilter(params, num_particles=50, seed=42)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (10, params.lambda_r.shape[0])) * 0.1

        result = pf.filter(observations)

        assert jnp.isfinite(result.log_likelihood)


class TestFilterComparison:
    def test_bellman_vs_particle_filter_convergence(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (50, 3)) * 0.1

        bf = BellmanFilter(simple_params)
        bf_result = bf.filter(observations)

        pf = ParticleFilter(simple_params, num_particles=1000, seed=42)
        pf_result = pf.filter(observations)

        bf_means, _ = bf.smooth_state(bf_result.means[0], bf_result.infos[0])

        correlation = jnp.corrcoef(bf_means, pf_result.means[0])[0, 1]

        assert correlation > 0.8
