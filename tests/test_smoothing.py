"""Comprehensive tests for rts_smoother and run_rbps (v2 architecture).

Feature: v2-architecture-migration
Property 2: Test Suite Consolidation and Coverage
Target Coverage: smoothing.py from 24% → 85%+
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import dfsv_params_strategy
from hypothesis import given, settings

from bellman_filter_dfsv import BellmanFilter, DFSVParams, rts_smoother, run_rbps

jax.config.update("jax_enable_x64", True)


class TestRTSSmoother:
    def test_rts_smoother_output_shapes(self, simple_params):
        bf = BellmanFilter(simple_params)
        observations = jnp.zeros((50, 3))
        filter_result = bf.filter(observations)

        smoother_result = rts_smoother(
            simple_params, filter_result.means, filter_result.infos
        )

        T = 50
        K = 1
        expected_dim = 2 * K

        assert smoother_result.smoothed_means.shape == (T, expected_dim)
        assert smoother_result.smoothed_covs.shape == (T, expected_dim, expected_dim)
        assert smoother_result.smoothed_lag1_covs.shape == (
            T,
            expected_dim,
            expected_dim,
        )

    def test_rts_smoother_final_state_matches_filtered(self, simple_params):
        bf = BellmanFilter(simple_params)
        observations = jnp.zeros((50, 3))
        filter_result = bf.filter(observations)

        smoother_result = rts_smoother(
            simple_params, filter_result.means, filter_result.infos
        )

        filtered_mean_final = filter_result.means[-1]
        smoothed_mean_final = smoother_result.smoothed_means[-1]

        assert jnp.allclose(filtered_mean_final, smoothed_mean_final, atol=1e-6)

    def test_rts_smoother_reduces_uncertainty(self, simple_params):
        bf = BellmanFilter(simple_params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (50, 3)) * 0.1
        filter_result = bf.filter(observations)

        smoother_result = rts_smoother(
            simple_params, filter_result.means, filter_result.infos
        )

        from bellman_filter_dfsv._bellman_math import invert_info_matrix

        filter_covs = jax.vmap(invert_info_matrix)(filter_result.infos)

        for t in range(filter_covs.shape[0] - 1):
            filter_trace = jnp.trace(filter_covs[t])
            smooth_trace = jnp.trace(smoother_result.smoothed_covs[t])

            assert smooth_trace <= filter_trace + 1e-6

    def test_rts_smoother_covariances_are_positive_definite(self, simple_params):
        bf = BellmanFilter(simple_params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1
        filter_result = bf.filter(observations)

        smoother_result = rts_smoother(
            simple_params, filter_result.means, filter_result.infos
        )

        for t in range(smoother_result.smoothed_covs.shape[0]):
            eigenvalues = jnp.linalg.eigvalsh(smoother_result.smoothed_covs[t])
            assert jnp.all(eigenvalues > -1e-6), (
                f"Smoothed covariance at t={t} is not PSD"
            )

    def test_rts_smoother_is_jit_compatible(self, simple_params):
        bf = BellmanFilter(simple_params)
        observations = jnp.zeros((10, 3))
        filter_result = bf.filter(observations)

        @jax.jit
        def run_smoother(params, means, infos):
            return rts_smoother(params, means, infos)

        smoother_result = run_smoother(
            simple_params, filter_result.means, filter_result.infos
        )

        assert smoother_result.smoothed_means.shape == (10, 2)

    def test_rts_smoother_handles_different_dimensions(self):
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

                bf = BellmanFilter(params)
                observations = jnp.zeros((20, N))
                filter_result = bf.filter(observations)

                smoother_result = rts_smoother(
                    params, filter_result.means, filter_result.infos
                )

                assert smoother_result.smoothed_means.shape == (20, 2 * K)

    @given(params=dfsv_params_strategy(N=3, K=1))
    @settings(max_examples=5, deadline=10000)
    @pytest.mark.property
    def test_property_rts_smoother_always_produces_valid_output(self, params):
        bf = BellmanFilter(params)
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (20, 3)) * 0.1
        filter_result = bf.filter(observations)

        smoother_result = rts_smoother(params, filter_result.means, filter_result.infos)

        assert jnp.all(jnp.isfinite(smoother_result.smoothed_means))
        assert jnp.all(jnp.isfinite(smoother_result.smoothed_covs))
        assert jnp.all(jnp.isfinite(smoother_result.smoothed_lag1_covs))


class TestRBPS:
    def test_rbps_output_shapes(self, simple_params):
        observations = jnp.zeros((30, 3))

        num_particles = 50
        num_trajectories = 10

        result = run_rbps(
            simple_params,
            observations,
            num_particles=num_particles,
            num_trajectories=num_trajectories,
            seed=42,
        )

        T = 30
        K = 1

        assert result.h_samples.shape == (num_trajectories, T, K)
        assert result.f_smooth_means.shape == (num_trajectories, T, K)
        assert result.f_smooth_covs.shape == (num_trajectories, T, K, K)
        assert result.f_smooth_lag1_covs.shape == (num_trajectories, T - 1, K, K)

    def test_rbps_runs_without_error(self, simple_params):
        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (30, 3)) * 0.1

        result = run_rbps(
            simple_params,
            observations,
            num_particles=50,
            num_trajectories=10,
            seed=42,
        )

        assert jnp.all(jnp.isfinite(result.h_samples))
        assert jnp.all(jnp.isfinite(result.f_smooth_means))
        assert jnp.all(jnp.isfinite(result.f_smooth_covs))

    def test_rbps_f_smooth_covs_are_positive_definite(self, simple_params):
        observations = jnp.zeros((20, 3))

        result = run_rbps(
            simple_params,
            observations,
            num_particles=50,
            num_trajectories=10,
            seed=42,
        )

        for m in range(result.f_smooth_covs.shape[0]):
            for t in range(result.f_smooth_covs.shape[1]):
                cov = result.f_smooth_covs[m, t]
                eigenvalues = jnp.linalg.eigvalsh(cov)
                assert jnp.all(eigenvalues > -1e-6), (
                    f"Cov at trajectory={m}, t={t} is not PSD"
                )

    def test_rbps_different_particle_counts(self, simple_params):
        observations = jnp.zeros((20, 3))

        for num_particles in [30, 50, 100]:
            result = run_rbps(
                simple_params,
                observations,
                num_particles=num_particles,
                num_trajectories=5,
                seed=42,
            )

            assert result.h_samples.shape == (5, 20, 1)

    def test_rbps_different_trajectory_counts(self, simple_params):
        observations = jnp.zeros((20, 3))

        for num_trajectories in [5, 10, 20]:
            result = run_rbps(
                simple_params,
                observations,
                num_particles=50,
                num_trajectories=num_trajectories,
                seed=42,
            )

            assert result.h_samples.shape == (num_trajectories, 20, 1)

    def test_rbps_reproducibility_with_same_seed(self, simple_params):
        observations = jnp.zeros((20, 3))

        result1 = run_rbps(
            simple_params,
            observations,
            num_particles=50,
            num_trajectories=10,
            seed=42,
        )

        result2 = run_rbps(
            simple_params,
            observations,
            num_particles=50,
            num_trajectories=10,
            seed=42,
        )

        assert jnp.allclose(result1.h_samples, result2.h_samples)
        assert jnp.allclose(result1.f_smooth_means, result2.f_smooth_means)

    def test_rbps_handles_different_dimensions(self):
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

                observations = jnp.zeros((15, N))

                result = run_rbps(
                    params,
                    observations,
                    num_particles=30,
                    num_trajectories=5,
                    seed=42,
                )

                assert result.h_samples.shape == (5, 15, K)
                assert result.f_smooth_means.shape == (5, 15, K)
