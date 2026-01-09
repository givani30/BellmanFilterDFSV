"""
Academic verification tests for RTS smoother.

Tests the smoother against known theoretical properties:
1. Smoothed MSE <= Filtered MSE (smoother uses future data)
2. Smoothed covariance <= Filtered covariance (trace inequality)
3. Boundary condition: at T, smoothed == filtered
4. Linear system recovery: compare against ground truth on synthetic data
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


class TestRTSSmootherAcademicVerification:
    """Academic verification tests for RTS smoother correctness."""

    @pytest.fixture
    def simple_linear_system(self):
        """
        Create a simple linear state-space model for verification.

        Model:
            x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
            y_t = H x_t + v_t,      v_t ~ N(0, R)

        Using scalar system for analytical tractability:
            F = 0.9 (AR coefficient)
            H = 1.0 (direct observation)
            Q = 0.1 (process noise)
            R = 0.5 (measurement noise)
        """
        return {
            "F": 0.9,
            "H": 1.0,
            "Q": 0.1,
            "R": 0.5,
            "x0": 0.0,
            "P0": 1.0,
        }

    @pytest.fixture
    def generate_synthetic_data(self, simple_linear_system):
        """Generate synthetic data from the linear system with known ground truth."""

        def _generate(T: int, seed: int = 42):
            key = jr.PRNGKey(seed)
            params = simple_linear_system

            F, H, Q, R = params["F"], params["H"], params["Q"], params["R"]
            x0, P0 = params["x0"], params["P0"]

            # Generate true states
            key, k1, k2, k3 = jr.split(key, 4)
            x_true = np.zeros(T)
            x_true[0] = x0 + np.sqrt(P0) * float(jr.normal(k1))

            process_noise = np.array(jr.normal(k2, (T,))) * np.sqrt(Q)
            for t in range(1, T):
                x_true[t] = F * x_true[t - 1] + process_noise[t]

            # Generate observations
            obs_noise = np.array(jr.normal(k3, (T,))) * np.sqrt(R)
            y = H * x_true + obs_noise

            return {
                "x_true": x_true,
                "y": y,
                "params": params,
                "T": T,
            }

        return _generate

    def _run_kalman_filter(self, y, params):
        """Run standard Kalman filter on scalar system."""
        T = len(y)
        F, H, Q, R = params["F"], params["H"], params["Q"], params["R"]
        x0, P0 = params["x0"], params["P0"]

        # Storage
        x_filt = np.zeros(T)
        P_filt = np.zeros(T)
        x_pred = np.zeros(T)
        P_pred = np.zeros(T)

        # Initialize
        x_pred[0] = x0
        P_pred[0] = P0

        for t in range(T):
            # Update (incorporate observation)
            K = P_pred[t] * H / (H * P_pred[t] * H + R)
            x_filt[t] = x_pred[t] + K * (y[t] - H * x_pred[t])
            P_filt[t] = (1 - K * H) * P_pred[t]

            # Predict (for next step)
            if t < T - 1:
                x_pred[t + 1] = F * x_filt[t]
                P_pred[t + 1] = F * P_filt[t] * F + Q

        return x_filt, P_filt, x_pred, P_pred

    def _run_rts_smoother(self, x_filt, P_filt, x_pred, P_pred, params):
        """Run RTS smoother on scalar system."""
        T = len(x_filt)
        F = params["F"]

        x_smooth = np.zeros(T)
        P_smooth = np.zeros(T)

        # Initialize: at T, smoothed = filtered
        x_smooth[T - 1] = x_filt[T - 1]
        P_smooth[T - 1] = P_filt[T - 1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            # Smoother gain
            J = P_filt[t] * F / P_pred[t + 1] if P_pred[t + 1] > 1e-10 else 0.0

            # Smoothed estimates
            x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t + 1] - P_pred[t + 1])

        return x_smooth, P_smooth

    def test_smoothed_mse_less_than_filtered_mse(self, generate_synthetic_data):
        """
        Smoother should have lower or equal MSE compared to filter.

        This is a fundamental property: the smoother uses future observations,
        so it should never be worse than the filter.
        """
        data = generate_synthetic_data(T=200, seed=123)
        x_true = data["x_true"]
        y = data["y"]
        params = data["params"]

        # Run filter
        x_filt, P_filt, x_pred, P_pred = self._run_kalman_filter(y, params)

        # Run smoother
        x_smooth, P_smooth = self._run_rts_smoother(
            x_filt, P_filt, x_pred, P_pred, params
        )

        # Compute MSE
        mse_filt = np.mean((x_filt - x_true) ** 2)
        mse_smooth = np.mean((x_smooth - x_true) ** 2)

        print(f"Filtered MSE: {mse_filt:.6f}")
        print(f"Smoothed MSE: {mse_smooth:.6f}")
        print(f"Improvement: {(1 - mse_smooth / mse_filt) * 100:.1f}%")

        assert mse_smooth <= mse_filt * 1.01, (
            f"Smoothed MSE ({mse_smooth:.6f}) should be <= Filtered MSE ({mse_filt:.6f})"
        )

    def test_smoothed_covariance_less_than_filtered(self, generate_synthetic_data):
        """
        Smoothed covariance should be <= filtered covariance at each time step.

        P_{t|T} <= P_{t|t} for all t (in PSD sense, here just scalar comparison)
        """
        data = generate_synthetic_data(T=100, seed=456)
        y = data["y"]
        params = data["params"]

        x_filt, P_filt, x_pred, P_pred = self._run_kalman_filter(y, params)
        x_smooth, P_smooth = self._run_rts_smoother(
            x_filt, P_filt, x_pred, P_pred, params
        )

        # Check at each time step (except possibly edges due to boundary effects)
        for t in range(1, len(y) - 1):
            assert P_smooth[t] <= P_filt[t] + 1e-10, (
                f"At t={t}: P_smooth ({P_smooth[t]:.6f}) > P_filt ({P_filt[t]:.6f})"
            )

        # Check average
        avg_reduction = np.mean(P_filt - P_smooth)
        print(f"Average covariance reduction: {avg_reduction:.6f}")
        assert avg_reduction >= 0, "Smoother should reduce average covariance"

    def test_boundary_condition_at_final_time(self, generate_synthetic_data):
        """
        At final time T, smoothed estimate should equal filtered estimate.

        There is no future data at T, so x_{T|T} == x_{T|T} (trivially).
        """
        data = generate_synthetic_data(T=50, seed=789)
        y = data["y"]
        params = data["params"]

        x_filt, P_filt, x_pred, P_pred = self._run_kalman_filter(y, params)
        x_smooth, P_smooth = self._run_rts_smoother(
            x_filt, P_filt, x_pred, P_pred, params
        )

        T = len(y)

        np.testing.assert_allclose(
            x_smooth[T - 1],
            x_filt[T - 1],
            atol=1e-10,
            err_msg="Smoothed state at T should equal filtered state",
        )
        np.testing.assert_allclose(
            P_smooth[T - 1],
            P_filt[T - 1],
            atol=1e-10,
            err_msg="Smoothed covariance at T should equal filtered covariance",
        )

    def test_smoothing_improvement_increases_toward_center(
        self, generate_synthetic_data
    ):
        """
        Smoothing improvement should be larger in the middle of the series.

        At boundaries, smoother has less future/past data to use.
        """
        data = generate_synthetic_data(T=200, seed=111)
        y = data["y"]
        params = data["params"]

        x_filt, P_filt, x_pred, P_pred = self._run_kalman_filter(y, params)
        x_smooth, P_smooth = self._run_rts_smoother(
            x_filt, P_filt, x_pred, P_pred, params
        )

        # Compare improvement at edges vs center
        edge_size = 20
        center_start = len(y) // 2 - 25
        center_end = len(y) // 2 + 25

        edge_improvement = np.mean(P_filt[:edge_size] - P_smooth[:edge_size])
        center_improvement = np.mean(
            P_filt[center_start:center_end] - P_smooth[center_start:center_end]
        )

        print(f"Edge improvement: {edge_improvement:.6f}")
        print(f"Center improvement: {center_improvement:.6f}")

        # Center improvement should be at least as good as edge
        # (This is a soft check - depends on system dynamics)
        assert center_improvement >= edge_improvement * 0.5, (
            "Center should benefit more from smoothing than edges"
        )

    def test_monte_carlo_coverage(self, simple_linear_system):
        """
        Verify that smoothed confidence intervals have correct coverage.

        If P_smooth is correct, then ~95% of true states should fall within
        +/- 1.96 * sqrt(P_smooth) of x_smooth.
        """
        params = simple_linear_system
        n_trials = 100
        T = 50
        coverage_count = 0
        total_points = 0

        for seed in range(n_trials):
            key = jr.PRNGKey(seed)
            F, H, Q, R = params["F"], params["H"], params["Q"], params["R"]
            x0, P0 = params["x0"], params["P0"]

            # Generate data
            key, k1, k2, k3 = jr.split(key, 4)
            x_true = np.zeros(T)
            x_true[0] = x0 + np.sqrt(P0) * float(jr.normal(k1))

            process_noise = np.array(jr.normal(k2, (T,))) * np.sqrt(Q)
            for t in range(1, T):
                x_true[t] = F * x_true[t - 1] + process_noise[t]

            obs_noise = np.array(jr.normal(k3, (T,))) * np.sqrt(R)
            y = H * x_true + obs_noise

            # Filter and smooth
            x_filt, P_filt, x_pred, P_pred = self._run_kalman_filter(y, params)
            x_smooth, P_smooth = self._run_rts_smoother(
                x_filt, P_filt, x_pred, P_pred, params
            )

            # Check coverage (exclude first few for burn-in)
            for t in range(5, T):
                std = np.sqrt(P_smooth[t])
                lower = x_smooth[t] - 1.96 * std
                upper = x_smooth[t] + 1.96 * std
                if lower <= x_true[t] <= upper:
                    coverage_count += 1
                total_points += 1

        coverage = coverage_count / total_points
        print(f"95% CI coverage: {coverage * 100:.1f}%")

        # Should be close to 95% (allow some tolerance)
        assert 0.90 <= coverage <= 0.99, (
            f"95% CI coverage should be ~95%, got {coverage * 100:.1f}%"
        )

    @pytest.mark.skip(reason="Requires v1 architecture (removed)")
    def test_bif_smoother_matches_numpy_oracle(self, generate_synthetic_data):
        """
        CRITICAL TEST: Verifies the BIF production JAX smoother against the
        trusted NumPy 'Oracle' implementation defined in this test file.

        If this fails, the EM algorithm will be optimizing noise.

        We test using a simplified DFSV model that reduces to a linear Gaussian
        state-space model (by zeroing out the SV components).
        """
        from bellman_filter_dfsv.core.filters.bellman_information import (
            DFSVBellmanInformationFilter,
        )
        from bellman_filter_dfsv.core.models.dfsv import DFSVParamsDataclass

        data = generate_synthetic_data(T=100, seed=999)
        y = data["y"]
        params = data["params"]

        F_val = params["F"]
        H_val = params["H"]
        Q_val = params["Q"]
        R_val = params["R"]

        params_jax = DFSVParamsDataclass(
            N=1,
            K=1,
            lambda_r=jnp.array([[H_val]]),
            Phi_f=jnp.array([[F_val]]),
            Phi_h=jnp.array([[0.0]]),
            mu=jnp.array([0.0]),
            Q_h=jnp.array([[Q_val]]),
            sigma2=jnp.array([R_val]),
        )

        bif = DFSVBellmanInformationFilter(N=1, K=1)

        y_2d = jnp.array(y)[:, None]
        states, covs, ll = bif.filter(params_jax, y_2d)

        print(f"Filter completed: ll = {ll:.2f}")
        print(f"filtered_states shape: {states.shape}")

        smoothed_states, smoothed_covs, lag1_covs = bif.smooth(params_jax)

        x_filt_oracle, P_filt_oracle, x_pred_oracle, P_pred_oracle = (
            self._run_kalman_filter(y, params)
        )
        x_smooth_oracle, P_smooth_oracle = self._run_rts_smoother(
            x_filt_oracle, P_filt_oracle, x_pred_oracle, P_pred_oracle, params
        )

        jax_smooth_means = np.array(smoothed_states[:, 0])
        jax_smooth_vars = np.array(smoothed_covs[:, 0, 0])

        err_mean = np.abs(jax_smooth_means - x_smooth_oracle).max()
        err_var = np.abs(jax_smooth_vars - P_smooth_oracle).max()

        print(f"Max discrepancy (Mean): {err_mean:.2e}")
        print(f"Max discrepancy (Var): {err_var:.2e}")

        corr = np.corrcoef(jax_smooth_means, x_smooth_oracle)[0, 1]
        print(f"Correlation between JAX and Oracle means: {corr:.6f}")

        mse_jax = np.mean((jax_smooth_means - data["x_true"]) ** 2)
        mse_oracle = np.mean((x_smooth_oracle - data["x_true"]) ** 2)
        print(f"MSE (JAX vs true): {mse_jax:.6f}")
        print(f"MSE (Oracle vs true): {mse_oracle:.6f}")

        assert corr > 0.95, f"JAX smoother poorly correlated with Oracle! corr={corr}"

        print("SUCCESS: BIF smoother is reasonably aligned with NumPy oracle.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
