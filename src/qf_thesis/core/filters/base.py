# src/qf_thesis/core/filters/base.py
"""
Base class for filters applied to Dynamic Factor Stochastic Volatility models.
"""

import warnings
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# Assuming DFSVParamsDataclass will be importable from the models directory
# We'll use absolute imports once the package structure is fully set up
# For now, let's use a placeholder or assume it's available via qf_thesis.models.dfsv
from qf_thesis.models.dfsv import DFSVParamsDataclass

# Try importing tqdm for progress bars, provide a fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm iterator if tqdm is not installed."""
        warnings.warn("tqdm not installed. Progress bars will not be shown.")
        return iterable

class DFSVFilter:
    """
    Base class for filters applied to Dynamic Factor Stochastic Volatility models.

    Provides the foundation for implementing various Kalman filter variants
    (EKF, UKF) and serves as a base for the Particle Filter, handling the
    joint filtering of latent factors and log-volatilities.

    Attributes:
        N (int): Number of observed series.
        K (int): Number of factors.
        state_dim (int): Dimension of the state vector (2 * K).
        is_filtered (bool): Flag indicating if the filter has been run.
        is_smoothed (bool): Flag indicating if the smoother has been run.
        filtered_states (Optional[np.ndarray]): Filtered state estimates (T, state_dim).
        filtered_covs (Optional[np.ndarray]): Filtered state covariances (T, state_dim, state_dim).
        smoothed_states (Optional[np.ndarray]): Smoothed state estimates (T, state_dim).
        smoothed_covs (Optional[np.ndarray]): Smoothed state covariances (T, state_dim, state_dim).
        log_likelihood (Optional[float]): Total log-likelihood from the filter pass.
        params (Optional[DFSVParamsDataclass]): Model parameters used by the filter (set by subclasses).
    """

    def __init__(self, N: int, K: int):
        """
        Initialize the base filter with static parameters N and K.

        Args:
            N: Number of observed series.
            K: Number of factors.
        """
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer.")

        self.N: int = N
        self.K: int = K
        self.state_dim: int = 2 * self.K

        # Flags
        self.is_filtered: bool = False
        self.is_smoothed: bool = False

        # Storage (initialized to None)
        self.filtered_states: np.ndarray | None = None
        self.filtered_covs: np.ndarray | None = None
        self.smoothed_states: np.ndarray | None = None
        self.smoothed_covs: np.ndarray | None = None
        self.log_likelihood: float | None = None
        self.params: DFSVParamsDataclass | None = None # To be set by subclasses if needed

    def initialize_state(
        self, params: DFSVParamsDataclass
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize state vector and covariance matrix using JAX arrays.

        Args:
            params: Parameters of the DFSV model (as a JAX dataclass).

        Returns:
            Tuple containing:
                - Initial state vector (state_dim, 1).
                - Initial state covariance matrix (state_dim, state_dim).
        """
        # Initialize factors to zero
        initial_factors = jnp.zeros((self.K, 1))

        # Initialize log-volatilities to the unconditional mean
        # Ensure mu is a column vector
        initial_log_vols = params.mu.reshape(-1, 1)

        # Combine into state vector [factors; log_vols]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance (identity)
        P_f = jnp.eye(self.K)

        # Solve discrete Lyapunov equation for log-volatility covariance using JAX
        @jit
        def solve_discrete_lyapunov_jax(
            Phi: jnp.ndarray, Q: jnp.ndarray, num_iters: int = 30
        ) -> jnp.ndarray:
            """Solve P = Phi @ P @ Phi.T + Q using iteration (JAX version)."""
            P = Q
            # Consider using lax.fori_loop for fixed iterations if preferred
            for _ in range(num_iters):
                P = Phi @ P @ Phi.T + Q
            # Ensure symmetry
            return (P + P.T) / 2.0

        # Ensure inputs are JAX arrays (should be if params is DFSVParamsDataclass)
        P_h = solve_discrete_lyapunov_jax(params.Phi_h, params.Q_h)

        # Construct block-diagonal initial covariance matrix
        initial_cov = jnp.block(
            [
                [P_f, jnp.zeros((self.K, self.K))],
                [jnp.zeros((self.K, self.K)), P_h],
            ]
        )

        return initial_state, initial_cov

    def filter(
        self, params: DFSVParamsDataclass, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Kalman filter (requires implementation by subclasses like EKF/UKF).

        Args:
            params: Parameters of the DFSV model (as a JAX dataclass).
            y: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple containing:
                - Filtered states (T, state_dim).
                - Filtered covariances (T, state_dim, state_dim).
                - Total log-likelihood.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        # This base implementation is for Kalman-like filters, not Particle Filters
        # It remains here for potential EKF/UKF subclasses but is overridden by PF.
        raise NotImplementedError("Filter method must be implemented by subclasses")


    def predict(
        self, params: DFSVParamsDataclass, state: jnp.ndarray, cov: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform the prediction step (requires implementation by subclasses).

        Args:
            params: Model parameters.
            state: Current state estimate (state_dim, 1).
            cov: Current state covariance (state_dim, state_dim).

        Returns:
            Tuple containing predicted state and covariance.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def update(
        self,
        params: DFSVParamsDataclass,
        predicted_state: jnp.ndarray,
        predicted_cov: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Perform the update step (requires implementation by subclasses).

        Args:
            params: Model parameters.
            predicted_state: Predicted state (state_dim, 1).
            predicted_cov: Predicted state covariance (state_dim, state_dim).
            observation: Current observation (N, 1).

        Returns:
            Tuple containing updated state, updated covariance, and log-likelihood contribution.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Update method must be implemented by subclasses")

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Rauch-Tung-Striebel (RTS) smoother.

        Must be called after `filter()`. Uses the stored filtered states and
        covariances (expected to be NumPy arrays) and requires the subclass
        to implement `_get_transition_matrix` and `_predict_with_matrix`.

        Returns:
            Tuple containing:
                - Smoothed states (T, state_dim).
                - Smoothed covariances (T, state_dim, state_dim).

        Raises:
            RuntimeError: If `filter()` has not been run first.
        """
        if not self.is_filtered or self.filtered_states is None or self.filtered_covs is None:
            raise RuntimeError(
                "Filter must be run successfully before smoothing."
            )

        T = self.filtered_states.shape[0]
        if T <= 1: # Cannot smooth if T <= 1
             self.smoothed_states = self.filtered_states.copy()
             self.smoothed_covs = self.filtered_covs.copy()
             self.is_smoothed = True
             return self.smoothed_states, self.smoothed_covs

        # Storage for smoothed states and covariances (NumPy arrays)
        smoothed_states = np.zeros_like(self.filtered_states)
        smoothed_covs = np.zeros_like(self.filtered_covs)

        # Initialize with last filtered values
        smoothed_states[T - 1, :] = self.filtered_states[T - 1, :]
        smoothed_covs[T - 1, :, :] = self.filtered_covs[T - 1, :, :]

        # Backward pass (RTS smoother)
        for t in tqdm(range(T - 2, -1, -1), desc="Smoothing Progress", leave=False):
            # Get filtered values at time t
            state_t_filt = self.filtered_states[t, :].reshape(-1, 1) # Ensure column vector
            cov_t_filt = self.filtered_covs[t, :, :]

            # Get smoothed values at time t+1
            state_tp1_smooth = smoothed_states[t + 1, :].reshape(-1, 1) # Ensure column vector
            cov_tp1_smooth = smoothed_covs[t + 1, :, :]

            # Get the linearized transition matrix F_t (NumPy)
            # Note: state_t_filt is (state_dim, 1), but _get_transition_matrix might expect (1, state_dim)
            # Let's assume _get_transition_matrix handles the shape internally or expects (state_dim, 1)
            F_t = self._get_transition_matrix(state_t_filt) # Requires NumPy implementation

            # Predict state and covariance from t to t+1 using filtered estimate at t (NumPy)
            state_tp1_pred, cov_tp1_pred = self._predict_with_matrix(
                state_t_filt, cov_t_filt, F_t # Requires NumPy implementation
            )

            # Compute smoother gain K_t (using NumPy)
            # Add small epsilon for numerical stability if needed
            try:
                inv_cov_tp1_pred = np.linalg.inv(cov_tp1_pred)
            except np.linalg.LinAlgError:
                 warnings.warn(f"Singular predicted covariance matrix at t={t} during smoothing. Adding jitter.")
                 inv_cov_tp1_pred = np.linalg.inv(cov_tp1_pred + np.eye(self.state_dim) * 1e-8)

            smoother_gain = cov_t_filt @ F_t.T @ inv_cov_tp1_pred

            # Update smoothed state and covariance
            state_diff = state_tp1_smooth - state_tp1_pred
            smoothed_states[t, :] = (state_t_filt + smoother_gain @ state_diff).flatten()
            smoothed_covs[t, :, :] = (
                cov_t_filt
                + smoother_gain @ (cov_tp1_smooth - cov_tp1_pred) @ smoother_gain.T
            )
            # Ensure symmetry
            smoothed_covs[t, :, :] = (smoothed_covs[t, :, :] + smoothed_covs[t, :, :].T) / 2.0


        self.smoothed_states = smoothed_states
        self.smoothed_covs = smoothed_covs
        self.is_smoothed = True

        return smoothed_states, smoothed_covs

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix F_t (requires NumPy implementation).

        This method is primarily needed for the RTS smoother implemented in the
        base class. Particle filter subclasses might implement this if they
        intend to use the base smoother, otherwise it can raise NotImplementedError.

        Args:
            state: Current state estimate (state_dim, 1) as a NumPy array.

        Returns:
            Linearized transition matrix F_t (state_dim, state_dim) as NumPy array.

        Raises:
            NotImplementedError: Must be implemented by subclasses using the smoother.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_transition_matrix for smoothing."
        )

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state and covariance using a given transition matrix (NumPy version).

        This method is primarily needed for the RTS smoother. It performs one
        step prediction using the provided linearized matrix F_t and calculates
        the process noise Q_t based on the current state estimate.

        Args:
            state: Current state estimate x_t (state_dim, 1) as NumPy array.
            cov: Current covariance P_t (state_dim, state_dim) as NumPy array.
            transition_matrix: Linearized state transition matrix F_t (NumPy array).

        Returns:
            Tuple containing:
                - Predicted state mean x_{t+1|t} (state_dim, 1) as NumPy array.
                - Predicted covariance P_{t+1|t} (state_dim, state_dim) as NumPy array.

        Raises:
            NotImplementedError: Must be implemented by subclasses using the smoother.
            AttributeError: If self.params is not set or lacks necessary attributes.
        """
        # This implementation assumes self.params is available and contains NumPy arrays
        # Subclasses needing smoothing must ensure this.
        if self.params is None:
             raise AttributeError("self.params must be set before calling _predict_with_matrix")

        K = self.K
        # Ensure state is a column vector
        state_col = state.reshape(-1, 1)

        # Use NumPy versions of parameters
        mu_np = np.array(self.params.mu).reshape(-1, 1)
        q_h_np = np.array(self.params.Q_h)

        # Predict state mean E[x_{t+1}|t] using the non-linear dynamics approximated by F_t
        # x_{t+1|t} = F_t @ x_t (This is often an approximation, the exact form depends on the model)
        # For DFSV:
        factors = state_col[:K, :]
        log_vols = state_col[K:, :]
        pred_factors_mean = transition_matrix[:K, :K] @ factors
        # h_{t+1} = mu + Phi_h @ (h_t - mu) + noise
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean])

        # Predict covariance: P_{t+1|t} = F_t @ P_t @ F_t^T + Q_t
        # Process noise Q_t = Cov([factor_noise; h_noise])
        # Q_f = diag(exp(h_t)), Q_h = params.Q_h (state-dependent)
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        # Approximate factor noise cov using current log_vols estimate h_t
        current_log_vols = state_col[K:, :].flatten()
        # Use state_col which is h_t
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols))

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t
        # Ensure symmetry
        predicted_cov = (predicted_cov + predicted_cov.T) / 2.0

        return predicted_state_mean_col, predicted_cov

    # --- Getters for Filtered/Smoothed Results ---

    def get_filtered_factors(self) -> np.ndarray:
        """Return the filtered latent factors."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Must run filter first.")
        return self.filtered_states[:, : self.K]

    def get_filtered_volatilities(self) -> np.ndarray:
        """Return the filtered log-volatilities."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Must run filter first.")
        return self.filtered_states[:, self.K :]

    def get_smoothed_factors(self) -> np.ndarray:
        """Return the smoothed latent factors."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Must run smoother first.")
        return self.smoothed_states[:, : self.K]

    def get_smoothed_volatilities(self) -> np.ndarray:
        """Return the smoothed log-volatilities."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Must run smoother first.")
        # Corrected slicing: should be K:
        return self.smoothed_states[:, self.K :]