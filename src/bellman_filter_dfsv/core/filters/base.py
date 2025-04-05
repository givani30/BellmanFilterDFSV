# src/bellman_filter_dfsv/core/filters/base.py
"""Base class for filters applied to Dynamic Factor Stochastic Volatility models.

Provides a common interface and shared utilities for various filtering algorithms
used within the DFSV framework.
"""

import warnings
from typing import Tuple, Optional, Union, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# Assuming DFSVParamsDataclass will be importable from the models directory
# We'll use absolute imports once the package structure is fully set up
# For now, let's use a placeholder or assume it's available via qf_thesis.models.dfsv
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

# Try importing tqdm for progress bars, provide a fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm iterator if tqdm is not installed."""
        warnings.warn("tqdm not installed. Progress bars will not be shown.")
        return iterable

class DFSVFilter:
    """Base class for DFSV filtering algorithms.

    Provides a common interface and shared utilities for filters like the
    Bellman Filter (covariance and information forms) and potentially others
    applied to Dynamic Factor Stochastic Volatility (DFSV) models.

    Attributes:
        N (int): Number of observed time series.
        K (int): Number of latent factors.
        state_dim (int): Dimension of the state vector (2 * K).
        is_filtered (bool): Flag indicating if the filter has been run.
        is_smoothed (bool): Flag indicating if the smoother has been run (if applicable).
        filtered_states (Optional[np.ndarray]): Filtered state estimates
            (T, state_dim) as NumPy array.
        filtered_covs (Optional[np.ndarray]): Filtered state covariances
            (T, state_dim, state_dim) as NumPy array (or None if information
            filter used).
        filtered_infos (Optional[np.ndarray]): Filtered state information matrices
            (T, state_dim, state_dim) as NumPy array (or None if covariance
            filter used). Specific to information filters.
        smoothed_states (Optional[np.ndarray]): Smoothed state estimates
            (T, state_dim) as NumPy array.
        smoothed_covs (Optional[np.ndarray]): Smoothed state covariances
            (T, state_dim, state_dim) as NumPy array.
        log_likelihood (Optional[float]): Total log-likelihood from the filter pass.
        params (Optional[DFSVParamsDataclass]): Model parameters used by the filter
            (set by subclasses).
    """

    N: int
    K: int
    state_dim: int
    is_filtered: bool
    is_smoothed: bool
    filtered_states: Optional[np.ndarray]
    filtered_covs: Optional[np.ndarray]
    filtered_infos: Optional[np.ndarray] # Added for BIF
    smoothed_states: Optional[np.ndarray]
    smoothed_covs: Optional[np.ndarray]
    log_likelihood: Optional[float]
    params: Optional[DFSVParamsDataclass]

    def __init__(self, N: int, K: int):
        """Initializes the DFSVFilter.

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
        self.filtered_infos: np.ndarray | None = None # Added for BIF
        self.smoothed_states: np.ndarray | None = None
        self.smoothed_covs: np.ndarray | None = None
        self.log_likelihood: float | None = None
        self.params: DFSVParamsDataclass | None = None # To be set by subclasses if needed

    # --- Common Helper Methods ---

    def _process_params(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass]
    ) -> DFSVParamsDataclass:
        """Converts/validates parameters to the internal DFSVParamsDataclass format.

        Ensures that the parameters used internally by the filter are consistently
        represented as a DFSVParamsDataclass containing JAX arrays with the
        correct shapes and float64 dtype. Handles conversion from dictionaries
        and validates existing dataclasses.

        Args:
            params: Model parameters, either as a dictionary or a
                DFSVParamsDataclass instance.

        Returns:
            A DFSVParamsDataclass instance containing validated JAX arrays.

        Raises:
            TypeError: If the input `params` type is not a dict or
                DFSVParamsDataclass.
            KeyError: If `params` is a dict and is missing required keys.
            ValueError: If N/K values in `params` don't match the filter's N/K,
                or if array shapes/types cannot be correctly converted/validated.
        """
        if isinstance(params, dict):
            # Convert dictionary to DFSVParamsDataclass
            N = params.get('N', self.N)
            K = params.get('K', self.K)
            if N != self.N or K != self.K:
                 raise ValueError(f"N/K in params dict ({N},{K}) don't match filter ({self.N},{self.K})")
            try:
                # Ensure all required keys are present before creating dataclass
                required_keys = ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]
                missing_keys = [key for key in required_keys if key not in params]
                if missing_keys:
                    raise KeyError(f"Missing required parameter key(s) in dict: {missing_keys}")
                # Create a temporary dict with only the required keys for the dataclass
                dataclass_params = {k: params[k] for k in required_keys}
                params_dc = DFSVParamsDataclass(N=N, K=K, **dataclass_params)
            except TypeError as e: # Catch potential issues during dataclass creation
                 raise TypeError(f"Error creating DFSVParamsDataclass from dict: {e}")

        elif isinstance(params, DFSVParamsDataclass):
            if params.N != self.N or params.K != self.K:
                 raise ValueError(f"N/K in params dataclass ({params.N},{params.K}) don't match filter ({self.N},{self.K})")
            params_dc = params # Assume it might already have JAX arrays
        else:
            raise TypeError(f"Unsupported parameter type: {type(params)}. Expected Dict or DFSVParamsDataclass.")

        # Ensure internal arrays are JAX arrays with correct dtype and shape
        default_dtype = jnp.float64
        updates = {}
        changed = False
        expected_shapes = {
            "lambda_r": (self.N, self.K),
            "Phi_f": (self.K, self.K),
            "Phi_h": (self.K, self.K),
            "mu": (self.K,), # Expect 1D
            "sigma2": (self.N,), # Expect 1D
            "Q_h": (self.K, self.K),
        }

        for field_name, expected_shape in expected_shapes.items():
            current_value = getattr(params_dc, field_name)
            is_jax_array = isinstance(current_value, jnp.ndarray)
            # Check dtype compatibility, allowing for different float/int types initially
            correct_dtype = is_jax_array and jnp.issubdtype(current_value.dtype, jnp.number)
            correct_shape = is_jax_array and current_value.shape == expected_shape

            # Convert if not JAX array, wrong dtype (target float64), or wrong shape
            if not (is_jax_array and current_value.dtype == default_dtype and correct_shape):
                try:
                    # Convert to JAX array with default dtype first
                    val = jnp.asarray(current_value, dtype=default_dtype)
                    # Reshape if necessary, ensuring compatibility
                    if field_name in ["mu", "sigma2"]:
                        val = val.flatten() # Ensure 1D
                        if val.shape != expected_shape:
                             raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {val.shape} after flatten")
                    elif val.shape != expected_shape:
                         # Allow broadcasting for scalars if target is matrix, e.g. Phi_f=0.9
                         if val.ndim == 0 and len(expected_shape) == 2 and expected_shape[0] == expected_shape[1]:
                             print(f"Warning: Broadcasting scalar '{field_name}' to {expected_shape}")
                             val = jnp.eye(expected_shape[0], dtype=default_dtype) * val
                         elif val.shape != expected_shape: # Check again after potential broadcast
                             raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {val.shape}")

                    updates[field_name] = val
                    changed = True
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Could not convert/validate parameter '{field_name}': {e}")

        if changed:
            # Create a new dataclass instance with the updated JAX arrays
            return params_dc.replace(**updates)
        else:
            # Return the original if no changes were needed
            return params_dc

    @staticmethod
    @jit
    def _solve_discrete_lyapunov_jax(
        Phi: jnp.ndarray, Q: jnp.ndarray, num_iters: int = 30
    ) -> jnp.ndarray:
        """Solves the discrete Lyapunov equation P = Phi @ P @ Phi.T + Q via iteration.

        This is a common operation for finding the stationary covariance of an AR(1)
        process, often used for initializing the log-volatility covariance.

        Args:
            Phi: The state transition matrix (K, K).
            Q: The process noise covariance matrix (K, K).
            num_iters: The number of iterations to perform. More iterations lead
                to a more accurate solution but increase computation time.

        Returns:
            The solution P (K, K), representing the stationary covariance matrix.
        """
        P = Q
        def body_fn(i, P_carry):
            return Phi @ P_carry @ Phi.T + Q
        P_final = jax.lax.fori_loop(0, num_iters, body_fn, P)
        # Ensure symmetry
        return (P_final + P_final.T) / 2.0

    @staticmethod
    def _get_transition_matrix(params: DFSVParamsDataclass, K: int) -> jnp.ndarray:
        """Constructs the state transition matrix F.

        This matrix is constant for the standard DFSV model.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            K: Number of factors (passed explicitly as it's static).

        Returns:
            The state transition matrix F (state_dim, state_dim) as a JAX array.
        """
        Phi_f = params.Phi_f
        Phi_h = params.Phi_h

        F_t = jnp.block([
            [Phi_f,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), Phi_h]
        ])
        return F_t

    # --- Abstract Methods / Methods requiring subclass implementation ---

    def initialize_state(
        self, params: DFSVParamsDataclass
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initializes the state vector and covariance/information matrix.

        Calculates the initial state mean based on unconditional moments
        (factors=0, log-vols=mu) and the initial covariance P_0 using the
        discrete Lyapunov equation for the log-volatility block.

        Note:
            Subclasses (like BIF) might override this to return the initial
            information matrix (Omega_0 = P_0^-1) instead of the covariance.

        Args:
            params: Parameters of the DFSV model (as a JAX dataclass).

        Returns:
            A tuple containing:
                - initial_state: The initial state vector (state_dim, 1) as JAX array.
                - initial_cov: The initial state covariance matrix
                  (state_dim, state_dim) as JAX array.
        """
        params = self._process_params(params) # Ensure params are processed

        # Initialize factors to zero
        initial_factors = jnp.zeros((self.K, 1), dtype=jnp.float64)

        # Initialize log-volatilities to the unconditional mean
        initial_log_vols = params.mu.reshape(-1, 1)

        # Combine into state vector [factors; log_vols]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance (identity)
        P_f = jnp.eye(self.K, dtype=jnp.float64)

        # Solve discrete Lyapunov equation for log-volatility covariance
        P_h = self._solve_discrete_lyapunov_jax(params.Phi_h, params.Q_h)

        # Construct block-diagonal initial covariance matrix
        initial_cov = jnp.block(
            [
                [P_f, jnp.zeros((self.K, self.K), dtype=jnp.float64)],
                [jnp.zeros((self.K, self.K), dtype=jnp.float64), P_h],
            ]
        )

        return initial_state, initial_cov

    def filter(
        self, params: DFSVParamsDataclass, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Runs the primary filtering algorithm.

        This method must be implemented by subclasses to perform the specific
        filtering steps (e.g., predict and update loops).

        Args:
            params: Parameters of the DFSV model (as a JAX dataclass or dict).
            y: Observed returns with shape (T, N).

        Returns:
            A tuple containing:
                - filtered_states: Filtered state estimates (T, state_dim) as NumPy array.
                - filtered_covs_or_infos: Filtered state covariances or information
                  matrices (T, state_dim, state_dim) as NumPy array.
                - total_log_likelihood: Total log-likelihood (float).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Filter method must be implemented by subclasses")


    def predict(
        self, params: DFSVParamsDataclass, state: jnp.ndarray, cov: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs the prediction step of the filter.

        This method must be implemented by subclasses to define how the state
        and its uncertainty (covariance or information) are propagated forward
        in time according to the model dynamics.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            state: Current state estimate (state_dim, 1) as JAX array.
            cov: Current state covariance or information matrix (state_dim, state_dim)
                 as JAX array.

        Returns:
            A tuple containing the predicted state and predicted covariance/information
            matrix as JAX arrays.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def update(
        self,
        params: DFSVParamsDataclass,
        predicted_state: jnp.ndarray,
        predicted_cov: jnp.ndarray, # Or predicted_info for BIF
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Performs the update step of the filter.

        This method must be implemented by subclasses to define how the predicted
        state and uncertainty are updated using the current observation, and to
        calculate the log-likelihood contribution of that observation.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            predicted_state: Predicted state (state_dim, 1) as JAX array.
            predicted_cov: Predicted state covariance or information matrix
                           (state_dim, state_dim) as JAX array.
            observation: Current observation (N,) as JAX array.

        Returns:
            A tuple containing:
                - updated_state: Updated state estimate (state_dim, 1) as JAX array.
                - updated_cov: Updated covariance or information matrix
                  (state_dim, state_dim) as JAX array.
                - log_lik_contrib: Log-likelihood contribution for this step (scalar).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Update method must be implemented by subclasses")

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the Rauch-Tung-Striebel (RTS) smoother.

        This performs a backward pass after filtering to refine state estimates
        using all available observations. It requires `filtered_states` and
        `filtered_covs` to be available as NumPy arrays.

        Note:
            This base implementation assumes a covariance-based filter was run.
            Information filter subclasses might need to override this or provide
            a way to compute smoothed covariances from smoothed information matrices.

        Returns:
            A tuple containing:
                - smoothed_states: Smoothed state estimates (T, state_dim) as NumPy array.
                - smoothed_covs: Smoothed state covariances (T, state_dim, state_dim)
                  as NumPy array.

        Raises:
            RuntimeError: If `filter()` has not been run first or if filtered
                          results (`filtered_states`, `filtered_covs`) are missing.
        """
        if not self.is_filtered or self.filtered_states is None or self.filtered_covs is None:
            raise RuntimeError(
                "Filter must be run successfully and store filtered_states and "
                "filtered_covs before smoothing."
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

            # Get the transition matrix F_t (NumPy)
            F_t = self._get_transition_matrix_np(state_t_filt) # Requires NumPy implementation

            # Predict state and covariance from t to t+1 using filtered estimate at t (NumPy)
            state_tp1_pred, cov_tp1_pred = self._predict_with_matrix(
                state_t_filt, cov_t_filt, F_t # Requires NumPy implementation
            )

            # Compute smoother gain K_t (using NumPy)
            try:
                # Use pseudo-inverse for potentially better stability
                inv_cov_tp1_pred = np.linalg.pinv(cov_tp1_pred)
            except np.linalg.LinAlgError:
                 warnings.warn(f"Singular predicted covariance matrix at t={t} during smoothing. Using pseudo-inverse.")
                 inv_cov_tp1_pred = np.linalg.pinv(cov_tp1_pred)

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

    def _get_transition_matrix_np(self, state: np.ndarray) -> np.ndarray:
        """Gets the state transition matrix F_t using NumPy.

        This method is primarily needed for the RTS smoother implemented in the
        base class, which operates on NumPy arrays. It uses the static JAX
        implementation and converts the result to NumPy.

        Args:
            state: Current state estimate (state_dim, 1) as a NumPy array.
                   (Note: state is not actually used in the standard DFSV model's
                   constant transition matrix, but kept for potential future extensions).

        Returns:
            The transition matrix F_t (state_dim, state_dim) as NumPy array.

        Raises:
            AttributeError: If self.params is not set.
        """
        if self.params is None:
             raise AttributeError("self.params must be set before calling _get_transition_matrix_np")

        # Use the static JAX implementation and convert to NumPy
        F_t_jax = self._get_transition_matrix(self.params, self.K)
        return np.asarray(F_t_jax)

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts state and covariance using a given transition matrix (NumPy).

        This method is primarily needed for the RTS smoother. It performs one
        step prediction using the provided transition matrix F_t and calculates
        the process noise Q_t based on the current state estimate.

        Args:
            state: Current state estimate x_t (state_dim, 1) as NumPy array.
            cov: Current covariance P_t (state_dim, state_dim) as NumPy array.
            transition_matrix: State transition matrix F_t (NumPy array).

        Returns:
            A tuple containing:
                - predicted_state_mean: Predicted state mean x_{t+1|t}
                  (state_dim, 1) as NumPy array.
                - predicted_cov: Predicted covariance P_{t+1|t}
                  (state_dim, state_dim) as NumPy array.

        Raises:
            AttributeError: If self.params is not set or lacks necessary attributes.
        """
        if self.params is None:
             raise AttributeError("self.params must be set before calling _predict_with_matrix")

        K = self.K
        state_col = state.reshape(-1, 1)

        # Use NumPy versions of parameters
        mu_np = np.asarray(self.params.mu).reshape(-1, 1)
        q_h_np = np.asarray(self.params.Q_h)

        # Predict state mean E[x_{t+1}|t]
        # For DFSV: x_{t+1|t} = F_t @ x_t (approximation for mean)
        # More accurately:
        factors = state_col[:K, :]
        log_vols = state_col[K:, :]
        pred_factors_mean = transition_matrix[:K, :K] @ factors
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean])


        # Predict covariance: P_{t+1|t} = F_t @ P_t @ F_t^T + Q_t
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        # Use predicted log vols for Q_f: E[exp(h_t)] is complex, use exp(E[h_t]) approx
        # For smoother predict step, we use h_{t|t} (state_col)
        current_log_vols = state_col[K:, :].flatten()
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols))

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t
        predicted_cov = (predicted_cov + predicted_cov.T) / 2.0 # Ensure symmetry

        return predicted_state_mean_col, predicted_cov

    # --- Getters for Filtered/Smoothed Results ---

    def get_filtered_factors(self) -> np.ndarray:
        """Returns the filtered latent factors f_{t|t}."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Filter must be run before getting filtered factors.")
        return self.filtered_states[:, : self.K]

    def get_filtered_volatilities(self) -> np.ndarray:
        """Returns the filtered log-volatilities h_{t|t}."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Filter must be run before getting filtered volatilities.")
        return self.filtered_states[:, self.K :]

    def get_smoothed_factors(self) -> np.ndarray:
        """Returns the smoothed latent factors f_{t|T}."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Smoother must be run before getting smoothed factors.")
        return self.smoothed_states[:, : self.K]

    def get_smoothed_volatilities(self) -> np.ndarray:
        """Returns the smoothed log-volatilities h_{t|T}."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Smoother must be run before getting smoothed volatilities.")
        return self.smoothed_states[:, self.K :]