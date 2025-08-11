# src/bellman_filter_dfsv/filters/base.py # Corrected path comment
"""Base class for filters applied to Dynamic Factor Stochastic Volatility models.

Provides a common interface and shared utilities for various filtering algorithms
used within the DFSV framework.
"""

import warnings
from typing import Tuple, Optional, Union, Dict, Any, Callable # Added Callable

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import jax_dataclasses as jdc # Added import

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

@jdc.pytree_dataclass(frozen=True)
class SmootherResults:
    """Holds the results of the RTS smoother pass.

    Attributes:
        smoothed_states: Smoothed state estimates α_{t|T} (T, state_dim).
        smoothed_covs: Smoothed state covariances P_{t|T} (T, state_dim, state_dim).
        smoothed_lag1_covs: Smoothed lag-1 covariances P_{t+1,t|T}
                           (T, state_dim, state_dim). Note: The array at index `t`
                           stores P_{t+1,t|T}, computed at step `t` of the backward
                           pass (t=T-1 down to 0). So the array contains
                           [P_{1,0|T}, P_{2,1|T}, ..., P_{T,T-1|T}].
    """
    smoothed_states: jnp.ndarray
    smoothed_covs: jnp.ndarray
    smoothed_lag1_covs: jnp.ndarray # Stores P_{t+1,t|T} at index t


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
        smoothed_lag1_covs (Optional[np.ndarray]): Smoothed lag-1 state covariances
            P_{t+1,t|T} (T, state_dim, state_dim) as NumPy array. Index t holds P_{t+1,t|T}.
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
    smoothed_lag1_covs: Optional[np.ndarray] # Added for RTS lag-1 covariance
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
        self.smoothed_lag1_covs: np.ndarray | None = None # Initialize
        self.log_likelihood: float | None = None
        self.params: DFSVParamsDataclass | None = None # To be set by subclasses if needed

    # --- Common Helper Methods ---

    def _process_params(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass],default_dtype=jnp.float64
    ) -> DFSVParamsDataclass:
        """Converts/validates parameters to the internal DFSVParamsDataclass format.

        Ensures that the parameters used internally by the filter are consistently
        represented as a DFSVParamsDataclass containing JAX arrays with the
        correct shapes and float64 dtype. Handles conversion from dictionaries
        and validates existing dataclasses.

        Args:
            params: Model parameters, either as a dictionary or a
                DFSVParamsDataclass instance.
            default_dtype: The target JAX dtype for numerical parameters.

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
    @eqx.filter_jit
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

    @staticmethod
    @eqx.filter_jit
    def _predict_jax(
        params: DFSVParamsDataclass, state: jnp.ndarray, cov: jnp.ndarray, K: int, state_dim: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predicts state and covariance using JAX operations.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            state: Current state estimate x_t (state_dim,) as JAX array.
            cov: Current covariance P_t (state_dim, state_dim) as JAX array.
            K: Number of factors.
            state_dim: Dimension of the state vector.

        Returns:
            A tuple containing:
                - predicted_state_mean: Predicted state mean x_{t+1|t} (state_dim,) as JAX array.
                - predicted_cov: Predicted covariance P_{t+1|t} (state_dim, state_dim) as JAX array.
        """
        # Get transition matrix F (constant for standard DFSV)
        F_t = DFSVFilter._get_transition_matrix(params, K) # Use static method

        # Predict state mean E[x_{t+1}|t]
        # For DFSV: x_{t+1|t} = F_t @ x_t (approximation for mean)
        # More accurately, handle the constant mu for log-vols:
        mu = params.mu
        factors = state[:K]
        log_vols = state[K:]
        pred_factors_mean = F_t[:K, :K] @ factors
        # E[h_{t+1}] = mu + Phi_h @ (h_t - mu)
        pred_log_vols_mean = mu + F_t[K:, K:] @ (log_vols - mu)
        predicted_state_mean = jnp.concatenate([pred_factors_mean, pred_log_vols_mean])

        # Predict covariance: P_{t+1|t} = F_t @ P_t @ F_t^T + Q_t
        # Construct Q_t based on current state h_t
        Q_t = jnp.zeros((state_dim, state_dim), dtype=cov.dtype)
        # Q_f = diag(exp(h_t))
        Q_f_diag = jnp.exp(log_vols)
        Q_t = Q_t.at[:K, :K].set(jnp.diag(Q_f_diag))
        # Q_h is constant
        Q_t = Q_t.at[K:, K:].set(params.Q_h)

        predicted_cov = F_t @ cov @ F_t.T + Q_t
        # Ensure symmetry
        predicted_cov = (predicted_cov + predicted_cov.T) / 2.0

        return predicted_state_mean, predicted_cov

    # --- Abstract Methods / Methods requiring subclass implementation ---

    def initialize_state(
        self, params: DFSVParamsDataclass
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initializes the state vector and covariance/information matrix.

        Calculates the initial state mean based on unconditional moments
        (factors=0, log-vols=mu) and the initial covariance P_0 using the
        discrete Lyapunov equation for the log-volatility block.

        Note:
            Subclasses (like information filters) might override this to return the
            initial information matrix (Omega_0 = P_0^-1) instead of the covariance
            matrix P_0 in the second element of the returned tuple.

        Args:
            params: Parameters of the DFSV model (as a JAX dataclass).

        Returns:
            A tuple containing:
                - initial_state: The initial state vector (state_dim,) as JAX array (flattened).
                - initial_cov_or_info: The initial state covariance matrix P_0 or
                  information matrix Omega_0 (state_dim, state_dim) as JAX array.
        """
        params = self._process_params(params) # Ensure params are processed

        # Initialize factors to zero
        initial_factors = jnp.zeros((self.K, 1), dtype=jnp.float64)

        # Initialize log-volatilities to the unconditional mean
        initial_log_vols = params.mu.reshape(-1, 1)

        # Combine into state vector [factors; log_vols]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance (identity)
        # Consider making this configurable or based on data variance? For now, identity.
        P_f = jnp.eye(self.K, dtype=jnp.float64) * 1e6 # Large initial variance for factors

        # Solve discrete Lyapunov equation for log-volatility covariance
        P_h = self._solve_discrete_lyapunov_jax(params.Phi_h, params.Q_h)

        # Construct block-diagonal initial covariance matrix
        initial_cov = jnp.block(
            [
                [P_f, jnp.zeros((self.K, self.K), dtype=jnp.float64)],
                [jnp.zeros((self.K, self.K), dtype=jnp.float64), P_h],
            ]
        )

        # Return column vector for state (not flattened)
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
        self, params: DFSVParamsDataclass, state: jnp.ndarray, cov_or_info: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs the prediction step of the filter.

        This method must be implemented by subclasses to define how the state
        and its uncertainty (covariance or information matrix) are propagated forward
        in time according to the model dynamics.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            state: Current state estimate (state_dim,) as JAX array.
            cov_or_info: Current state covariance or information matrix
                         (state_dim, state_dim) as JAX array.

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
        predicted_cov_or_info: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Performs the update step of the filter.

        This method must be implemented by subclasses to define how the predicted
        state and uncertainty (covariance or information matrix) are updated using
        the current observation, and to calculate the log-likelihood contribution
        of that observation.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            predicted_state: Predicted state (state_dim,) as JAX array.
            predicted_cov_or_info: Predicted state covariance or information matrix
                                   (state_dim, state_dim) as JAX array.
            observation: Current observation (N,) as JAX array.

        Returns:
            A tuple containing:
                - updated_state: Updated state estimate (state_dim,) as JAX array.
                - updated_cov_or_info: Updated covariance or information matrix
                                       (state_dim, state_dim) as JAX array.
                - log_lik_contrib: Log-likelihood contribution for this step (scalar).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Update method must be implemented by subclasses")

    def log_likelihood_wrt_params(
        self, params: DFSVParamsDataclass, observations: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the log-likelihood of the observations given the parameters.

        This method must be implemented by subclasses to compute the total
        log-likelihood of the entire observation sequence `y` given a specific
        set of model parameters `params`. This is often the objective function
        for parameter estimation.

        Args:
            params: The model parameters (DFSVParamsDataclass) for which to
                    calculate the likelihood.
            observations: The sequence of observations (T, N) as a JAX array.

        Returns:
            The total log-likelihood value (scalar) as a JAX array.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "log_likelihood_wrt_params method must be implemented by subclasses"
        )

    def jit_log_likelihood_wrt_params(self) -> Callable:
        """Returns a JIT-compiled version of the log-likelihood calculation.

        This method should return a callable function (often a JIT-compiled version
        of `log_likelihood_wrt_params` or a related internal function) that takes
        parameters and observations as input and efficiently computes the
        log-likelihood. This is crucial for optimization routines.

        Returns:
            A callable function, typically JIT-compiled, for log-likelihood calculation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "jit_log_likelihood_wrt_params method must be implemented by subclasses"
        )

    def smooth(self, params: DFSVParamsDataclass) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the Rauch-Tung-Striebel (RTS) smoother using JAX.

        This performs a backward pass after filtering to refine state estimates
        using all available observations. It requires `filtered_states` and
        `filtered_covs` to be available (typically as NumPy arrays from a previous
        filter run). It computes smoothed states, covariances, and lag-1 covariances.

        Args:
            params: Model parameters (DFSVParamsDataclass).

        Returns:
            A tuple containing:
                - smoothed_states: Smoothed state estimates α_{t|T} (T, state_dim) as NumPy array.
                - smoothed_covs: Smoothed state covariances P_{t|T} (T, state_dim, state_dim) as NumPy array.
                - smoothed_lag1_covs: Smoothed lag-1 covariances P_{t+1,t|T}
                  (T, state_dim, state_dim) as NumPy array. Index t holds P_{t+1,t|T}.

        Raises:
            RuntimeError: If `filter()` has not been run first or if filtered/predicted
                          results (`filtered_states`, `filtered_covs`, `predicted_states`,
                          `predicted_covs`) are missing.
            ValueError: If parameter processing fails.
        """
        # Check for required filter outputs (states are essential)
        # Covariances will be fetched via getters to support info filters
        required_attrs = ['filtered_states', 'predicted_states']
        missing_attrs = [attr for attr in required_attrs if getattr(self, attr, None) is None]
        if not self.is_filtered or missing_attrs:
             raise RuntimeError(
                 f"Filter must be run successfully and store required state attributes "
                 f"({', '.join(required_attrs)}) before smoothing. Missing: {missing_attrs}"
             )
        # Also check if covariance getter methods exist
        if not hasattr(self, 'get_filtered_covariances') or not hasattr(self, 'get_predicted_covariances'):
            raise AttributeError("Filter instance must have 'get_filtered_covariances' and 'get_predicted_covariances' methods for smoothing.")


        T = self.filtered_states.shape[0]
        if T <= 0: # Handle empty case
             warnings.warn("Cannot smooth with T=0 observations.")
             empty_states = np.empty((0, self.state_dim))
             empty_covs = np.empty((0, self.state_dim, self.state_dim))
             self.smoothed_states = empty_states
             self.smoothed_covs = empty_covs
             self.smoothed_lag1_covs = empty_covs
             self.is_smoothed = True
             return empty_states, empty_covs, empty_covs
        elif T == 1: # Cannot smooth if T == 1, return filtered
             self.smoothed_states = self.filtered_states.copy()
             self.smoothed_covs = self.filtered_covs.copy()
             # Lag-1 cov doesn't exist for T=1, return zeros or similar shape
             self.smoothed_lag1_covs = np.zeros_like(self.filtered_covs)
             self.is_smoothed = True
             return self.smoothed_states, self.smoothed_covs, self.smoothed_lag1_covs

        # Process parameters and ensure they are JAX arrays
        try:
            params_jax = self._process_params(params)
        except (TypeError, KeyError, ValueError) as e:
            raise ValueError(f"Parameter processing failed during smoothing: {e}")

        # Convert filtered results to JAX arrays
        # Ensure states are (T, state_dim) and covs are (T, state_dim, state_dim)
        filtered_states_jax = jnp.asarray(self.filtered_states)
        # Fetch covariances using getter method to support info filters
        filtered_covs_np = self.get_filtered_covariances()
        if filtered_covs_np is None:
             raise RuntimeError("get_filtered_covariances() returned None during smoothing.")
        filtered_covs_jax = jnp.asarray(filtered_covs_np)


        # Get constant transition matrix
        F_t = self._get_transition_matrix(params_jax, self.K)

        # Define the JAX smoother step function
        # Note: Using instance methods like _predict_jax inside jit can cause issues
        # if the instance itself changes. Pass static info like K, state_dim explicitly.
        K_static = self.K
        state_dim_static = self.state_dim

        @eqx.filter_jit
        def _rts_smoother_step(carry, xs_t):
            # carry: (state_{t+1|T}, cov_{t+1|T}) from previous step (or last filtered for init)
            # xs_t: (state_{t|t}, cov_{t|t}, state_{t+1|t}, cov_{t+1|t}) from filtered/predicted results
            state_tp1_smooth, cov_tp1_smooth = carry
            state_t_filt, cov_t_filt, state_tp1_pred, cov_tp1_pred = xs_t # Unpack predicted values
            #flatten states
            state_t_filt = state_t_filt.flatten()
            state_tp1_pred = state_tp1_pred.flatten()
            # Predicted state and covariance (state_{t+1|t}, cov_{t+1|t}) are now directly available from xs_t

            # Compute smoother gain J_t = P_{t|t} F_t' P_{t+1|t}^{-1}
            # Use pseudo-inverse for numerical stability
            inv_cov_tp1_pred = jnp.linalg.pinv(cov_tp1_pred)
            smoother_gain = cov_t_filt @ F_t.T @ inv_cov_tp1_pred # J_t

            # Update smoothed state: α_{t|T} = α_{t|t} + J_t (α_{t+1|T} - α_{t+1|t})
            state_diff = state_tp1_smooth - state_tp1_pred
            state_t_smooth = state_t_filt + smoother_gain @ state_diff

            # Debug assertion: state_t_smooth should be 1D vector
            assert state_t_smooth.ndim == 1 and state_t_smooth.shape[0] == state_dim_static, \
                f"RTS smoother: state_t_smooth has unexpected shape {state_t_smooth.shape}"

            # Update smoothed covariance: P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'
            cov_diff = cov_tp1_smooth - cov_tp1_pred
            cov_t_smooth = cov_t_filt + smoother_gain @ cov_diff @ smoother_gain.T
            # Ensure symmetry
            cov_t_smooth = (cov_t_smooth + cov_t_smooth.T) / 2.0

            # Compute smoothed lag-1 covariance: P_{t+1,t|T} = P_{t+1|T} J_t'
            # This is computed at step t (backward pass) for the pair (t+1, t)
            cov_tp1_t_smooth = cov_tp1_smooth @ smoother_gain.T
            # P_{t+1,t|T} is not necessarily symmetric.

            # New carry for next step (t-1): (state_{t|T}, cov_{t|T})
            carry_new = (state_t_smooth.flatten(), cov_t_smooth)
            # Result to store for this step t: (state_{t|T}, cov_{t|T}, P_{t+1,t|T})
            # Use the SmootherResults pytree for structured output from scan
            result_t = SmootherResults(
                smoothed_states=state_t_smooth,
                smoothed_covs=cov_t_smooth,
                smoothed_lag1_covs=cov_tp1_t_smooth
            )

            return carry_new, result_t

        # Prepare inputs for scan
        # Initial carry is the smoothed state/cov at time T (which is the filtered state/cov at T)
        init_carry = (filtered_states_jax[T - 1, :].flatten(), filtered_covs_jax[T - 1, :, :])
        # xs are the filtered states/covs from T-2 down to 0
        # Ensure correct shapes for scan: state (state_dim,), cov (state_dim, state_dim)
        # Convert predicted results to JAX arrays
        predicted_states_jax = jnp.asarray(self.predicted_states)
        # Fetch predicted covariances using getter method
        predicted_covs_np = self.get_predicted_covariances()
        if predicted_covs_np is None:
             raise RuntimeError("get_predicted_covariances() returned None during smoothing.")
        predicted_covs_jax = jnp.asarray(predicted_covs_np)


        # xs now includes filtered and predicted values for time t (filtered) and t+1 (predicted)
        # The scan runs from t=T-2 down to 0.
        # xs_t corresponds to time t. We need filtered_{t|t} and predicted_{t+1|t}.
        xs = (
            filtered_states_jax[:-1, :],    # state_{t|t} for t=0..T-2
            filtered_covs_jax[:-1, :, :],   # cov_{t|t} for t=0..T-2
            predicted_states_jax[1:, :],    # state_{t+1|t} for t=0..T-2 (index 1 is t=0 -> state_{1|0})
            predicted_covs_jax[1:, :, :]    # cov_{t+1|t} for t=0..T-2 (index 1 is t=0 -> cov_{1|0})
        )


        # Run the backward scan
        _, results_scan = jax.lax.scan(_rts_smoother_step, init_carry, xs, reverse=True)

        # Unpack results from the scan (Pytree of SmootherResults)
        smoothed_states_scan = results_scan.smoothed_states
        smoothed_covs_scan = results_scan.smoothed_covs
        smoothed_lag1_covs_scan = results_scan.smoothed_lag1_covs

        # Combine results: Smoothed values include the last time step (T-1) from init_carry
        # and the scanned results for times T-2 down to 0.
        final_smoothed_states = jnp.vstack([smoothed_states_scan, init_carry[0][jnp.newaxis, :]])
        final_smoothed_covs = jnp.vstack([smoothed_covs_scan, init_carry[1][jnp.newaxis, :, :]])

        # Lag-1 covs P_{t+1,t|T} are computed for t = T-2 down to 0.
        # The result `smoothed_lag1_covs_scan` has shape (T-1, state_dim, state_dim).
        # Index `i` corresponds to time `t = T-2-i`. It stores P_{t+1,t|T}.
        # We want the final array to have shape (T, state_dim, state_dim), where index t
        # stores P_{t+1,t|T}. The scan gives us P_{T-1,T-2|T}, ..., P_{1,0|T}.
        # We need to pad this array at the end (for t=T-1, where P_{T,T-1|T} is needed).
        # Let's compute P_{T,T-1|T} = P_{T|T} J_{T-1}' using the last filtered values.

        # Recompute J_{T-1} needed for P_{T,T-1|T}
        state_T_minus_1_filt = filtered_states_jax[T - 2, :] # Not needed anymore
        cov_T_minus_1_filt = filtered_covs_jax[T - 2, :, :]
        # Use the pre-computed predicted covariance P_{T|T-1}
        cov_T_pred = predicted_covs_jax[T-1, :, :] # This is P_{T|T-1}

        # Use stable Cholesky-based inversion instead of pinv
        jitter_smooth = 1e-6 # Use a small jitter
        cov_T_pred_jittered = cov_T_pred + jitter_smooth * jnp.eye(state_dim_static, dtype=jnp.float64)
        chol_T_pred = jax.scipy.linalg.cholesky(cov_T_pred_jittered, lower=True)
        inv_cov_T_pred = jax.scipy.linalg.cho_solve((chol_T_pred, True), jnp.eye(state_dim_static, dtype=jnp.float64))
        inv_cov_T_pred = (inv_cov_T_pred + inv_cov_T_pred.T) / 2.0 # Ensure symmetry

        smoother_gain_T_minus_1 = cov_T_minus_1_filt @ F_t.T @ inv_cov_T_pred # J_{T-1}

        # P_{T,T-1|T} = P_{T|T} J_{T-1}'
        cov_T_Tminus1_smooth = init_carry[1] @ smoother_gain_T_minus_1.T # P_{T|T} J_{T-1}'

        # Combine lag-1 covs: [P_{1,0|T}, ..., P_{T-1,T-2|T}] from scan + P_{T,T-1|T}
        final_smoothed_lag1_covs = jnp.vstack([smoothed_lag1_covs_scan, cov_T_Tminus1_smooth[jnp.newaxis, :, :]])


        # Store results as NumPy arrays
        smoothed_states_np = np.asarray(final_smoothed_states)
        smoothed_covs_np = np.asarray(final_smoothed_covs)
        smoothed_lag1_covs_np = np.asarray(final_smoothed_lag1_covs)

        self.smoothed_states = smoothed_states_np
        self.smoothed_covs = smoothed_covs_np
        self.smoothed_lag1_covs = smoothed_lag1_covs_np # Store the new result
        self.is_smoothed = True

        return smoothed_states_np, smoothed_covs_np, smoothed_lag1_covs_np

    # --- Deprecated NumPy Helpers (kept for potential reference/compatibility) ---

    def _get_transition_matrix_np(self, params: DFSVParamsDataclass, state: np.ndarray) -> np.ndarray:
        """Gets the state transition matrix F_t using NumPy. (Potentially deprecated)"""
        warnings.warn("_get_transition_matrix_np might be deprecated if smoother is fully JAX.", DeprecationWarning)
        if params is None:
             raise AttributeError("params must be provided to _get_transition_matrix_np")
        F_t_jax = self._get_transition_matrix(params, self.K)
        return np.asarray(F_t_jax)

    def _predict_with_matrix(
        self, params: DFSVParamsDataclass, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts state and covariance using a given transition matrix (NumPy). (Potentially deprecated)"""
        warnings.warn("_predict_with_matrix might be deprecated if smoother is fully JAX.", DeprecationWarning)
        if params is None:
             raise AttributeError("params must be provided to _predict_with_matrix")

        K = self.K
        state_col = state.reshape(-1, 1)
        mu_np = np.asarray(params.mu).reshape(-1, 1)
        q_h_np = np.asarray(params.Q_h)

        factors = state_col[:K, :]
        log_vols = state_col[K:, :]
        pred_factors_mean = transition_matrix[:K, :K] @ factors
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean])

        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        current_log_vols = state_col[K:, :].flatten()
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols))

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t
        predicted_cov = (predicted_cov + predicted_cov.T) / 2.0

        return predicted_state_mean_col, predicted_cov

    # --- Getters for Filtered/Smoothed Results ---

    def get_filtered_states(self) -> np.ndarray:
        """Returns the filtered state vectors alpha_{t|t} with shape (T, state_dim)."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Filter must be run before getting filtered states.")
        # Always return shape (T, state_dim)
        if self.filtered_states.ndim == 3:
            return self.filtered_states[:, :, 0]
        else:
            return self.filtered_states

    def get_predicted_states(self) -> np.ndarray:
        """Returns the predicted state vectors alpha_{t|t-1} with shape (T, state_dim)."""
        if not self.is_filtered or not hasattr(self, 'predicted_states') or self.predicted_states is None:
            raise RuntimeError("Filter must be run before getting predicted states.")
        # Always return shape (T, state_dim)
        if self.predicted_states.ndim == 3:
            return self.predicted_states[:, :, 0]
        else:
            return self.predicted_states

    def get_filtered_factors(self) -> np.ndarray:
        """Returns the filtered latent factors f_{t|t}."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Filter must be run before getting filtered factors.")
        # Handle both (T, state_dim) and (T, state_dim, 1) shapes
        if self.filtered_states.ndim == 3:
            return self.filtered_states[:, : self.K, 0]
        else:
            return self.filtered_states[:, : self.K]

    def get_filtered_volatilities(self) -> np.ndarray:
        """Returns the filtered log-volatilities h_{t|t}."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Filter must be run before getting filtered volatilities.")
        # Handle both (T, state_dim) and (T, state_dim, 1) shapes
        if self.filtered_states.ndim == 3:
            return self.filtered_states[:, self.K:, 0]
        else:
            return self.filtered_states[:, self.K:]

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

    def get_smoothed_lag1_covariances(self) -> np.ndarray:
        """Returns the smoothed lag-1 covariances P_{t+1,t|T}.

        Note that the array returned has shape (T, state_dim, state_dim),
        where the element at index `t` corresponds to P_{t+1,t|T}.
        """
        if not self.is_smoothed or self.smoothed_lag1_covs is None:
            raise RuntimeError("Smoother must be run before getting smoothed lag-1 covariances.")
        return self.smoothed_lag1_covs