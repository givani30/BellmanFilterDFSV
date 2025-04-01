from functools import partial
from typing import Tuple, Union, Dict, Any

import optimistix as optx
# from altair import LogicalAndPredicate # Removed unused import
import jax
import jax.numpy as jnp
import jax.scipy.linalg # Added import
# import jaxopt # Removed unused import
import numpy as np
from jax import jit

from .base import DFSVFilter # Import base class from sibling module
# Update imports to use models.dfsv instead
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
# Remove redundant jax_params import
from ._bellman_impl import build_covariance_impl, fisher_information_impl, log_posterior_impl, kl_penalty_impl
from ._bellman_optim import update_factors, update_h_bfgs # Import optimization helpers


class DFSVBellmanFilter(DFSVFilter):
    """
    Bellman Filter for Dynamic Factor Stochastic Volatility (DFSV) models.

    This class implements a Bellman filter for state estimation in DFSV models,
    using dynamic programming principles to recursively compute the optimal
    state estimates and covariances. JAX is used for automatic differentiation
    and JIT compilation to optimize the posterior distribution efficiently.

    The filter estimates factors (f) and log-volatilities (h) using a block
    coordinate descent approach.

    Attributes:
        N (int): Number of observed time series.
        K (int): Number of latent factors.
        filtered_states (jnp.ndarray | None): Filtered states [f; h] (T, state_dim) stored internally as JAX array.
        filtered_covs (jnp.ndarray | None): Filtered covariances (T, state_dim, state_dim) stored internally as JAX array.
        predicted_states (jnp.ndarray | None): Predicted states (T, state_dim, 1) stored internally as JAX array.
        predicted_covs (jnp.ndarray | None): Predicted covariances (T, state_dim, state_dim) stored internally as JAX array.
        log_likelihoods (jnp.ndarray | None): Log-likelihood contributions per step (T,) stored internally as JAX array.
        total_log_likelihood (jnp.ndarray | float | None): Total log-likelihood after running filter (JAX scalar from scan, float otherwise).
    """

    def __init__(self, N: int, K: int):
        """
        Initialize the Bellman Filter.

        Args:
            N (int): Number of observed time series.
            K (int): Number of latent factors.
        """
        super().__init__(N, K)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Initialize state storage
        self.filtered_states = None
        self.filtered_covs = None
        self.predicted_states = None
        self.predicted_covs = None
        self.log_likelihoods = None
        self.total_log_likelihood = None


        self._setup_jax_functions()


    # Helper method to standardize parameter handling
    def _process_params(self, params: Union[Dict[str, Any], DFSVParamsDataclass]) -> DFSVParamsDataclass:
        """
        Convert parameter dictionary or ensure it's a DFSVParamsDataclass.
        Ensures internal arrays are JAX arrays with correct dtype and shape.

        Args:
            params: Parameters in DFSVParamsDataclass or dictionary format.

        Returns:
            DFSVParamsDataclass: Parameters in the standardized dataclass format with JAX arrays.

        Raises:
            TypeError: If the input params type is not supported.
            KeyError: If required keys are missing in the dictionary.
            ValueError: If parameter conversion fails or N/K mismatch.
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


    def _setup_jax_functions(self):
        """
        Set up JIT-compiled JAX functions for efficiency.
        """
        # JIT the imported implementation functions
        self.build_covariance_jit = jit(build_covariance_impl)
        self.fisher_information_jit = jit(
            partial(fisher_information_impl, K=self.K, build_covariance_fn=self.build_covariance_jit)
        )
        self.log_posterior_jit = jit(
            partial(log_posterior_impl, K=self.K, build_covariance_fn=self.build_covariance_jit)
        )
        self.kl_penalty_jit = jit(kl_penalty_impl)

        # Instantiate the BFGS solver for the h update step once
        self.h_solver = optx.BFGS(rtol=1e-4, atol=1e-6)

        # Create JIT versions of core steps for scan
        # These assume inputs are correctly typed/shaped JAX arrays
        self.predict_jax = jit(self.__predict_jax)
        self.update_jax = jit(self.__update_jax)
        self.block_coordinate_update_impl_jit = jit(
            self._block_coordinate_update_impl,
            static_argnums=(6, 7),  # max_iters and h_solver are static
        )

        # Try to precompile JIT functions NOTE: Disabled for now
        try:
            self._precompile_jax_functions()
            print("JAX functions successfully precompiled")
        except Exception as e:
            print(f"Warning: JAX precompilation failed: {e}")
            print("Functions will be compiled during first filter run")

    def build_covariance(
        self, lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
    ) -> jnp.ndarray:
        """Public API for build_covariance."""
        return self.build_covariance_jit(lambda_r, exp_h, sigma2)

    def fisher_information(
        self, lambda_r: jnp.ndarray, sigma2: jnp.ndarray, alpha: jnp.ndarray
    ) -> jnp.ndarray:
        """Public API for fisher_information."""
        return self.fisher_information_jit(lambda_r, sigma2, alpha)

    def log_posterior(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> float:
        """Public API for log_posterior."""
        return self.log_posterior_jit(lambda_r, sigma2, alpha, observation)

    def _block_coordinate_update_impl(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray, # Expect 1D JAX array
        alpha: jnp.ndarray,
        pred_state: jnp.ndarray,
        I_pred: jnp.ndarray,
        observation: jnp.ndarray,
        max_iters: int, # Static arg
        h_solver: optx.AbstractMinimiser # Static arg
    ) -> jnp.ndarray:
        """
        Static implementation of block_coordinate_update using external helpers.
        Operates purely on JAX arrays.

        Args:
            lambda_r: Factor loading matrix (JAX).
            sigma2: Idiosyncratic variances (1D JAX array).
            alpha: Initial state vector [f, h] (JAX).
            pred_state: Predicted state vector (JAX).
            I_pred: Predicted precision matrix (JAX).
            observation: Observation vector (JAX).
            max_iters: Maximum number of outer block coordinate iterations (static).
            h_solver: Pre-configured Optimistix solver instance (static).

        Returns:
            Updated state vector (JAX array).
        """
        K = self.K
        alpha = alpha.flatten()
        pred_state = pred_state.flatten()
        observation = observation.flatten()

        # Split states
        factors_guess = alpha[:K]
        log_vols_guess = alpha[K:]
        factors_pred = pred_state[:K]
        log_vols_pred = pred_state[K:]

        # Partition information matrix
        I_f = I_pred[:K, :K]
        I_fh = I_pred[:K, K:]

        # Define the loop body using external functions
        def body_fn(
            i, carry: Tuple[jnp.ndarray, jnp.ndarray] # Add loop index i
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """ Single iteration of the block-coordinate update. `carry` is (f, h). """
            f_current, h_current = carry

            # Update factors using external function
            f_new = update_factors(
                log_volatility=h_current,
                lambda_r=lambda_r,
                sigma2=sigma2, # Pass 1D sigma2
                observation=observation,
                factors_pred=factors_pred,
                log_vols_pred=log_vols_pred,
                I_f=I_f,
                I_fh=I_fh,
                build_covariance_fn=self.build_covariance_jit # Pass JITted function
            )

            # Update log-vols using external function
            # update_h_bfgs now returns (h_new, success_status)
            h_new, h_update_success = update_h_bfgs(
                h_init=h_current,
                factors=f_new, # Use the newly updated factors
                lambda_r=lambda_r,
                sigma2=sigma2, # Pass 1D sigma2
                pred_state=pred_state, # Pass full predicted state
                I_pred=I_pred,         # Pass full predicted precision
                observation=observation,
                K=K,
                build_covariance_fn=self.build_covariance_jit, # Pass JITted function
                log_posterior_fn=self.log_posterior_jit,     # Pass JITted function
                h_solver=h_solver,                           # Pass solver instance
                inner_max_steps=100                          # Increased inner steps
            )
            # Note: h_update_success is currently ignored, but could be used for diagnostics
            # Consider adding jax.debug.print("h_update success: {}", h_update_success) if needed
            return (f_new, h_new)

        # Use lax.fori_loop to run max_iters times.
        init_carry = (factors_guess, log_vols_guess)
        f_final, h_final = jax.lax.fori_loop(0, max_iters, body_fn, init_carry)

        # Return updated state
        return jnp.concatenate([f_final, h_final])

    # Public API wrapper (optional, could just use JITted version directly)
    def block_coordinate_update(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        pred_alpha: jnp.ndarray,
        I_pred: jnp.ndarray,
        observation: jnp.ndarray,
        max_iters: int = 10,
    ) -> jnp.ndarray:
        """
        Public API that calls the JIT-compiled version of block_coordinate_update.
        """
        # Ensure sigma2 is 1D for the JITted function
        sigma2_1d = jnp.asarray(sigma2).flatten()
        return self.block_coordinate_update_impl_jit(
            jnp.asarray(lambda_r), sigma2_1d, jnp.asarray(alpha), jnp.asarray(pred_alpha),
            jnp.asarray(I_pred), jnp.asarray(observation), max_iters, self.h_solver
        )

    def _precompile_jax_functions(self):
        """Precompile JIT functions with dummy data"""
        return # Disabled for now
        K = self.K
        N = self.N
        state_dim = self.state_dim

        # Create dummy JAX arrays
        dummy_lambda_r = jnp.ones((N, K), dtype=jnp.float64)
        dummy_sigma2 = jnp.ones(N, dtype=jnp.float64) # 1D
        dummy_exp_h = jnp.ones(K, dtype=jnp.float64)
        dummy_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        dummy_pred_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        dummy_I_pred = jnp.eye(state_dim, dtype=jnp.float64)
        dummy_observation = jnp.zeros(N, dtype=jnp.float64)
        dummy_a_updated = jnp.ones(state_dim, dtype=jnp.float64)
        dummy_I_updated = jnp.eye(state_dim, dtype=jnp.float64)
        dummy_state = jnp.zeros((state_dim, 1), dtype=jnp.float64)
        dummy_cov = jnp.eye(state_dim, dtype=jnp.float64)

        # Create dummy params dataclass
        dummy_params = DFSVParamsDataclass(
            N=N, K=K,
            lambda_r=dummy_lambda_r,
            Phi_f=jnp.eye(K, dtype=jnp.float64) * 0.9,
            Phi_h=jnp.eye(K, dtype=jnp.float64) * 0.95,
            mu=jnp.zeros(K, dtype=jnp.float64),
            sigma2=dummy_sigma2,
            Q_h=jnp.eye(K, dtype=jnp.float64) * 0.1
        )

        print("Precompiling build_covariance...")
        _ = self.build_covariance_jit(dummy_lambda_r, dummy_exp_h, dummy_sigma2).block_until_ready()
        print("Precompiling fisher_information...")
        _ = self.fisher_information_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha).block_until_ready()
        print("Precompiling log_posterior...")
        _ = self.log_posterior_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_observation).block_until_ready()
        print("Precompiling kl_penalty...")
        _ = self.kl_penalty_jit(dummy_pred_alpha, dummy_a_updated, dummy_I_pred, dummy_I_updated).block_until_ready()
        print("Precompiling block_coordinate_update...")
        _ = self.block_coordinate_update_impl_jit(
            dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_pred_alpha,
            dummy_I_pred, dummy_observation, max_iters=2, h_solver=self.h_solver
        ).block_until_ready()
        print("Precompiling predict_jax...")
        _ = self.predict_jax(dummy_params, dummy_state, dummy_cov)[0].block_until_ready()
        print("Precompiling update_jax...")
        _ = self.update_jax(dummy_params, dummy_state, dummy_cov, dummy_observation)[0].block_until_ready()


    def kl_penalty(self, a_pred, a_updated, I_pred, I_updated):
        """Public API for KL penalty calculation."""
        return self.kl_penalty_jit(a_pred, a_updated, I_pred, I_updated)

    def initialize_state(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return JAX arrays
        """
        Initialize state and covariance as JAX arrays.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Initial state and covariance (JAX arrays).
        """
        params = self._process_params(params) # Ensure JAX arrays inside

        K = self.K
        initial_factors = jnp.zeros((K, 1), dtype=jnp.float64)
        initial_log_vols = params.mu.reshape(-1, 1) # mu is already 1D JAX array

        # Combine into state vector [f; h]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance
        P_f = jnp.eye(K, dtype=jnp.float64)

        # Solve discrete Lyapunov equation using JAX
        @jit
        def solve_discrete_lyapunov_fixed(Phi, Q, num_iters=20):
            # Ensure inputs are JAX arrays
            Phi = jnp.asarray(Phi)
            Q = jnp.asarray(Q)
            def body_fn(i, X):
                return Phi @ X @ Phi.T + Q
            # Start with Q (a reasonable initial guess)
            X_final = jax.lax.fori_loop(0, num_iters, body_fn, Q)
            return X_final

        P_h = solve_discrete_lyapunov_fixed(params.Phi_h, params.Q_h)

        # Construct block-diagonal covariance
        initial_cov = jnp.block([
            [P_f,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), P_h]
        ])

        return initial_state, initial_cov


    def predict(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Bellman prediction step (public API).

        Handles parameter processing, calls the internal JAX implementation,
        and returns results as NumPy arrays.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            state: Current state estimate (NumPy array, shape (state_dim,) or (state_dim, 1)).
            cov: Current state covariance (NumPy array, shape (state_dim, state_dim)).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted state (NumPy), predicted covariance (NumPy).
        """
        params_jax = self._process_params(params) # Ensure JAX arrays inside
        state_jax = jnp.asarray(state, dtype=jnp.float64)
        cov_jax = jnp.asarray(cov, dtype=jnp.float64)

        # Ensure state has shape (state_dim, 1) for internal consistency if needed
        if state_jax.ndim == 1:
            state_jax = state_jax.reshape(-1, 1)
        # Call the JIT-compiled JAX implementation
        predicted_state_jax, predicted_cov_jax = self.predict_jax( # Call JIT version
            params_jax, state_jax, cov_jax
        )
        print(f"DEBUG: predict_jax returned state: {predicted_state_jax}, cov: {predicted_cov_jax}") # Debug print
        # Convert results back to NumPy
        return np.asarray(predicted_state_jax), np.asarray(predicted_cov_jax)


    def update(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass],
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the Bellman update step (public API).

        Handles parameter processing, calls the internal JAX implementation,
        and returns results as NumPy arrays/float.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            predicted_state: Predicted state (NumPy array, shape (state_dim,) or (state_dim, 1)).
            predicted_cov: Predicted state covariance (NumPy array, shape (state_dim, state_dim)).
            observation: Current observation (NumPy array, shape (N,) or (N, 1)).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Updated state (NumPy), updated covariance (NumPy), log-likelihood contribution (float).
        """
        params_jax = self._process_params(params) # Ensure JAX arrays inside
        predicted_state_jax = jnp.asarray(predicted_state, dtype=jnp.float64)
        predicted_cov_jax = jnp.asarray(predicted_cov, dtype=jnp.float64)
        observation_jax = jnp.asarray(observation, dtype=jnp.float64)

        # Ensure state has shape (state_dim, 1) for internal consistency if needed
        if predicted_state_jax.ndim == 1:
            predicted_state_jax = predicted_state_jax.reshape(-1, 1)
        # Ensure observation has shape (N,) for internal consistency
        if observation_jax.ndim > 1:
             observation_jax = observation_jax.flatten()

        # Call the JIT-compiled JAX implementation
        updated_state_jax, updated_cov_jax, log_lik_jax = self.update_jax( # Call JIT version
            params_jax, predicted_state_jax, predicted_cov_jax, observation_jax
        )

        # Convert results back to NumPy/float
        return np.asarray(updated_state_jax), np.asarray(updated_cov_jax), float(log_lik_jax)

    # Internal JAX version of predict
    def __predict_jax(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        state: jnp.ndarray,          # Expect JAX array
        cov: jnp.ndarray,            # Expect JAX array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return JAX arrays
        """
        Perform the Bellman prediction step. Operates purely on JAX arrays.

        Args:
            params: Parameters (DFSVParamsDataclass with JAX arrays).
            state: Current state estimate (JAX array).
            cov: Current state covariance (JAX array).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predicted state (JAX), predicted covariance (JAX).
        """
        K = self.K
        Phi_f = params.Phi_f
        Phi_h = params.Phi_h
        mu = params.mu # Assumed 1D JAX array

        state = state.flatten()
        # cov is already JAX array

        factors = state[:K]
        log_vols = state[K:]

        predicted_factors = Phi_f @ factors
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)
        predicted_state = jnp.concatenate([predicted_factors, predicted_log_vols])

        F_t = self._get_transition_matrix(params) # Returns JAX array

        Q_t = jnp.zeros((self.state_dim, self.state_dim), dtype=jnp.float64)
        Q_f = jnp.diag(jnp.exp(predicted_log_vols)) # Use predicted h_{t|t-1}
        Q_h = params.Q_h

        Q_t = Q_t.at[:K, :K].set(Q_f)
        Q_t = Q_t.at[K:, K:].set(Q_h)

        predicted_cov = F_t @ cov @ F_t.T + Q_t
        predicted_cov = (predicted_cov + predicted_cov.T) / 2

        return predicted_state.reshape(-1, 1), predicted_cov

    # Internal JAX version of update
    def __update_jax(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        predicted_state: jnp.ndarray, # Expect JAX array
        predicted_cov: jnp.ndarray,   # Expect JAX array
        observation: jnp.ndarray,     # Expect JAX array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: # Return JAX arrays, including log_lik as JAX scalar
        """
        Perform the Bellman update step using JAX-based optimization.
        Operates purely on JAX arrays.

        Args:
            params: Parameters (DFSVParamsDataclass with JAX arrays).
            predicted_state: Predicted state (JAX array).
            predicted_cov: Predicted state covariance (JAX array).
            observation: Current observation (JAX array).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Updated state (JAX), updated covariance (JAX), log-likelihood contribution (JAX scalar).
        """
        K = self.K
        N = self.N
        lambda_r = params.lambda_r
        sigma2 = params.sigma2 # Assumed 1D JAX array

        jax_observation = observation.flatten()

        # Compute information matrix I_pred = P_pred^{-1}
        jax_predicted_cov = predicted_cov + 1e-8 * jnp.eye(self.state_dim, dtype=jnp.float64)
        try:
            chol_pred_cov = jax.scipy.linalg.cholesky(jax_predicted_cov, lower=True)
            I_pred = jax.scipy.linalg.cho_solve((chol_pred_cov, True), jnp.eye(self.state_dim, dtype=jnp.float64))
        except jnp.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            I_pred = jnp.linalg.pinv(jax_predicted_cov)
        I_pred = (I_pred + I_pred.T) / 2

        # Initial guess for optimization
        alpha_init = predicted_state.flatten()

        # Run block coordinate update (using JITted version)
        alpha_updated = self.block_coordinate_update_impl_jit(
            lambda_r, sigma2, alpha_init, predicted_state.flatten(), I_pred, jax_observation,
            max_iters=10, # Reverted to default
            h_solver=self.h_solver
        )

        # Compute updated information matrix I_updated = I_pred + Fisher
        I_fisher = self.fisher_information_jit(lambda_r, sigma2, alpha_updated)
        I_updated = I_pred + I_fisher + 1e-6 * jnp.eye(self.state_dim, dtype=jnp.float64) # Increased jitter
        I_updated = (I_updated + I_updated.T) / 2

        # Compute updated covariance P_updated = I_updated^{-1}
        try:
            chol_I_updated = jax.scipy.linalg.cholesky(I_updated, lower=True)
            updated_cov = jax.scipy.linalg.cho_solve((chol_I_updated, True), jnp.eye(self.state_dim, dtype=jnp.float64))
        except jnp.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            updated_cov = jnp.linalg.pinv(I_updated)
        updated_cov = (updated_cov + updated_cov.T) / 2

        # Calculate KL divergence penalty
        kl_div = self.kl_penalty_jit(predicted_state.flatten(), alpha_updated, I_pred, I_updated)
        log_lik_contrib = kl_div # Using KL div as proxy for likelihood contribution here

        # Return results as JAX arrays (reshaped state), keep log_lik as JAX scalar
        return alpha_updated.reshape(-1, 1), updated_cov, log_lik_contrib


    def _get_transition_matrix(
        self, params: DFSVParamsDataclass
    ) -> jnp.ndarray: # Return JAX array
        """
        Construct the state transition matrix F (which is constant in this model).

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).

        Returns:
            jnp.ndarray: State transition matrix F (JAX array).
        """
        K = self.K
        Phi_f = params.Phi_f
        Phi_h = params.Phi_h

        F_t = jnp.block([
            [Phi_f,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), Phi_h]
        ])
        return F_t

    # --- Methods to retrieve filtered results (converting internal JAX arrays to NumPy) ---
    def get_filtered_states(self) -> np.ndarray | None:
        """Returns the filtered states [f; h] (T, state_dim) as a NumPy array."""
        states_jax = getattr(self, 'filtered_states', None)
        return np.asarray(states_jax) if states_jax is not None else None

    def get_filtered_factors(self) -> np.ndarray | None:
        """Returns the filtered factors f (T, K) as a NumPy array."""
        states_np = self.get_filtered_states() # Gets NumPy array
        if states_np is not None:
            # Slicing on the NumPy array
            return states_np[:, :self.K]
        return None

    def get_filtered_volatilities(self) -> np.ndarray | None:
        """Returns the filtered log-volatilities h (T, K) as a NumPy array."""
        states_np = self.get_filtered_states() # Gets NumPy array
        if states_np is not None:
            # Slicing on the NumPy array
            return states_np[:, self.K:]
        return None

    def get_filtered_covariances(self) -> np.ndarray | None:
        """Returns the filtered state covariances (T, state_dim, state_dim) as a NumPy array."""
        covs_jax = getattr(self, 'filtered_covs', None)
        return np.asarray(covs_jax) if covs_jax is not None else None

    # Optional: Add getters for predicted states/covs if needed, following the same pattern
    def get_predicted_states(self) -> np.ndarray | None:
        """Returns the predicted states (T, state_dim, 1) as a NumPy array."""
        states_jax = getattr(self, 'predicted_states', None)
        return np.asarray(states_jax) if states_jax is not None else None

    def get_predicted_covariances(self) -> np.ndarray | None:
        """Returns the predicted state covariances (T, state_dim, state_dim) as a NumPy array."""
        covs_jax = getattr(self, 'predicted_covs', None)
        return np.asarray(covs_jax) if covs_jax is not None else None

    def get_log_likelihoods(self) -> np.ndarray | None:
        """Returns the log-likelihood contributions per step (T,) as a NumPy array."""
        lls_jax = getattr(self, 'log_likelihoods', None)
        return np.asarray(lls_jax) if lls_jax is not None else None

    # --- Filtering Methods ---

    def filter(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter using a standard Python loop.
        NOTE: This method uses NumPy loops and conversions, may not be JAX-optimal
              and might be incompatible with JIT if called within a JIT context.
              Prefer filter_scan for JAX integration.

        Args:
            params: Parameters of the DFSV model.
            observations: Observed returns with shape (T, N).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Filtered states, filtered covariances, total log-likelihood.
        """
        params_jax = self._process_params(params) # Ensure correct format with JAX arrays

        T = observations.shape[0]
        state_dim = self.state_dim

        # Initialize storage (JAX arrays)
        filtered_states_jax = jnp.zeros((T, state_dim), dtype=jnp.float64)
        filtered_covs_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        predicted_states_jax = jnp.zeros((T, state_dim, 1), dtype=jnp.float64)
        predicted_covs_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        log_likelihoods_jax = jnp.zeros(T, dtype=jnp.float64)

        # Initialization (t=0) - Use JAX results from initialize_state
        initial_state_jax, initial_cov_jax = self.initialize_state(params_jax)
        predicted_states_jax = predicted_states_jax.at[0].set(initial_state_jax)
        predicted_covs_jax = predicted_covs_jax.at[0].set(initial_cov_jax)

        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        # Filtering loop (using JAX arrays for storage and computation)
        for t in tqdm(range(T), desc="Bellman Filtering (JAX Loop)"):
            # Get inputs for update step (already JAX arrays)
            pred_state_t_jax = predicted_states_jax[t]
            pred_cov_t_jax = predicted_covs_jax[t]
            # Convert observation for this step to JAX array
            obs_t_jax = jnp.array(observations[t])

            # Update step (operates in JAX, returns JAX)
            updated_state_t_jax, updated_cov_t_jax, log_lik_t_jax = self.update_jax(
                params_jax, pred_state_t_jax, pred_cov_t_jax, obs_t_jax
            )

            # Store results using JAX functional updates
            filtered_states_jax = filtered_states_jax.at[t].set(updated_state_t_jax.flatten()) # Flatten state before storing
            filtered_covs_jax = filtered_covs_jax.at[t].set(updated_cov_t_jax)
            log_likelihoods_jax = log_likelihoods_jax.at[t].set(log_lik_t_jax) # Store JAX scalar

            # Predict step for next iteration (if not the last step)
            if t < T - 1:
                # Predict step (operates in JAX, returns JAX)
                pred_state_next_jax, pred_cov_next_jax = self.predict_jax(
                    params_jax, updated_state_t_jax, updated_cov_t_jax # Use JAX arrays from update
                )
                # Store predicted results using JAX functional updates
                predicted_states_jax = predicted_states_jax.at[t + 1].set(pred_state_next_jax)
                predicted_covs_jax = predicted_covs_jax.at[t + 1].set(pred_cov_next_jax)

        # Store results internally as JAX arrays
        self.filtered_states = filtered_states_jax
        self.filtered_covs = filtered_covs_jax
        self.predicted_states = predicted_states_jax
        self.predicted_covs = predicted_covs_jax
        self.log_likelihoods = log_likelihoods_jax
        self.total_log_likelihood = float(jnp.sum(log_likelihoods_jax)) # Sum JAX array, convert to float

        # Return NumPy arrays by calling getter methods (which will handle conversion)
        return self.get_filtered_states(), self.get_filtered_covariances(), self.total_log_likelihood


    def filter_scan(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, jnp.ndarray]: # Return JAX scalar for loglik
        """
        Run the Bellman filter using jax.lax.scan for potential speedup.
        Ensures operations within scan use JAX arrays.

        Args:
            params: Parameters of the DFSV model.
            observations: Observed returns with shape (T, N).

        Returns:
            Tuple[np.ndarray, np.ndarray, jnp.ndarray]: Filtered states (NumPy), filtered covariances (NumPy), total log-likelihood (JAX scalar).
        """
        params_jax = self._process_params(params) # Ensure correct format (contains JAX arrays)
        T = observations.shape[0]

        # Initialization (get initial state/cov as JAX arrays)
        initial_state_jax, initial_cov_jax = self.initialize_state(params_jax)
        # Ensure carry types are JAX compatible (float for sum)
        initial_carry = (initial_state_jax, initial_cov_jax, jnp.array(0.0, dtype=jnp.float64)) # state, cov, log_lik_sum

        # JAX observations
        jax_observations = jnp.array(observations)

        # Define the step function for lax.scan (operates purely on JAX types)
        # Make params_jax static for the step function if JITting scan
        # @partial(jit, static_argnums=(0,)) # Optional: JIT the step function itself
        def filter_step(carry, obs_t):
            state_t_minus_1_jax, cov_t_minus_1_jax, log_lik_sum_t_minus_1 = carry

            # Predict step (predict state t based on t-1) -> returns JAX arrays
            pred_state_t_jax, pred_cov_t_jax = self.predict_jax(params_jax, state_t_minus_1_jax, cov_t_minus_1_jax)

            # Update step (update state t using observation t) -> returns JAX arrays
            updated_state_t_jax, updated_cov_t_jax, log_lik_t_jax = self.update_jax(
                params_jax, pred_state_t_jax, pred_cov_t_jax, obs_t
            )

            # Prepare carry for next step (using JAX arrays)
            next_carry = (updated_state_t_jax, updated_cov_t_jax, log_lik_sum_t_minus_1 + log_lik_t_jax)

            # What we store for this time step t (JAX arrays)
            scan_output = (pred_state_t_jax, pred_cov_t_jax, updated_state_t_jax, updated_cov_t_jax, log_lik_t_jax)
            return next_carry, scan_output

        # Run the scan
        final_carry, scan_results = jax.lax.scan(filter_step, initial_carry, jax_observations)

        # Unpack results (still JAX arrays)
        predicted_states_scan, predicted_covs_scan, filtered_states_scan, filtered_covs_scan, log_likelihoods_scan = scan_results

        # Assign final results directly as JAX arrays
        self.predicted_states = predicted_states_scan # Shape (T, state_dim, 1)
        self.predicted_covs = predicted_covs_scan   # Shape (T, state_dim, state_dim)
        # Reshape filtered states from (T, state_dim, 1) to (T, state_dim) before storing
        self.filtered_states = filtered_states_scan.reshape(T, self.state_dim)
        self.filtered_covs = filtered_covs_scan     # Shape (T, state_dim, state_dim)
        self.log_likelihoods = log_likelihoods_scan # Shape (T,)
        # Store final log-likelihood sum as JAX scalar
        self.total_log_likelihood = final_carry[2]


        # Return NumPy arrays for states/covs, JAX scalar for loglik
        return self.get_filtered_states(), self.get_filtered_covariances(), self.total_log_likelihood


    def log_likelihood_of_params(
        self, params_dict: Dict[str, Any], observations: np.ndarray
    ) -> jnp.ndarray: # Return JAX scalar
        """
        Calculate the log-likelihood for given parameters and observations.
        Uses the filter_scan method internally.

        Args:
            params_dict: Dictionary of parameters.
            observations: Observed returns (T, N).

        Returns:
            jnp.ndarray: Total log-likelihood (JAX scalar).
        """
        try:
            # Convert dict to dataclass, ensuring N and K are correct
            params_jax = self._process_params(params_dict)
            _, _, total_log_lik = self.filter_scan(params_jax, observations)
            # Handle potential NaN/Inf values from filtering (using JAX functions)
            # Note: np.isnan/isinf would cause ConcretizationTypeError under jax.grad
            # Return the JAX scalar directly
            return jnp.where(jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik), -jnp.inf, total_log_lik)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e: # Use np.linalg.LinAlgError
            # Handle errors during parameter processing or filtering
            print(f"Warning: Error calculating likelihood: {e}")
            # Return JAX representation of -inf
            return jnp.array(-jnp.inf, dtype=jnp.float64)


    def _log_likelihood_of_params_impl(
        self, params: DFSVParamsDataclass, observations: jnp.ndarray
    ) -> float:
        """
        JAX-compatible implementation for log-likelihood calculation using scan.
        Designed to be JIT-compiled. Assumes inputs are JAX arrays.

        Args:
            params: DFSVParamsDataclass instance with JAX arrays.
            observations: Observed returns (T, N) as JAX array.

        Returns:
            float: Total log-likelihood (as JAX scalar).
        """
        T = observations.shape[0]

        # Initialization (use JAX arrays)
        initial_state_jax, initial_cov_jax = self.initialize_state(params)
        # Ensure carry types are JAX compatible
        initial_carry = (initial_state_jax, initial_cov_jax, jnp.array(0.0, dtype=jnp.float64)) # state, cov, log_lik_sum

        # Define the step function for lax.scan (operates purely on JAX types)
        def filter_step(carry, obs_t):
            state_t_minus_1_jax, cov_t_minus_1_jax, log_lik_sum_t_minus_1 = carry
            # Predict step -> returns JAX arrays
            pred_state_t_jax, pred_cov_t_jax = self.predict_jax(params, state_t_minus_1_jax, cov_t_minus_1_jax)
            # Update step -> returns JAX arrays
            updated_state_t_jax, updated_cov_t_jax, log_lik_t_jax = self.update_jax(
                params, pred_state_t_jax, pred_cov_t_jax, obs_t
            )
            # Prepare carry for next step
            next_carry = (updated_state_t_jax, updated_cov_t_jax, log_lik_sum_t_minus_1 + log_lik_t_jax)
            # We only need the carry for the final likelihood
            return next_carry, None # Don't store intermediate results

        # Run the scan
        final_carry, _ = jax.lax.scan(filter_step, initial_carry, observations)

        total_log_lik = final_carry[2] # JAX scalar

        # Replace NaN/Inf with -inf for optimization stability
        return jnp.where(jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik), -jnp.inf, total_log_lik)


    # Method to get a JIT-compiled version of the log-likelihood function
    def jit_log_likelihood_of_params(self):
        """
        Returns a JIT-compiled function to compute the log-likelihood.
        The returned function takes params (DFSVParamsDataclass) and observations (JAX array).
        """
        # JIT the implementation method directly
        # Caller must ensure inputs are correct JAX types (DFSVParamsDataclass, jnp.ndarray)
        return jit(self._log_likelihood_of_params_impl)
