from functools import partial
from typing import Any, Dict, Tuple, Union

import equinox as eqx

# from altair import LogicalAndPredicate # Removed unused import
import jax
import jax.numpy as jnp
import jax.scipy.linalg  # Added import

# import jaxopt # Removed unused import
import numpy as np
import optimistix as optx
from jax import jit

# Update imports to use models.dfsv instead
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

# Remove redundant jax_params import
from ._bellman_impl import (  # Removed kl_penalty_impl import
    bif_likelihood_penalty_impl,
    build_covariance_impl,
    log_posterior_impl,
    observed_fim_impl,
)
from ._bellman_optim import (
    _block_coordinate_update_impl,  # Import optimization helpers including the shared block update
)
from .base import DFSVFilter  # Import base class from sibling module


class DFSVBellmanFilter(DFSVFilter):
    """Bellman Filter for Dynamic Factor Stochastic Volatility (DFSV) models.

    This class implements a Bellman filter for state estimation in DFSV models,
    using dynamic programming principles. It recursively computes optimal state
    estimates and covariances. JAX is used for automatic differentiation and
    JIT compilation to optimize the posterior distribution efficiently.

    The filter estimates factors (f) and log-volatilities (h) using a block
    coordinate descent approach within the update step. This version uses the
    BIF pseudo-likelihood calculation (Lange et al., 2024) for potentially
    improved stability and parameter estimation.

    Attributes:
        N (int): Number of observed time series.
        K (int): Number of latent factors.
        filtered_states (Optional[jnp.ndarray]): Filtered states [f; h]
            (T, state_dim) stored internally as JAX array.
        filtered_covs (Optional[jnp.ndarray]): Filtered covariances
            (T, state_dim, state_dim) stored internally as JAX array.
        predicted_states (Optional[jnp.ndarray]): Predicted states
            (T, state_dim, 1) stored internally as JAX array.
        predicted_covs (Optional[jnp.ndarray]): Predicted covariances
            (T, state_dim, state_dim) stored internally as JAX array.
        log_likelihoods (Optional[jnp.ndarray]): Log-likelihood contributions per
            step (T,) stored internally as JAX array.
        total_log_likelihood (Optional[Union[jnp.ndarray, float]]): Total
            log-likelihood after running filter (JAX scalar from scan, float
            otherwise).
        h_solver (optx.AbstractMinimiser): Optimistix solver for 'h' update.
        build_covariance_jit (Callable): JIT-compiled covariance builder.
        fisher_information_jit (Callable): JIT-compiled Fisher info calculator.
        log_posterior_jit (Callable): JIT-compiled log posterior calculator.
        kl_penalty_jit (Callable): JIT-compiled BIF penalty calculator.
        predict_jax (Callable): JIT-compiled prediction step.
        update_jax (Callable): JIT-compiled update step.
        block_coordinate_update_impl_jit (Callable): JIT-compiled block update.
    """

    def __init__(self, N: int, K: int):
        """Initializes the DFSVBellmanFilter.

        Args:
            N: Number of observed time series.
            K: Number of latent factors.
        """
        super().__init__(N, K)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Initialize state storage (as None initially)
        self.filtered_states = None
        self.filtered_covs = None
        self.predicted_states = None
        self.predicted_covs = None
        self.log_likelihoods = None
        self.total_log_likelihood = None


        self._setup_jax_functions()


    # _process_params is now inherited from DFSVFilter base class
    # def _process_params(self, params: Union[Dict[str, Any], DFSVParamsDataclass]) -> DFSVParamsDataclass:
    #     """
    #     Convert parameter dictionary or ensure it's a DFSVParamsDataclass.
    #     Ensures internal arrays are JAX arrays with correct dtype and shape.
    #
    #     Args:
    #         params: Parameters in DFSVParamsDataclass or dictionary format.
    #
    #     Returns:
    #         DFSVParamsDataclass: Parameters in the standardized dataclass format with JAX arrays.
    #
    #     Raises:
    #         TypeError: If the input params type is not supported.
    #         KeyError: If required keys are missing in the dictionary.
    #         ValueError: If parameter conversion fails or N/K mismatch.
    #     """
    #     if isinstance(params, dict):
    #         # Convert dictionary to DFSVParamsDataclass
    #         N = params.get('N', self.N)
    #         K = params.get('K', self.K)
    #         if N != self.N or K != self.K:
    #              raise ValueError(f"N/K in params dict ({N},{K}) don't match filter ({self.N},{self.K})")
    #         try:
    #             # Ensure all required keys are present before creating dataclass
    #             required_keys = ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]
    #             missing_keys = [key for key in required_keys if key not in params]
    #             if missing_keys:
    #                 raise KeyError(f"Missing required parameter key(s) in dict: {missing_keys}")
    #             # Create a temporary dict with only the required keys for the dataclass
    #             dataclass_params = {k: params[k] for k in required_keys}
    #             params_dc = DFSVParamsDataclass(N=N, K=K, **dataclass_params)
    #         except TypeError as e: # Catch potential issues during dataclass creation
    #              raise TypeError(f"Error creating DFSVParamsDataclass from dict: {e}")
    #
    #     elif isinstance(params, DFSVParamsDataclass):
    #         if params.N != self.N or params.K != self.K:
    #              raise ValueError(f"N/K in params dataclass ({params.N},{params.K}) don't match filter ({self.N},{self.K})")
    #         params_dc = params # Assume it might already have JAX arrays
    #     else:
    #         raise TypeError(f"Unsupported parameter type: {type(params)}. Expected Dict or DFSVParamsDataclass.")
    #
    #     # Ensure internal arrays are JAX arrays with correct dtype and shape
    #     default_dtype = jnp.float64
    #     updates = {}
    #     changed = False
    #     expected_shapes = {
    #         "lambda_r": (self.N, self.K),
    #         "Phi_f": (self.K, self.K),
    #         "Phi_h": (self.K, self.K),
    #         "mu": (self.K,), # Expect 1D
    #         "sigma2": (self.N,), # Expect 1D
    #         "Q_h": (self.K, self.K),
    #     }
    #
    #     for field_name, expected_shape in expected_shapes.items():
    #         current_value = getattr(params_dc, field_name)
    #         is_jax_array = isinstance(current_value, jnp.ndarray)
    #         # Check dtype compatibility, allowing for different float/int types initially
    #         correct_dtype = is_jax_array and jnp.issubdtype(current_value.dtype, jnp.number)
    #         correct_shape = is_jax_array and current_value.shape == expected_shape
    #
    #         # Convert if not JAX array, wrong dtype (target float64), or wrong shape
    #         if not (is_jax_array and current_value.dtype == default_dtype and correct_shape):
    #             try:
    #                 # Convert to JAX array with default dtype first
    #                 val = jnp.asarray(current_value, dtype=default_dtype)
    #                 # Reshape if necessary, ensuring compatibility
    #                 if field_name in ["mu", "sigma2"]:
    #                     val = val.flatten() # Ensure 1D
    #                     if val.shape != expected_shape:
    #                          raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {val.shape} after flatten")
    #                 elif val.shape != expected_shape:
    #                      # Allow broadcasting for scalars if target is matrix, e.g. Phi_f=0.9
    #                      if val.ndim == 0 and len(expected_shape) == 2 and expected_shape[0] == expected_shape[1]:
    #                          print(f"Warning: Broadcasting scalar '{field_name}' to {expected_shape}")
    #                          val = jnp.eye(expected_shape[0], dtype=default_dtype) * val
    #                      elif val.shape != expected_shape: # Check again after potential broadcast
    #                          raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {val.shape}")
    #
    #                 updates[field_name] = val
    #                 changed = True
    #             except (TypeError, ValueError) as e:
    #                 raise ValueError(f"Could not convert/validate parameter '{field_name}': {e}")
    #
    #     if changed:
    #         # Create a new dataclass instance with the updated JAX arrays
    #         return params_dc.replace(**updates)
    #     else:
    #         # Return the original if no changes were needed
    #         return params_dc

    def _setup_jax_functions(self):
        """Sets up and JIT-compiles the core JAX functions used by the filter."""
        # JIT the imported implementation functions
        self.build_covariance_jit = jit(build_covariance_impl)
        self.fisher_information_jit = jit(
            partial(observed_fim_impl, K=self.K)
        )
        self.log_posterior_jit = jit(
            partial(log_posterior_impl, K=self.K, build_covariance_fn=self.build_covariance_jit)
        )
        # Use BIF penalty function (imported from _bellman_impl)
        self.kl_penalty_jit = jit(bif_likelihood_penalty_impl)

        # Instantiate the BFGS solver for the h update step once
        self.h_solver = optx.BFGS(rtol=1e-4, atol=1e-6)

        # Create JIT versions of core steps for scan
        # These assume inputs are correctly typed/shaped JAX arrays
        self.predict_jax = jit(self.__predict_jax)
        self.update_jax = jit(self.__update_jax)

        # JIT the imported _block_coordinate_update_impl
        # Pass static K and JITted dependencies via partial application
        self.block_coordinate_update_impl_jit = eqx.filter_jit(
            partial(
                _block_coordinate_update_impl, # Use imported function
                K=self.K, # Pass K explicitly
                h_solver=self.h_solver, # Pass solver instance dynamically
                build_covariance_fn=self.build_covariance_jit, # Pass JITted dependency
                log_posterior_fn=self.log_posterior_jit # Pass JITted dependency
            ))

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
        """Builds the observation covariance matrix Sigma_t.

        Args:
            lambda_r: Factor loading matrix (N, K).
            exp_h: Exponentiated log-volatilities exp(h_t) (K,).
            sigma2: Idiosyncratic variances (N,).

        Returns:
            The observation covariance matrix Sigma_t (N, N).
        """
        return self.build_covariance_jit(lambda_r, exp_h, sigma2)

    def fisher_information(
        self, lambda_r: jnp.ndarray, sigma2: jnp.ndarray, alpha: jnp.ndarray, observation: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the observed Fisher Information matrix J_observed.

        Args:
            lambda_r: Factor loading matrix (N, K).
            sigma2: Idiosyncratic variances (N,).
            alpha: State vector [f, h] (state_dim,).
            observation: Observation vector y_t (N,).

        Returns:
            The observed Fisher Information matrix (state_dim, state_dim).
        """
        # Pass observation to the JITted function
        return self.fisher_information_jit(lambda_r, sigma2, alpha, observation)

    def log_posterior(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> float:
        """Calculates the log posterior log p(y_t | alpha_t).

        Args:
            lambda_r: Factor loading matrix (N, K).
            sigma2: Idiosyncratic variances (N,).
            alpha: State vector [f, h] (state_dim,).
            observation: Observation vector y_t (N,).

        Returns:
            The log posterior value (scalar float).
        """
        return self.log_posterior_jit(lambda_r, sigma2, alpha, observation)

    # _block_coordinate_update_impl is now imported from _bellman_optim
    # def _block_coordinate_update_impl(
    #     self,
    #     lambda_r: jnp.ndarray,
    #     sigma2: jnp.ndarray, # Expect 1D JAX array
    #     alpha: jnp.ndarray,
    #     pred_state: jnp.ndarray,
    #     I_pred: jnp.ndarray,
    #     observation: jnp.ndarray,
    #     max_iters: int, # Static arg
    #     h_solver: optx.AbstractMinimiser # Static arg
    # ) -> jnp.ndarray:
    #     """
    #     Static implementation of block_coordinate_update using external helpers.
    #     Operates purely on JAX arrays.
    #
    #     Args:
    #         lambda_r: Factor loading matrix (JAX).
    #         sigma2: Idiosyncratic variances (1D JAX array).
    #         alpha: Initial state vector [f, h] (JAX).
    #         pred_state: Predicted state vector (JAX).
    #         I_pred: Predicted precision matrix (JAX).
    #         observation: Observation vector (JAX).
    #         max_iters: Maximum number of outer block coordinate iterations (static).
    #         h_solver: Pre-configured Optimistix solver instance (static).
    #
    #     Returns:
    #         Updated state vector (JAX array).
    #     """
    #     K = self.K
    #     alpha = alpha.flatten()
    #     pred_state = pred_state.flatten()
    #     observation = observation.flatten()
    #
    #     # Split states
    #     factors_guess = alpha[:K]
    #     log_vols_guess = alpha[K:]
    #     factors_pred = pred_state[:K]
    #     log_vols_pred = pred_state[K:]
    #
    #     # Partition information matrix
    #     I_f = I_pred[:K, :K]
    #     I_fh = I_pred[:K, K:]
    #
    #     # Define the loop body using external functions
    #     def body_fn(
    #         i, carry: Tuple[jnp.ndarray, jnp.ndarray] # Add loop index i
    #     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #         """ Single iteration of the block-coordinate update. `carry` is (f, h). """
    #         f_current, h_current = carry
    #
    #         # Update factors using external function
    #         f_new = update_factors(
    #             log_volatility=h_current,
    #             lambda_r=lambda_r,
    #             sigma2=sigma2, # Pass 1D sigma2
    #             observation=observation,
    #             factors_pred=factors_pred,
    #             log_vols_pred=log_vols_pred,
    #             I_f=I_f,
    #             I_fh=I_fh,
    #             build_covariance_fn=self.build_covariance_jit # Pass JITted function
    #         )
    #
    #         # Update log-vols using external function
    #         # update_h_bfgs now returns (h_new, success_status)
    #         h_new, h_update_success = update_h_bfgs(
    #             h_init=h_current,
    #             factors=f_new, # Use the newly updated factors
    #             lambda_r=lambda_r,
    #             sigma2=sigma2, # Pass 1D sigma2
    #             pred_state=pred_state, # Pass full predicted state
    #             I_pred=I_pred,         # Pass full predicted precision
    #             observation=observation,
    #             K=K,
    #             build_covariance_fn=self.build_covariance_jit, # Pass JITted function
    #             log_posterior_fn=self.log_posterior_jit,     # Pass JITted function
    #             h_solver=h_solver,                           # Pass solver instance
    #             inner_max_steps=100                          # Increased inner steps
    #         )
    #         # Note: h_update_success is currently ignored, but could be used for diagnostics
    #         # Consider adding jax.debug.print("h_update success: {}", h_update_success) if needed
    #         return (f_new, h_new)
    #
    #     # Use lax.fori_loop to run max_iters times.
    #     init_carry = (factors_guess, log_vols_guess)
    #     f_final, h_final = jax.lax.fori_loop(0, max_iters, body_fn, init_carry)
    #
    #     # Return updated state
    #     return jnp.concatenate([f_final, h_final])

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
        """Performs block coordinate update to find the posterior mode.

        Calls the JIT-compiled implementation `_block_coordinate_update_impl_jit`.

        Args:
            lambda_r: Factor loading matrix (N, K).
            sigma2: Idiosyncratic variances (N,).
            alpha: Initial guess for the state vector [f, h] (state_dim,).
            pred_alpha: Predicted state vector [f_pred, h_pred] (state_dim,).
            I_pred: Predicted precision matrix Omega_{t|t-1} (state_dim, state_dim).
            observation: Observation vector y_t (N,).
            max_iters: Maximum number of block coordinate descent iterations.

        Returns:
            The optimized state vector alpha_{t|t} (state_dim,).
        """
        # Ensure sigma2 is 1D for the JITted function
        sigma2_1d = jnp.asarray(sigma2).flatten()
        # Call the JITted function (h_solver is bound via partial in setup)
        return self.block_coordinate_update_impl_jit(
            (lambda_r, sigma2_1d, alpha, pred_alpha, I_pred, observation), # Args tuple for partial
            max_iters # Static argument
        )

    def _precompile_jax_functions(self):
        """Precompiles JIT functions with dummy data (currently disabled)."""
        return # Disabled for now
        # K = self.K
        # N = self.N
        # state_dim = self.state_dim
        #
        # # Create dummy JAX arrays
        # dummy_lambda_r = jnp.ones((N, K), dtype=jnp.float64)
        # dummy_sigma2 = jnp.ones(N, dtype=jnp.float64) # 1D
        # dummy_exp_h = jnp.ones(K, dtype=jnp.float64)
        # dummy_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        # dummy_pred_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        # dummy_I_pred = jnp.eye(state_dim, dtype=jnp.float64)
        # dummy_observation = jnp.zeros(N, dtype=jnp.float64)
        # dummy_a_updated = jnp.ones(state_dim, dtype=jnp.float64)
        # dummy_I_updated = jnp.eye(state_dim, dtype=jnp.float64)
        # dummy_state = jnp.zeros((state_dim, 1), dtype=jnp.float64)
        # dummy_cov = jnp.eye(state_dim, dtype=jnp.float64)
        #
        # # Create dummy params dataclass
        # dummy_params = DFSVParamsDataclass(
        #     N=N, K=K,
        #     lambda_r=dummy_lambda_r,
        #     Phi_f=jnp.eye(K, dtype=jnp.float64) * 0.9,
        #     Phi_h=jnp.eye(K, dtype=jnp.float64) * 0.95,
        #     mu=jnp.zeros(K, dtype=jnp.float64),
        #     sigma2=dummy_sigma2,
        #     Q_h=jnp.eye(K, dtype=jnp.float64) * 0.1
        # )
        #
        # print("Precompiling build_covariance...")
        # _ = self.build_covariance_jit(dummy_lambda_r, dummy_exp_h, dummy_sigma2).block_until_ready()
        # print("Precompiling fisher_information...")
        # _ = self.fisher_information_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_observation).block_until_ready() # Added observation
        # print("Precompiling log_posterior...")
        # _ = self.log_posterior_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_observation).block_until_ready()
        # print("Precompiling kl_penalty...")
        # _ = self.kl_penalty_jit(dummy_pred_alpha, dummy_a_updated, dummy_I_pred, dummy_I_updated).block_until_ready()
        # print("Precompiling block_coordinate_update...")
        # # Need to adjust call signature for the partial application
        # _ = self.block_coordinate_update_impl_jit(
        #     (dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_pred_alpha, dummy_I_pred, dummy_observation),
        #     max_iters=2
        # ).block_until_ready()
        # print("Precompiling predict_jax...")
        # _ = self.predict_jax(dummy_params, dummy_state, dummy_cov)[0].block_until_ready()
        # print("Precompiling update_jax...")
        # _ = self.update_jax(dummy_params, dummy_state, dummy_cov, dummy_observation)[0].block_until_ready()


    def kl_penalty(self, a_pred, a_updated, I_pred, I_updated):
        """Calculates the BIF pseudo-likelihood penalty term.

        Args:
            a_pred: Predicted state mean alpha_{t|t-1} (state_dim,).
            a_updated: Updated state mean alpha_{t|t} (state_dim,).
            I_pred: Predicted state precision Omega_{t|t-1} (state_dim, state_dim).
            I_updated: Updated state precision Omega_{t|t} (state_dim, state_dim).

        Returns:
            The BIF penalty value (scalar float).
        """
        # Note: Internally uses bif_likelihood_penalty_impl via kl_penalty_jit alias
        return self.kl_penalty_jit(a_pred, a_updated, I_pred, I_updated)

    def initialize_state(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return JAX arrays
        """Initializes the state vector and covariance matrix.

        Uses the base class implementation.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).

        Returns:
            A tuple containing:
                - initial_state: Initial state vector (state_dim, 1) as JAX array.
                - initial_cov: Initial covariance matrix (state_dim, state_dim)
                  as JAX array.
        """
        # Use the base class method directly
        return super().initialize_state(params)


    def predict(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the Bellman prediction step (public API).

        Handles parameter processing, calls the internal JAX implementation,
        and returns results as NumPy arrays.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            state: Current state estimate (NumPy array, shape (state_dim,) or
                (state_dim, 1)).
            cov: Current state covariance (NumPy array, shape (state_dim, state_dim)).

        Returns:
            A tuple containing:
                - predicted_state: Predicted state (NumPy array, shape (state_dim, 1)).
                - predicted_cov: Predicted covariance (NumPy array, shape
                  (state_dim, state_dim)).
        """
        params_jax = self._process_params(params) # Ensure JAX arrays inside
        state_jax = jnp.asarray(state, dtype=jnp.float64).flatten()
        cov_jax = jnp.asarray(cov, dtype=jnp.float64)

        # Ensure state is 1D vector
        state_jax = state_jax.flatten()
        # Call the JIT-compiled JAX implementation
        predicted_state_jax, predicted_cov_jax = self.predict_jax( # Call JIT version
            params_jax, state_jax, cov_jax
        )
        # Convert results back to NumPy
        return np.asarray(predicted_state_jax), np.asarray(predicted_cov_jax)


    def update(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass],
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs the Bellman update step (public API).

        Handles parameter processing, calls the internal JAX implementation
        (which now uses the BIF pseudo-likelihood), and returns results as
        NumPy arrays/float.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            predicted_state: Predicted state (NumPy array, shape (state_dim,) or
                (state_dim, 1)).
            predicted_cov: Predicted state covariance (NumPy array, shape
                (state_dim, state_dim)).
            observation: Current observation (NumPy array, shape (N,) or (N, 1)).

        Returns:
            A tuple containing:
                - updated_state: Updated state (NumPy array, shape (state_dim, 1)).
                - updated_cov: Updated covariance (NumPy array, shape
                  (state_dim, state_dim)).
                - log_lik_contrib: Log-likelihood contribution for this step (float).
        """
        params_jax = self._process_params(params) # Ensure JAX arrays inside
        predicted_state_jax = jnp.asarray(predicted_state, dtype=jnp.float64)
        predicted_cov_jax = jnp.asarray(predicted_cov, dtype=jnp.float64)
        observation_jax = jnp.asarray(observation, dtype=jnp.float64)

        # Ensure state is 1D vector
        predicted_state_jax = predicted_state_jax.flatten()
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
        state: jnp.ndarray,          # Expect JAX array (state_dim, 1)
        cov: jnp.ndarray,            # Expect JAX array (state_dim, state_dim)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return JAX arrays
        """Performs the Bellman prediction step (internal JAX implementation).

        Args:
            params: Parameters (DFSVParamsDataclass with JAX arrays).
            state: Current state estimate alpha_{t-1|t-1} (JAX array, shape (state_dim, 1)).
            cov: Current state covariance P_{t-1|t-1} (JAX array, shape (state_dim, state_dim)).

        Returns:
            A tuple containing:
                - predicted_state: Predicted state alpha_{t|t-1} (JAX array, shape (state_dim, 1)).
                - predicted_cov: Predicted covariance P_{t|t-1} (JAX array, shape (state_dim, state_dim)).
        """
        K = self.K
        mu = params.mu # Assumed 1D JAX array

        state_flat = state.flatten()
        factors = state_flat[:K]
        log_vols = state_flat[K:]

        # Predict state mean using model dynamics
        predicted_factors = params.Phi_f @ factors
        predicted_log_vols = mu + params.Phi_h @ (log_vols - mu)
        predicted_state = jnp.concatenate([predicted_factors, predicted_log_vols])

        # Get transition matrix F_t
        F_t = self._get_transition_matrix(params, K) # Use base class static method

        # Calculate process noise covariance Q_t (state-dependent for factors)
        Q_t = jnp.zeros((self.state_dim, self.state_dim), dtype=jnp.float64)
        # Use predicted log vols for Q_f variance: E[exp(h_t)] is complex, use exp(E[h_t]) approx
        Q_f = jnp.diag(jnp.exp(predicted_log_vols))
        Q_h = params.Q_h
        Q_t = Q_t.at[:K, :K].set(Q_f)
        Q_t = Q_t.at[K:, K:].set(Q_h)

        # Predict covariance: P_{t|t-1} = F_t @ P_{t-1|t-1} @ F_t^T + Q_t
        predicted_cov = F_t @ cov @ F_t.T + Q_t
        predicted_cov = (predicted_cov + predicted_cov.T) / 2 # Ensure symmetry

        return predicted_state.flatten(), predicted_cov

    # Internal JAX version of update
    def __update_jax(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        predicted_state: jnp.ndarray, # Expect JAX array (state_dim, 1)
        predicted_cov: jnp.ndarray,   # Expect JAX array (state_dim, state_dim)
        observation: jnp.ndarray,     # Expect JAX array (N,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: # Return JAX arrays
        """Performs the Bellman update step using BIF pseudo-likelihood (internal JAX).

        Calculates the updated state (posterior mode) and covariance using the
        predicted state/covariance and the current observation. Computes the
        log-likelihood contribution based on the BIF formula (Lange et al., 2024).

        Args:
            params: Parameters (DFSVParamsDataclass with JAX arrays).
            predicted_state: Predicted state alpha_{t|t-1} (JAX array, shape (state_dim, 1)).
            predicted_cov: Predicted covariance P_{t|t-1} (JAX array, shape (state_dim, state_dim)).
            observation: Current observation y_t (JAX array, shape (N,)).

        Returns:
            A tuple containing:
                - updated_state: Updated state alpha_{t|t} (JAX array, shape (state_dim, 1)).
                - updated_cov: Updated covariance P_{t|t} (JAX array, shape (state_dim, state_dim)).
                - log_lik_contrib: Log-likelihood contribution log p(y_t|F_{t-1}) (JAX scalar).
        """
        K = self.K
        N = self.N
        lambda_r = params.lambda_r
        sigma2 = params.sigma2 # Assumed 1D JAX array

        jax_observation = observation.flatten()
        state_dim = self.state_dim # Get state dimension
        jitter_pred = 1e-8 # Jitter for predicted covariance inversion
        jitter_post = 1e-6 # Jitter for posterior information inversion

        # --- Calculate Predicted Information Matrix (Omega_pred = P_pred^-1) ---
        jax_predicted_cov_jittered = predicted_cov + jitter_pred * jnp.eye(state_dim, dtype=jnp.float64)
        # Use Cholesky decomposition to invert the jittered predicted covariance
        chol_pred_cov = jax.scipy.linalg.cholesky(jax_predicted_cov_jittered, lower=True)
        Omega_pred = jax.scipy.linalg.cho_solve((chol_pred_cov, True), jnp.eye(state_dim, dtype=jnp.float64))
        Omega_pred = (Omega_pred + Omega_pred.T) / 2 # Ensure symmetry

        # --- State Update (Posterior Mode Calculation) ---
        # Initial guess for optimization is the predicted state
        alpha_init_guess = predicted_state.flatten()

        # Run block coordinate update (using JITted version)
        # This optimizes alpha_t to find the posterior mode alpha_{t|t}
        # Note: h_solver, K, build_covariance_fn, log_posterior_fn are bound via partial in _setup_jax_functions
        alpha_updated = self.block_coordinate_update_impl_jit(
            lambda_r,                  # Pass args individually
            sigma2,
            alpha_init_guess,
            predicted_state.flatten(),
            Omega_pred,                # Pass predicted info matrix
            jax_observation,
            max_iters=10               # Pass static arg by name or position
        )

        # --- Calculate Updated Information Matrix (Omega_post = Omega_pred + J_observed) ---
        # Calculate Observed Fisher Information J_observed = -Hessian(log p(y_t|alpha_t)) at alpha_updated
        J_observed = self.fisher_information_jit(lambda_r, sigma2, alpha_updated, jax_observation)

        # --- Regularize J_observed to ensure PSD ---
        evals_j, evecs_j = jnp.linalg.eigh(J_observed)
        min_eigenvalue = 1e-8 # Small positive floor for J_observed eigenvalues
        evals_j_clipped = jnp.maximum(evals_j, min_eigenvalue)
        J_observed_psd = evecs_j @ jnp.diag(evals_j_clipped) @ evecs_j.T
        J_observed_psd = (J_observed_psd + J_observed_psd.T) / 2 # Ensure symmetry
        # --- End Regularization ---

        # Compute updated information matrix Omega_{t|t} = Omega_{t|t-1} + J_observed_psd
        Omega_post = Omega_pred + J_observed_psd + jitter_post * jnp.eye(state_dim, dtype=jnp.float64) # Add jitter
        Omega_post = (Omega_post + Omega_post.T) / 2 # Ensure symmetry

        # --- Calculate Updated Covariance (P_post = Omega_post^-1) ---
        # Use Cholesky decomposition to invert the posterior information matrix
        chol_Omega_post = jax.scipy.linalg.cholesky(Omega_post, lower=True) # Omega_post already has jitter
        updated_cov = jax.scipy.linalg.cho_solve((chol_Omega_post, True), jnp.eye(state_dim, dtype=jnp.float64))
        updated_cov = (updated_cov + updated_cov.T) / 2 # Ensure symmetry

        # --- Calculate Log-Likelihood Contribution using BIF formula (Lange Eq. 40) ---
        # log p(y_t|F_{t-1}) ≈ log p(y_t|alpha_{t|t}) - penalty

        # Calculate fit term: log p(y_t | alpha_{t|t})
        log_lik_fit = self.log_posterior_jit(lambda_r, sigma2, alpha_updated, jax_observation)

        # Calculate BIF penalty term (uses bif_likelihood_penalty_impl via kl_penalty_jit alias)
        # penalty = 0.5 * (log_det(Omega_post) - log_det(Omega_pred) + diff^T @ Omega_pred @ diff)
        bif_penalty = self.kl_penalty_jit(
            a_pred=predicted_state.flatten(),
            a_updated=alpha_updated,
            Omega_pred=Omega_pred,
            Omega_post=Omega_post
        )

        # Combine fit and penalty
        log_lik_contrib = log_lik_fit - bif_penalty

        # Return results as JAX arrays (reshaped state), keep log_lik as JAX scalar
        return alpha_updated.flatten(), updated_cov, log_lik_contrib


    # _get_transition_matrix is now inherited from DFSVFilter base class
    # def _get_transition_matrix(self, params: DFSVParamsDataclass) -> jnp.ndarray:
    #     """
    #     Construct the state transition matrix F (which is constant in this model).
    #
    #     Args:
    #         params: Model parameters (DFSVParamsDataclass with JAX arrays).
    #
    #     Returns:
    #         jnp.ndarray: State transition matrix F (JAX array).
    #     """
    #     # Implementation moved to base class static method
    #     return DFSVFilter._get_transition_matrix(params, self.K)

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
        """Runs the Bellman filter using a standard Python loop.

        NOTE: This method uses NumPy loops and conversions, may not be JAX-optimal
              and might be incompatible with JIT if called within a JIT context.
              Prefer filter_scan for JAX integration.

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed returns with shape (T, N).

        Returns:
            A tuple containing:
                - filtered_states: Filtered states (T, state_dim) as NumPy array.
                - filtered_covs: Filtered covariances (T, state_dim, state_dim)
                  as NumPy array.
                - total_log_likelihood: Total log-likelihood (float).
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
        """Runs the Bellman filter using `jax.lax.scan` for potential speedup.

        Ensures operations within the scan loop use JAX arrays for compatibility
        with JIT compilation.

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed returns with shape (T, N).

        Returns:
            A tuple containing:
                - filtered_states: Filtered states (T, state_dim) as NumPy array.
                - filtered_covs: Filtered covariances (T, state_dim, state_dim)
                  as NumPy array.
                - total_log_likelihood: Total log-likelihood (JAX scalar).
        """
        params_jax = self._process_params(params) # Ensure correct format (contains JAX arrays)
        T = observations.shape[0]

        # Initialization (get initial state/cov as JAX arrays)
        initial_state_jax, initial_cov_jax = self.initialize_state(params_jax)
        # Flatten initial state to ensure consistent carry shape
        initial_carry = (initial_state_jax.flatten(), initial_cov_jax, jnp.array(0.0, dtype=jnp.float64)) # state, cov, log_lik_sum

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
            next_carry = (updated_state_t_jax.flatten(), updated_cov_t_jax, log_lik_sum_t_minus_1 + log_lik_t_jax)

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


    # --- Smoothing Method ---
    def smooth(self, params: DFSVParamsDataclass) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Rauch-Tung-Striebel (RTS) smoothing.

        Requires the filter to have been run first. Uses the base class
        implementation which relies on stored filtered and predicted results.

        Returns:
            A tuple containing:
                - smoothed_states: Smoothed states (T, state_dim) as NumPy array.
                - smoothed_covs: Smoothed covariances (T, state_dim, state_dim)
                  as NumPy array.

        Raises:
            RuntimeError: If the filter has not been run yet (results are None).
        """
        # Check if filter results (JAX arrays) are available
        if getattr(self, 'filtered_states', None) is None or getattr(self, 'filtered_covs', None) is None:
            raise RuntimeError(
                "Filter must be run successfully (e.g., using filter_scan) "
                "before smoothing."
            )

        # Convert JAX arrays to NumPy arrays just before calling base smoother
        # Overwrite the attributes temporarily for the base class call
        self.filtered_states = np.asarray(self.filtered_states)
        self.filtered_covs = np.asarray(self.filtered_covs)
        self.is_filtered = True # Ensure base class knows filter was run

        # Call the base class implementation which expects NumPy arrays and now params
        smoothed_states_np, smoothed_covs_np,_ = super().smooth(params)

        # Note: self.smoothed_states and self.smoothed_covariances are set
        #       by the base class smoother (as NumPy arrays).

        return smoothed_states_np, smoothed_covs_np


    # --- Log-Likelihood Methods ---
    def log_likelihood_wrt_params(
        self, params_dict: Dict[str, Any], observations: np.ndarray
    ) -> jnp.ndarray: # Return JAX scalar
        """Calculates the log-likelihood for given parameters and observations.

        Uses the `filter_scan` method internally. This is the primary method
        for evaluating the likelihood for external use (e.g., optimization).

        Args:
            params_dict: Dictionary of parameters.
            observations: Observed returns (T, N).

        Returns:
            Total log-likelihood (JAX scalar). Returns -inf if errors occur
            during parameter processing or filtering results in NaN/Inf.
        """
        try:
            # Convert dict to dataclass, ensuring N and K are correct
            params_jax = self._process_params(params_dict)
            _, _, total_log_lik = self.filter_scan(params_jax, observations)
            # Handle potential NaN/Inf values from filtering (using JAX functions)
            return jnp.where(jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik), -jnp.inf, total_log_lik)
        except (ValueError, TypeError) as e: # Catch only pre-JAX processing errors
            print(f"Warning: Error calculating likelihood in Bellman Filter: {e}")
            return jnp.array(-jnp.inf, dtype=jnp.float64)


    def _log_likelihood_wrt_params_impl(
        self, params: DFSVParamsDataclass, observations: jnp.ndarray
    ) -> float:
        """Internal JAX-compatible implementation for log-likelihood using scan.

        Designed to be JIT-compiled. Assumes inputs are JAX arrays. Only returns
        the final log-likelihood sum, discarding intermediate filter states. This
        is the function that gets JIT-compiled by `jit_log_likelihood_wrt_params`.

        Args:
            params: DFSVParamsDataclass instance with JAX arrays.
            observations: Observed returns (T, N) as JAX array.

        Returns:
            Total log-likelihood (as JAX scalar). Returns -inf if NaN/Inf encountered.
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


    # JIT-compiled version of the log-likelihood calculation
    # This should be defined *after* the implementation it wraps
    @partial(eqx.filter_jit)
    def jit_log_likelihood_wrt_params(self):
        """Returns a JIT-compiled function for log-likelihood calculation.

        The returned function takes (params_dataclass, observations_jax) as input.
        """
        # Return the JIT-compiled internal implementation
        return self._log_likelihood_wrt_params_impl
