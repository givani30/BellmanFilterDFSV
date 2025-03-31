from functools import partial
from typing import Tuple, Union, Dict, Any

import optimistix as optx
from altair import LogicalAndPredicate
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jaxopt
import numpy as np
from jax import jit

from .base import DFSVFilter # Import base class from sibling module
# Update imports to use models.dfsv instead
from qf_thesis.models.dfsv import DFSVParamsDataclass
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

        self._setup_jax_functions()

    # Helper method to standardize parameter handling
    def _process_params(self, params: Union[Dict[str, Any], DFSVParamsDataclass]) -> DFSVParamsDataclass:
        """
        Convert parameter dictionary or ensure it's a DFSVParamsDataclass.

        Args:
            params: Parameters in DFSVParamsDataclass or dictionary format.

        Returns:
            DFSVParamsDataclass: Parameters in the standardized dataclass format.

        Raises:
            TypeError: If the input params type is not supported.
        """
        if isinstance(params, dict):
            # Convert dictionary to DFSVParamsDataclass
            # Ensure N and K are present or use instance defaults
            N = params.get('N', self.N)
            K = params.get('K', self.K)
            # Create the dataclass, potentially raising KeyError if keys missing
            return DFSVParamsDataclass.from_dict(params, N, K)
        elif isinstance(params, DFSVParamsDataclass):
            # If it's already the correct type, return it
            return params
        else:
            # Raise an error for unsupported types
            raise TypeError(f"Unsupported parameter type: {type(params)}. Expected Dict or DFSVParamsDataclass.")

    def _setup_jax_functions(self):
        """
        Set up JIT-compiled JAX functions for efficiency.
        """
        # Build static versions of methods that don't reference self directly

        # JIT the imported implementation functions
        self.build_covariance_jit = jit(build_covariance_impl)

        # Partially apply static args (K, build_covariance_fn) and JIT
        # Pass the JITted build_covariance function
        self.fisher_information_jit = jit(
            partial(fisher_information_impl, K=self.K, build_covariance_fn=self.build_covariance_jit)
        )
        self.log_posterior_jit = jit(
            partial(log_posterior_impl, K=self.K, build_covariance_fn=self.build_covariance_jit)
        )
        self.kl_penalty_jit = jit(kl_penalty_impl) # Renamed attribute

        # Instantiate the BFGS solver for the h update step once
        self.h_solver = optx.BFGS(rtol=1e-4, atol=1e-6)

        # Create block_coordinate_update
        self.block_coordinate_update_jit = jit(
            self._block_coordinate_update_impl,
            static_argnums=(6, 7),  # max_iters and h_solver are static arguments
        )

        # Try to precompile
        try:
            self._precompile_jax_functions()
            print("JAX functions successfully precompiled")
        except Exception as e:
            print(f"Warning: JAX precompilation failed: {e}")
            print("Functions will be compiled during first filter run")

    # Removed _build_covariance_impl (moved to _bellman_impl.py)
    def build_covariance(
        self, lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Public API that calls the JIT-compiled version of build_covariance.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            exp_h (jnp.ndarray): Exponentiated log-volatilities.
            sigma2 (jnp.ndarray): Idiosyncratic variances.

        Returns:
            jnp.ndarray: Covariance matrix.
        """
        return self.build_covariance_jit(lambda_r, exp_h, sigma2)

    # Removed _fisher_information_impl (moved to _bellman_impl.py)

    def fisher_information(
        self, lambda_r: jnp.ndarray, sigma2: jnp.ndarray, alpha: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Public API that calls the JIT-compiled version of fisher_information.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): State vector [f, h].

        Returns:
            jnp.ndarray: Fisher information matrix.
        """
        return self.fisher_information_jit(lambda_r, sigma2, alpha)

    # Removed _log_posterior_impl (moved to _bellman_impl.py)

    def log_posterior(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> float:
        """
        Public API that calls the JIT-compiled version of log_posterior.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): State vector [f, h].
            observation (jnp.ndarray): Observation vector.

        Returns:
            float: Log posterior value.
        """
        return self.log_posterior_jit(lambda_r, sigma2, alpha, observation)

    def _block_coordinate_update_impl(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        pred_state: jnp.ndarray,
        I_pred: jnp.ndarray,
        observation: jnp.ndarray,
        max_iters: int, # Static arg
        h_solver: optx.AbstractMinimiser # Static arg
    ) -> jnp.ndarray:
        """
        Static implementation of block_coordinate_update using external helpers.

        Args:
            lambda_r: Factor loading matrix.
            sigma2: Idiosyncratic variances.
            alpha: Initial state vector [f, h].
            pred_state: Predicted state vector.
            I_pred: Predicted precision matrix.
            observation: Observation vector.
            max_iters: Maximum number of outer block coordinate iterations.
            h_solver: Pre-configured Optimistix solver instance.

        Returns:
            Updated state vector.
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
        # I_h = I_pred[K:, K:] # Not directly needed here, but passed to update_h_bfgs
        I_fh = I_pred[:K, K:]

        # Define the loop body using external functions
        def body_fn(
            _, carry: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """ Single iteration of the block-coordinate update. `carry` is (f, h). """
            f_current, h_current = carry

            # Update factors using external function
            f_new = update_factors(
                log_volatility=h_current,
                lambda_r=lambda_r,
                sigma2=sigma2,
                observation=observation,
                factors_pred=factors_pred,
                log_vols_pred=log_vols_pred,
                I_f=I_f,
                I_fh=I_fh,
                build_covariance_fn=self.build_covariance_jit # Pass JITted function
            )

            # Update log-vols using external function
            h_new = update_h_bfgs(
                h_init=h_current,
                factors=f_new, # Use the newly updated factors
                lambda_r=lambda_r,
                sigma2=sigma2,
                pred_state=pred_state, # Pass full predicted state
                I_pred=I_pred,         # Pass full predicted precision
                observation=observation,
                K=K,
                build_covariance_fn=self.build_covariance_jit, # Pass JITted function
                log_posterior_fn=self.log_posterior_jit,     # Pass JITted function
                h_solver=h_solver,                           # Pass solver instance
                inner_max_steps=15                           # Keep inner steps consistent
            )
            return (f_new, h_new)

        # Use lax.fori_loop to run max_iters times.
        init_carry = (factors_guess, log_vols_guess)
        f_final, h_final = jax.lax.fori_loop(0, max_iters, body_fn, init_carry)

        # Return updated state
        return jnp.concatenate([f_final, h_final])

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

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): Initial state vector [f, h].
            pred_alpha (jnp.ndarray): Predicted state vector.
            I_pred (jnp.ndarray): Predicted precision matrix.
            observation (jnp.ndarray): Observation vector.
            max_iters (int): Maximum number of iterations for the outer block coordinate loop.

        Returns:
            jnp.ndarray: Updated state vector.
        """
        return self.block_coordinate_update_jit(
            lambda_r,
            sigma2,
            alpha,
            pred_alpha,
            I_pred,
            observation,
            max_iters,
            self.h_solver # Pass the pre-configured solver instance
        )

    def _precompile_jax_functions(self):
        """Precompile JIT functions with dummy data"""
        # Create dummy data
        K = self.K
        N = self.N
        state_dim = 2 * K

        # Create dummy arrays of appropriate shapes with float64 dtype
        dummy_lambda_r = jnp.ones((N, K), dtype=jnp.float64)
        dummy_sigma2 = jnp.ones(N, dtype=jnp.float64) # 1D for consistency
        dummy_exp_h = jnp.ones(K, dtype=jnp.float64)
        dummy_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        dummy_pred_alpha = jnp.zeros(state_dim, dtype=jnp.float64)
        dummy_I_pred = jnp.eye(state_dim, dtype=jnp.float64)
        dummy_observation = jnp.zeros(N, dtype=jnp.float64)
        dummy_a_updated = jnp.ones(state_dim, dtype=jnp.float64)
        dummy_I_updated = jnp.eye(state_dim, dtype=jnp.float64)

        # Precompile build_covariance
        print("Precompiling build_covariance...")
        _ = self.build_covariance_jit(dummy_lambda_r, dummy_exp_h, dummy_sigma2).block_until_ready()

        # Precompile fisher_information (K and build_covariance_fn are partially applied)
        print("Precompiling fisher_information...")
        _ = self.fisher_information_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha).block_until_ready()

        # Precompile log_posterior (K and build_covariance_fn are partially applied)
        print("Precompiling log_posterior...")
        _ = self.log_posterior_jit(dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_observation).block_until_ready()

        # Precompile kl_penalty
        print("Precompiling kl_penalty...")
        _ = self.kl_penalty_jit(dummy_pred_alpha, dummy_a_updated, dummy_I_pred, dummy_I_updated).block_until_ready()

        # Precompile block_coordinate_update
        print("Precompiling block_coordinate_update...")
        _ = self.block_coordinate_update_jit(
            dummy_lambda_r,
            dummy_sigma2,
            dummy_alpha,
            dummy_pred_alpha,
            dummy_I_pred,
            dummy_observation,
            max_iters=2, # Use small max_iters for precompilation
            h_solver=self.h_solver
        ).block_until_ready()

    # Removed neg_log_post_h (moved to _bellman_optim.py)

    # Removed _kl_penalty_impl (moved to _bellman_impl.py)

    def kl_penalty(self, a_pred, a_updated, I_pred, I_updated):
        """Public API for KL penalty calculation."""
        return self.kl_penalty_jit(a_pred, a_updated, I_pred, I_updated)

    # Removed obj_and_grad_fn (to be moved to _bellman_optim.py)

    def initialize_state(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass] # Removed DFSV_params from Union
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state and covariance.

        Args:
            params: Model parameters in a supported format (Dictionary or DFSVParamsDataclass).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Initial state and covariance.
        """
        # Process parameters to ensure correct format
        params = self._process_params(params)
        
        K = self.K
        initial_factors = jnp.zeros((K, 1))

        # Initialize log-volatilities to the unconditional mean
        if params.mu.ndim == 1:
            initial_log_vols = params.mu.reshape(-1, 1)
        else:
            initial_log_vols = params.mu.copy()

        # Combine into state vector [f; h]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance
        P_f = jnp.eye(K)

        # Solve discrete Lyapunov equation: P_h = Phi_h * P_h * Phi_h' + Q_h for
        @jit
        def solve_discrete_lyapunov_fixed(Phi, Q, num_iters=20):
            def body_fn(i, X):
                return Phi @ X @ Phi.T + Q

            # Start with Q (a reasonable initial guess)
            X_final = jax.lax.fori_loop(0, num_iters, body_fn, Q)
            return X_final

        P_h = solve_discrete_lyapunov_fixed(params.Phi_h, params.Q_h)

        # Construct block-diagonal covariance
        initial_cov = jnp.block([[P_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), P_h]])

        return initial_state, initial_cov

    def predict(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass], # Removed DFSV_params from Union
        state: np.ndarray,
        cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Bellman prediction step.

        Args:
            params: Parameters of the DFSV model in any supported format.
            state: Current state estimate.
            cov: Current state covariance.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted state and covariance.
        """
        # Process parameters to ensure correct format
        params = self._process_params(params)
        
        # Extract parameters
        K = self.K
        Phi_f = jnp.array(params.Phi_f)
        Phi_h = jnp.array(params.Phi_h)
        mu = jnp.array(params.mu.flatten() if params.mu.ndim > 1 else params.mu)

        # Convert state to JAX array and flatten
        state = jnp.array(state).flatten()
        cov = jnp.array(cov)

        # Extract current state components
        factors = state[:K]
        log_vols = state[K:]

        # Predict factors: E[f_{t+1}|t] = Phi_f @ f_t
        predicted_factors = Phi_f @ factors

        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h @ (log_vols - mu)
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)

        # Combine predicted state
        predicted_state = jnp.concatenate([predicted_factors, predicted_log_vols])

        # Get transition matrix
        F_t = self._get_transition_matrix(params)

        # Process noise covariance (using state-dependent factor noise)
        Q_t = jnp.zeros((self.state_dim, self.state_dim))
        Q_f = jnp.diag(jnp.exp(log_vols.flatten()))  # Factor process noise

 
        Q_h = jnp.array(params.Q_h)  # Convert to JAX array

        Q_t = Q_t.at[:K, :K].set(Q_f)
        Q_t = Q_t.at[K:, K:].set(Q_h)
        predicted_cov = F_t @ cov @ F_t.T + Q_t

        # Ensure the covariance is symmetric
        predicted_cov = (predicted_cov + predicted_cov.T) / 2

        # Ensure state is returned as a column vector
        return predicted_state.reshape(-1, 1), predicted_cov

    def update(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass], # Changed type hint
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the Bellman update step using JAX-based optimization.

        Args:
            params: Parameters of the DFSV model (Dictionary or DFSVParamsDataclass).
            predicted_state: Predicted state.
            predicted_cov: Predicted state covariance.
            observation: Current observation.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Updated state, updated covariance, and log-likelihood contribution.
        """
        # Extract parameters
        K = self.K
        N = self.N
        lambda_r = jnp.array(params.lambda_r)
        sigma2 = jnp.array(
            params.sigma2.reshape(-1, 1)
            if params.sigma2.ndim == 1
            else params.sigma2
        )

        # Convert inputs to JAX arrays
        jax_observation = jnp.array(observation)

        # Compute information matrix (inverse of predicted covariance)
        # Use Cholesky for numerical stability with regularization
        jax_predicted_cov = jnp.array(predicted_cov) + 1e-8 * jnp.eye(self.state_dim)
        try:
            L = jax.scipy.linalg.cholesky(jax_predicted_cov, lower=True)
            jax_I_pred = jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.state_dim))
        except:
            # Fallback to regularized pseudoinverse if Cholesky fails
            jax_I_pred = jnp.linalg.pinv(jax_predicted_cov)

        # Ensure symmetry
        jax_I_pred = (jax_I_pred + jax_I_pred.T) / 2.0

        # Initial guess is the predicted state
        alpha = jnp.array(predicted_state.flatten())

        # Run block diagonal update
        updated_state = self.block_coordinate_update(
            lambda_r,
            sigma2,
            alpha,
            predicted_state,
            jax_I_pred,
            observation,
            max_iters=10,
        ).reshape(-1, 1)

        # Compute the Hessian at the optimum for covariance estimation
        fisher_info = self.fisher_information(lambda_r, sigma2, updated_state)
        I_updated = fisher_info + jax_I_pred + 1e-8 * jnp.eye(self.state_dim)
        
        # Use pseudoinverse for better numerical stability
        updated_cov = jnp.linalg.pinv(I_updated)

        # Ensure symmetry
        updated_cov = (updated_cov + updated_cov.T) / 2.0

        # Augmented Likelihood for parameter estimation
        val = self.log_posterior(lambda_r, sigma2, updated_state, jax_observation)
        penalty = self.kl_penalty(predicted_state, updated_state, jax_I_pred, I_updated)
        log_likelihood = val - penalty  # Negate because we minimize negative log-posterior

        # Return JAX array for log-likelihood to maintain JAX compatibility
        return updated_state, updated_cov, log_likelihood

    def _get_transition_matrix(
        self, params: DFSVParamsDataclass, # Changed type hint
    ) -> jnp.ndarray:
        """
        Get the state transition matrix for the DFSV model.

        Args:
            params: Parameters of the DFSV model (DFSVParamsDataclass).

        Returns:
            jnp.ndarray: Transition matrix.
        """

        # Regular DFSV_params
        K = self.K
        Phi_f = jnp.array(params.Phi_f)
        Phi_h = jnp.array(params.Phi_h)

        # Initialize transition matrix
        F_t = jnp.zeros((2 * K, 2 * K))

        # Top-left block: factor transition
        F_t = F_t.at[:K, :K].set(Phi_f)

        # Bottom-right block: log-volatility transition
        F_t = F_t.at[K:, K:].set(Phi_h)

        return F_t

    def filter(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], y: np.ndarray # Changed type hint
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter on the provided data with JAX optimization.

        This implementation replaces the base class version with a more JAX-efficient
        approach that minimizes host-device transfers and leverages JAX's functional
        programming model.

        Args:
            params: Parameters of the DFSV model (Dictionary or DFSVParamsDataclass).
            y: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                Filtered states with shape (T, state_dim),
                filtered covariances with shape (T, state_dim, state_dim),
                and total log-likelihood.
        """
        try:
            from tqdm.auto import tqdm
        except ImportError:
            # Simple pass-through if tqdm not available
            def tqdm(iterable, **kwargs):
                return iterable

            print("Warning: tqdm not installed. No progress bar will be shown.")
            
        
        # Convert y to JAX array and ensure it's in (T, N) format
        y = jnp.array(y)
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]
        N = y.shape[1]  # Handle both parameter types

        # Create empty arrays for storing results
        filtered_states = jnp.zeros((T, self.state_dim))
        filtered_covs = jnp.zeros((T, self.state_dim, self.state_dim))

        # Initialize state and covariance
        state, cov = self.initialize_state(params)
        log_likelihood = 0.0

        # We can't use lax.scan directly for the full filter because of tqdm,
        # but we can still structure our code in a more JAX-friendly way
        for t in tqdm(range(T), desc="Bellman Filter Progress"):
            # Convert observation to JAX array with correct shape
            observation = y[t : t + 1, :].T.reshape(-1, 1)

            # Prediction step
            predicted_state, predicted_cov = self.predict(params, state, cov)

            # Update step
            state, cov, ll_contrib = self.update(
                params, predicted_state, predicted_cov, observation
            )

            # Store results
            filtered_states = filtered_states.at[t].set(state.flatten())
            filtered_covs = filtered_covs.at[t].set(cov)
            log_likelihood += ll_contrib

        # Store results in object for compatibility with the base class
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        return np.array(filtered_states), np.array(filtered_covs), float(log_likelihood)

    def filter_scan(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], y: np.ndarray # Changed type hint
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter using JAX's scan operation for maximum efficiency.

        This implementation uses JAX's scan operation for efficient filtering without a
        progress bar. Ideal for production use or when performance is critical.

        Args:
            params: Parameters of the DFSV model (Dictionary or DFSVParamsDataclass).
            y: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                Filtered states with shape (T, state_dim),
                filtered covariances with shape (T, state_dim, state_dim),
                and total log-likelihood.
        """

        # Convert y to JAX array and ensure it's in (T, N) format
        y = jnp.array(y)
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        # Initialize state and covariance
        init_state, init_cov = self.initialize_state(params)

        # Define a single step of the filter
        def filter_step(carry, observation):
            state, cov, ll_total = carry

            # Reshape observation
            obs = observation.reshape(-1, 1)

            # Prediction step
            predicted_state, predicted_cov = self.predict(params, state, cov)

            # Update step
            updated_state, updated_cov, ll_contrib = self.update(
                params, predicted_state, predicted_cov, obs
            )

            # Return updated carry and outputs to store
            return (updated_state, updated_cov, ll_total + ll_contrib), (
                updated_state.flatten(),
                updated_cov,
            )

        # Initial carry values for scan
        init_carry = (init_state, init_cov, 0.0)

        # Run filter using lax.scan (much more efficient than a Python loop)
        (_, _, log_likelihood), (filtered_states, filtered_covs) = jax.lax.scan(
            filter_step, init_carry, y
        )

        # Store results in object for compatibility with the base class
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        return filtered_states, filtered_covs, log_likelihood

    def log_likelihood_of_params(
        self,
        pytree_params: Union[Dict[str, Any], DFSVParamsDataclass], # Removed DFSV_params from Union
        observations: jnp.ndarray,
    ) -> float:
        """
        Calculate the log-likelihood of parameters given observations.

        This method is useful for parameter optimization. It uses the filter_scan
        method to efficiently calculate the log-likelihood.

        Args:
            pytree_params: Parameters in any supported format (DFSV_params, Dictionary, or DFSVParamsDataclass)
            observations: Observation data with shape (T, N) or (N, T)

        Returns:
            float: Log-likelihood value
        """
        # Process parameters to ensure correct format
        processed_params = self._process_params(pytree_params)
        
        # Use the non-jitted implementation to avoid issues with self
        return self._log_likelihood_of_params_impl(processed_params, observations)

    def _log_likelihood_of_params_impl(
        self,
        pytree_params: DFSVParamsDataclass, # Changed type hint
        observations: jnp.ndarray,
    ) -> float:
        """
        Implementation of log-likelihood calculation for parameters.

        Args:
            pytree_params: JAX-compatible parameters to evaluate
            observations: Observation data with shape (T, N) or (N, T)

        Returns:
            float: Log-likelihood value
        """
        # Format observations
        if observations.shape[0] < observations.shape[1]:  # If (N, T) format
            observations = observations.T

        # Initialize state
        init_state, init_cov = self.initialize_state(pytree_params)

        # Define a single step of the filter
        def filter_step(carry, observation):
            state, cov, ll_total = carry

            # Reshape observation
            obs = observation.reshape(-1, 1)

            # Prediction step
            predicted_state, predicted_cov = self.predict(pytree_params, state, cov)

            # Update step
            updated_state, updated_cov, ll_contrib = self.update(
                pytree_params, predicted_state, predicted_cov, obs
            )

            # Return only the accumulated log-likelihood and updated state/cov
            return (updated_state, updated_cov, ll_total + ll_contrib), None

        # Initial carry values for scan
        init_carry = (init_state, init_cov, 0.0)

        # Run filter using scan - we only care about the final log-likelihood
        (_, _, log_likelihood), _ = jax.lax.scan(filter_step, init_carry, observations)

        return log_likelihood

    # Create a jit-compiled version that can be used for automatic differentiation
    @staticmethod
    @partial(jit, static_argnums=(0,))
    def jit_log_likelihood_of_params(
        filter_instance,  # The filter instance, marked as static
        pytree_params: DFSVParamsDataclass, # Changed type hint
        observations: jnp.ndarray,
    ) -> float:
        """
        JIT-compiled log-likelihood calculation for automatic differentiation.

        Args:
            filter_instance: The DFSVBellmanFilter instance (static)
            pytree_params: JAX-compatible parameters to evaluate
            observations: Observation data with shape (T, N) or (N, T)

        Returns:
            float: Log-likelihood value
        """
        # Format observations
        if observations.shape[0] < observations.shape[1]:  # If (N, T) format
            observations = observations.T

        # Initialize state
        init_state, init_cov = filter_instance.initialize_state(pytree_params)
        # Define a single step of the filter
        def filter_step(carry, observation):
            state, cov, ll_total = carry

            # Reshape observation
            obs = observation.reshape(-1, 1)

            # Prediction step
            predicted_state, predicted_cov = filter_instance.predict(
                pytree_params, state, cov
            )

            # Update step
            updated_state, updated_cov, ll_contrib = filter_instance.update(
                pytree_params, predicted_state, predicted_cov, obs
            )

            # Return only the accumulated log-likelihood and updated state/cov
            return (updated_state, updated_cov, ll_total + ll_contrib), None

        # Initial carry values for scan
        init_carry = (init_state, init_cov, 0.0)

        # Run filter using scan - we only care about the final log-likelihood
        (_, _, log_likelihood), _ = jax.lax.scan(filter_step, init_carry, observations)

        return log_likelihood
