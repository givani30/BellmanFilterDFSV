# Imports based on bellman.py and bif_implementation_plan.md
from functools import partial
from typing import Tuple, Union, Dict, Any

from jax.experimental import checkify
import optimistix as optx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import jit
import equinox as eqx 
from .base import DFSVFilter # Import base class
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import parameter dataclass

# Import reusable components (adjust paths/names as needed based on actual implementation)
# Import reusable components from _bellman_impl
from ._bellman_impl import (
    build_covariance_impl,
    # fisher_information_impl, # This was commented out in _bellman_impl.py
    log_posterior_impl,
    bif_likelihood_penalty_impl,
    observed_fim_impl # Use this for Fisher Information calculation
)
from ._bellman_optim import update_factors, update_h_bfgs # Reusable optimization helpers

# Placeholder for potential new helper functions if needed
# from ._bellman_info_impl import _kl_penalty_pseudo_lik_impl, stable_log_det, stable_inverse

# TODO: Define helper functions like stable_log_det, stable_inverse if not using jax built-ins directly

class DFSVBellmanInformationFilter(DFSVFilter):
    """
    Bellman Information Filter (BIF) for Dynamic Factor Stochastic Volatility (DFSV) models.

    This class implements a Bellman filter variant that directly propagates the
    information state (state vector `alpha` and information matrix `Omega = P^-1`)
    instead of the covariance matrix `P`. This approach, based on Lange (2024),
    can offer improved numerical stability, especially when dealing with
    high-dimensional state spaces or near-singular covariance matrices.

    The filter estimates factors (f) and log-volatilities (h) using a block
    coordinate descent approach within the update step, similar to the covariance-based
    `DFSVBellmanFilter`, but adapted for the information form. JAX is used for
    automatic differentiation and JIT compilation.

    Attributes:
        N (int): Number of observed time series.
        K (int): Number of latent factors.
        filtered_states (jnp.ndarray | None): Filtered states alpha_{t|t} = [f_{t|t}; h_{t|t}] (T, state_dim) stored internally as JAX array.
        filtered_infos (jnp.ndarray | None): Filtered information matrices Omega_{t|t} (T, state_dim, state_dim) stored internally as JAX array.
        predicted_states (jnp.ndarray | None): Predicted states alpha_{t|t-1} (T, state_dim, 1) stored internally as JAX array.
        predicted_infos (jnp.ndarray | None): Predicted information matrices Omega_{t|t-1} (T, state_dim, state_dim) stored internally as JAX array.
        log_likelihoods (jnp.ndarray | None): Log-likelihood contributions log p(y_t | Y_{1:t-1}) per step (T,) stored internally as JAX array.
        total_log_likelihood (jnp.ndarray | float | None): Total log-likelihood after running filter (JAX scalar from scan, float otherwise).
        h_solver (optx.AbstractMinimiser): Optimistix solver instance used for the 'h' update step.
        # JIT-compiled functions (internal use)
        build_covariance_jit (callable): JITted function for building Sigma_t.
        fisher_information_jit (callable): JITted function for calculating Fisher Information J_observed.
        log_posterior_jit (callable): JITted function for calculating log p(y_t|alpha_t).
        bif_penalty_jit (callable): JITted function for calculating the BIF likelihood penalty. # <-- Renamed attribute
        block_coordinate_update_impl_jit (callable): JITted implementation of the state update optimization.
        predict_jax_info_jit (callable): JITted BIF prediction step.
        update_jax_info_jit (callable): JITted BIF update step.
    """

    def __init__(self, N: int, K: int):
        """
        Initialize the Bellman Information Filter.

        Args:
            N (int): Number of observed time series.
            K (int): Number of latent factors.
        """
        super().__init__(N, K)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Initialize state storage for information form results
        self.filtered_states = None      # (T, state_dim) -> alpha_{t|t}
        self.filtered_infos = None       # (T, state_dim, state_dim) -> Omega_{t|t}
        self.predicted_states = None     # (T, state_dim, 1) -> alpha_{t|t-1}
        self.predicted_infos = None      # (T, state_dim, state_dim) -> Omega_{t|t-1}
        self.log_likelihoods = None      # (T,) -> log p(y_t | Y_{1:t-1})
        self.total_log_likelihood = None # Scalar

        # Setup JIT functions on initialization
        self._setup_jax_functions()


    # Helper method to standardize parameter handling (Copied from bellman.py)
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
        Sets up and JIT-compiles the core JAX functions used by the BIF.

        This includes the prediction and update steps specific to the information filter,
        reused helper functions (like Fisher information, log posterior), the new KL-type
        penalty function, and the block coordinate update optimization routine.
        It also initializes the Optimistix solver used within the update step.
        """
        # --- JIT Helper Functions (Reused & New) ---
        # JIT build_covariance_impl directly
        self.build_covariance_jit = eqx.filter_jit(build_covariance_impl)

        # JIT the observed_fim_impl (renamed to fisher_information_impl)
        self.fisher_information_jit = eqx.filter_jit(partial(observed_fim_impl,K=self.K))

        # JIT log_posterior_impl, partially applying K and the JITted build_covariance
        self.log_posterior_jit = eqx.filter_jit(partial(log_posterior_impl, K=self.K, build_covariance_fn=self.build_covariance_jit))

        # JIT the BIF likelihood penalty function (using imported function)
        self.bif_penalty_jit = eqx.filter_jit(bif_likelihood_penalty_impl)

        # --- Instantiate Optimistix Solver ---
        self.h_solver = optx.BFGS(rtol=1e-4, atol=1e-6)

        # --- JIT Core BIF Steps & Block Coordinate Update ---
        # JIT the block coordinate update implementation
        # Make max_iters and h_solver static for JIT compilation
        # Pass the required JITted helpers via partial application
        self.block_coordinate_update_impl_jit = eqx.filter_jit(            partial(
                self._block_coordinate_update_impl,
                h_solver=self.h_solver,
                build_covariance_fn=self.build_covariance_jit,
                log_posterior_fn=self.log_posterior_jit
            ))

        # JIT the BIF prediction step
        self.predict_jax_info_jit = eqx.filter_jit(self.__predict_jax_info)

        # JIT the BIF update step
        # Pass the required JITted helpers via partial application
        self.update_jax_info_jit = eqx.filter_jit(partial(
                self.__update_jax_info,
                block_coord_update_fn=self.block_coordinate_update_impl_jit,
                fisher_info_fn=self.fisher_information_jit,
                log_posterior_fn=self.log_posterior_jit,
                kl_penalty_fn=self.bif_penalty_jit
            ))

        # Checkify the matrix inversion helper (but don't JIT it here)
        # We will call the checkified version inside the getter methods
        # self._invert_info_matrix_checked = checkify.checkify(self._invert_info_matrix, errors=checkify.float_checks)


    def initialize_state_info(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return JAX arrays (initial_state, initial_info)
        """
        Initializes the filter state (alpha_0) and information matrix (Omega_0).

        Calculates the initial state mean based on unconditional moments (factors=0, log-vols=mu)
        and the initial covariance P_0 using the discrete Lyapunov equation for the log-volatility block.
        The initial information matrix Omega_0 is then computed as the inverse of P_0 using stable methods.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Initial state vector alpha_0 (JAX array, shape (state_dim, 1))
                                             and initial information matrix Omega_0 (JAX array, shape (state_dim, state_dim)).
        """
        params = self._process_params(params) # Ensure JAX arrays inside

        K = self.K
        initial_factors = jnp.zeros((K, 1), dtype=jnp.float64)
        initial_log_vols = params.mu.reshape(-1, 1) # mu is already 1D JAX array

        # Combine into state vector [f; h]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance
        P_f = jnp.eye(K, dtype=jnp.float64)

        # Solve discrete Lyapunov equation using JAX (copied from bellman.py)
        # Note: Consider moving this helper to a shared location if used elsewhere.
        @jit
        def solve_discrete_lyapunov_fixed(Phi, Q, num_iters=20):
            Phi = jnp.asarray(Phi)
            Q = jnp.asarray(Q)
            def body_fn(i, X):
                return Phi @ X @ Phi.T + Q
            X_final = jax.lax.fori_loop(0, num_iters, body_fn, Q)
            return X_final

        P_h = solve_discrete_lyapunov_fixed(params.Phi_h, params.Q_h)

        # Construct block-diagonal initial covariance P_0
        initial_cov = jnp.block([
            [P_f,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), P_h]
        ])

        # Compute initial information matrix Omega_0 = P_0^{-1}
        jitter = 1e-8 # Small jitter for numerical stability during inversion
        initial_cov_jittered = initial_cov + jitter * jnp.eye(self.state_dim, dtype=jnp.float64)

        try:
            # Prefer Cholesky-based inversion for stability and efficiency with SPD matrices
            chol_P0 = jax.scipy.linalg.cholesky(initial_cov_jittered, lower=True)
            initial_info = jax.scipy.linalg.cho_solve((chol_P0, True), jnp.eye(self.state_dim, dtype=jnp.float64))
        except jnp.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails (e.g., matrix not positive definite)
            print("Warning: Cholesky decomposition failed during initial info calculation. Falling back to pinv.")
            initial_info = jnp.linalg.pinv(initial_cov_jittered)

        # Ensure symmetry
        initial_info = (initial_info + initial_info.T) / 2

        return initial_state, initial_info

    # Internal JAX version of predict for Information Filter
    def __predict_jax_info(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        state_post: jnp.ndarray,     # Posterior state alpha_{t-1|t-1} (JAX array)
        info_post: jnp.ndarray,      # Posterior information Omega_{t-1|t-1} (JAX array)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: # Return predicted state and info (JAX arrays)
        """
        Internal JAX implementation of the BIF prediction step.

        Calculates the predicted state mean alpha_{t|t-1} and predicted information matrix Omega_{t|t-1}
        based on the posterior estimates from the previous step (alpha_{t-1|t-1}, Omega_{t-1|t-1}).

        State Prediction: alpha_{t|t-1} = F_t @ alpha_{t-1|t-1} (adjusted for mean-reverting h)
        Information Prediction: Uses the Joseph form / Woodbury identity for numerical stability:
            Q_inv = Q_t^{-1} (calculated using predicted log-vols)
            M = Omega_{t-1|t-1} + F_t^T @ Q_inv @ F_t
            Omega_{t|t-1} = Q_inv - Q_inv @ F_t @ M^{-1} @ F_t^T @ Q_inv

        Operates purely on JAX arrays and is designed to be JIT-compiled.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            state_post: Posterior state estimate alpha_{t-1|t-1} (JAX array, shape (state_dim, 1)).
            info_post: Posterior information matrix Omega_{t-1|t-1} (JAX array, shape (state_dim, state_dim)).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predicted state alpha_{t|t-1} (JAX array, shape (state_dim, 1)),
                                             predicted information Omega_{t|t-1} (JAX array, shape (state_dim, state_dim)).
        """
        # jax.debug.print("predict >>> Inputs: state_post={sp}, info_post={ip}", sp=state_post, ip=info_post)
        K = self.K
        state_dim = self.state_dim
        jitter = 1e-8 # Small jitter for numerical stability
        # --- Add Parameter Checks ---
        # checkify.check(jnp.all(jnp.linalg.eigvalsh(params.Q_h) >= -1e-8), "Q_h must be positive semi-definite, eigenvalues: {evals}", evals=jnp.linalg.eigvalsh(params.Q_h))

        # --- State Prediction ---
        F_t = self._get_transition_matrix(params) # Returns JAX array
        state_post_flat = state_post.flatten()
        factors_post = state_post_flat[:K]
        log_vols_post = state_post_flat[K:]

        predicted_factors = params.Phi_f @ factors_post
        # Use correct state transition for h: h_t = mu + Phi_h (h_{t-1} - mu) + eta_t
        predicted_log_vols = params.mu + params.Phi_h @ (log_vols_post - params.mu)
        predicted_state = jnp.concatenate([predicted_factors, predicted_log_vols])

        # --- Information Prediction ---

        # 1. Calculate Q_t_inv (Inverse of Process Noise Covariance)
        # Q_f = diag(exp(predicted_log_vols)) -> Q_f_inv = diag(exp(-predicted_log_vols))
        ## jax.debug.print("predict: predicted_log_vols for Q_f_inv: {x}", x=predicted_log_vols)
        # jax.debug.print("predict --- predicted_log_vols: {x}", x=predicted_log_vols)
        Q_f_inv = jnp.diag(jnp.exp(-predicted_log_vols))
        ## jax.debug.print("predict: Q_f_inv diagonal after exp: {x}", x=jnp.diag(Q_f_inv))
        # jax.debug.print("predict --- Q_f_inv diag: {x}", x=jnp.diag(Q_f_inv))
        Q_f_inv = Q_f_inv + jitter * jnp.eye(K, dtype=jnp.float64) # Add jitter

        # Q_h_inv = params.Q_h^-1
        Q_h_jittered = params.Q_h + jitter * jnp.eye(K, dtype=jnp.float64)
        try:
            ## jax.debug.print("predict: Q_h_jittered for Cholesky: {x}", x=Q_h_jittered)
            # jax.debug.print("predict --- Q_h_jittered: {x}", x=Q_h_jittered)
            chol_Qh = jax.scipy.linalg.cholesky(Q_h_jittered, lower=True)
            Q_h_inv = jax.scipy.linalg.cho_solve((chol_Qh, True), jnp.eye(K, dtype=jnp.float64))
            ## jax.debug.print("predict: Q_h_inv after cho_solve: {x}", x=Q_h_inv)
            # jax.debug.print("predict --- Q_h_inv: {x}", x=Q_h_inv)
        except jnp.linalg.LinAlgError:
            print("Warning: Cholesky failed for Q_h inversion in predict. Falling back to pinv.")
            Q_h_inv = jnp.linalg.pinv(Q_h_jittered)
        Q_h_inv = (Q_h_inv + Q_h_inv.T) / 2 # Ensure symmetry

        # Construct block diagonal Q_t_inv
        Q_t_inv = jnp.block([
            [Q_f_inv,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), Q_h_inv]
        ])
        Q_t_inv = (Q_t_inv + Q_t_inv.T) / 2 # Ensure symmetry

        # 2. Calculate M = info_post + F_t.T @ Q_t_inv @ F_t
        M = info_post + F_t.T @ Q_t_inv @ F_t
        # jax.debug.print("predict --- M inputs: info_post={ip}, Ft={ft}, Qt_inv={qi}", ip=info_post, ft=F_t, qi=Q_t_inv)
        M = info_post + F_t.T @ Q_t_inv @ F_t
        # jax.debug.print("predict --- M: {x}", x=M)
        # --- Add Condition Number Check ---
        # cond_M = jnp.linalg.cond(M) # Commented out debug print
        ## jax.debug.print("predict: M condition number: {cond}", cond=cond_M) # Commented out debug print
        # --- End Condition Number Check ---
        M_jittered = M + jitter * jnp.eye(state_dim, dtype=jnp.float64)

        # 3. Invert M stably
        try:
            ## jax.debug.print("predict: M_jittered for Cholesky: {x}", x=M_jittered)
            # jax.debug.print("predict --- M_jittered: {x}", x=M_jittered)
            chol_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)
            M_inv = jax.scipy.linalg.cho_solve((chol_M, True), jnp.eye(state_dim, dtype=jnp.float64))
            ## jax.debug.print("predict: M_inv after cho_solve: {x}", x=M_inv)
            # jax.debug.print("predict --- M_inv: {x}", x=M_inv)
        except jnp.linalg.LinAlgError:
            print("Warning: Cholesky failed for M inversion in predict. Falling back to pinv.")
            M_inv = jnp.linalg.pinv(M_jittered)
        M_inv = (M_inv + M_inv.T) / 2 # Ensure symmetry

        # 4. Calculate predicted information: Omega_pred = Q_t_inv - Q_t_inv @ F_t @ M_inv @ F_t.T @ Q_t_inv
        term = Q_t_inv @ F_t @ M_inv @ F_t.T @ Q_t_inv
        predicted_info = Q_t_inv - term
        predicted_info = (predicted_info + predicted_info.T) / 2 # Ensure symmetry
        # jax.debug.print("predict <<< Output: predicted_info={pi}", pi=predicted_info)
        ## jax.debug.print("predict: predicted_info final: {x}", x=predicted_info)

        return predicted_state.reshape(-1, 1), predicted_info
        # jax.debug.print("predict <<< Output: predicted_state={ps}", ps=predicted_state.reshape(-1, 1))

    # --- Block Coordinate Update (Copied from bellman.py) ---
    # Note: This relies on JITted functions (build_covariance_jit, log_posterior_jit)
    # and an h_solver instance, which need to be set up in _setup_jax_functions.
    def _block_coordinate_update_impl(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray, # Expect 1D JAX array
        alpha: jnp.ndarray,
        pred_state: jnp.ndarray,
        I_pred: jnp.ndarray, # Predicted Information Matrix (Omega_{t|t-1})
        observation: jnp.ndarray,
        max_iters: int, # Static arg
        h_solver: optx.AbstractMinimiser, # Static arg
        build_covariance_fn: callable, # Pass JITted build_covariance
        log_posterior_fn: callable # Pass JITted log_posterior
    ) -> jnp.ndarray:
        """
        Internal static implementation of the block coordinate update optimization.

        Solves the state update optimization problem within the BIF update step:
        alpha_{t|t} = argmax_{alpha_t} [ log p(y_t|alpha_t) - 0.5 * ||alpha_t - alpha_{t|t-1}||^2_{Omega_{t|t-1}} ]
        using block coordinate descent on factors (f) and log-volatilities (h).

        This implementation is copied from the covariance-based Bellman filter and adapted
        to accept the predicted information matrix `I_pred` (Omega_{t|t-1}) directly.
        It relies on external helper functions (`update_factors`, `update_h_bfgs`) and
        JIT-compiled functions passed as arguments (`build_covariance_fn`, `log_posterior_fn`).

        Args:
            lambda_r: Factor loading matrix (JAX).
            sigma2: Idiosyncratic variances (1D JAX array).
            alpha: Initial guess for the state vector alpha_{t|t} (JAX array, flattened).
            pred_state: Predicted state vector alpha_{t|t-1} (JAX array, flattened).
            I_pred: Predicted information matrix Omega_{t|t-1} (JAX array).
            observation: Observation vector y_t (JAX array, flattened).
            max_iters: Maximum number of outer block coordinate iterations (static).
            h_solver: Pre-configured Optimistix solver instance (static).
            build_covariance_fn: JIT-compiled function to build observation covariance Sigma_t.
            log_posterior_fn: JIT-compiled function to calculate log p(y_t|alpha_t).

        Returns:
            jnp.ndarray: Optimized updated state vector alpha_{t|t} (JAX array, flattened).
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
            # Note: update_factors needs build_covariance_fn
            f_new = update_factors(
                log_volatility=h_current,
                lambda_r=lambda_r,
                sigma2=sigma2, # Pass 1D sigma2
                observation=observation,
                factors_pred=factors_pred,
                log_vols_pred=log_vols_pred,
                I_f=I_f,
                I_fh=I_fh,
                build_covariance_fn=build_covariance_fn # Pass JITted function
            )

            # Update log-vols using external function
            # Note: update_h_bfgs needs build_covariance_fn and log_posterior_fn
            h_new, h_update_success = update_h_bfgs(
                h_init=h_current,
                factors=f_new, # Use the newly updated factors
                lambda_r=lambda_r,
                sigma2=sigma2, # Pass 1D sigma2
                pred_state=pred_state, # Pass full predicted state
                I_pred=I_pred,         # Pass full predicted precision
                observation=observation,
                K=K,
                build_covariance_fn=build_covariance_fn, # Pass JITted function
                log_posterior_fn=log_posterior_fn,     # Pass JITted function
                h_solver=h_solver,                           # Pass solver instance
                inner_max_steps=100                          # Increased inner steps
            )
            # Note: h_update_success is currently ignored, but could be used for diagnostics
            return (f_new, h_new)

        # Use lax.fori_loop to run max_iters times.
        init_carry = (factors_guess, log_vols_guess)
        f_final, h_final = jax.lax.fori_loop(0, max_iters, body_fn, init_carry)

        # Return updated state
        return jnp.concatenate([f_final, h_final])

    # Internal JAX version of update for Information Filter
    # Decorator removed, checkify applied in _setup_jax_functions
    def __update_jax_info(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        predicted_state: jnp.ndarray, # Predicted state alpha_{t|t-1} (JAX array)
        predicted_info: jnp.ndarray,  # Predicted information Omega_{t|t-1} (JAX array)
        observation: jnp.ndarray,     # Observation y_t (JAX array)
        # Pass JITted functions required by block_coordinate_update and this method
        block_coord_update_fn: callable,
        fisher_info_fn: callable,
        log_posterior_fn: callable,
        kl_penalty_fn: callable
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: # Return updated state, info, log_lik (JAX arrays)
        """
        Internal JAX implementation of the BIF update step.

        Calculates the updated state alpha_{t|t} and information matrix Omega_{t|t},
        along with the log-likelihood contribution for the current time step,
        based on the predicted state/information and the current observation.

        State Update: Performed by optimizing the posterior mode using `block_coord_update_fn`.
        Information Update: Omega_{t|t} = Omega_{t|t-1} + J_observed, where J_observed is the
                          observed Fisher Information calculated using `fisher_info_fn`.
        Log-Likelihood Contribution: Calculated using the formula from Lange (2024, Eq. 40),
                                   combining a fit term (`log_posterior_fn`) and a KL-type
                                   penalty (`kl_penalty_fn`).

        Operates purely on JAX arrays and relies on JIT-compiled helper functions passed as arguments.

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            predicted_state: Predicted state alpha_{t|t-1} (JAX array, shape (state_dim, 1)).
            predicted_info: Predicted information matrix Omega_{t|t-1} (JAX array, shape (state_dim, state_dim)).
            observation: Current observation y_t (JAX array, shape (N,)).
            block_coord_update_fn: JITted block coordinate update function.
            fisher_info_fn: JITted Fisher information function.
            log_posterior_fn: JITted log posterior function.
            kl_penalty_fn: JITted KL-type penalty function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                Updated state alpha_{t|t} (JAX array, shape (state_dim, 1)),
                updated information Omega_{t|t} (JAX array, shape (state_dim, state_dim)),
                log-likelihood contribution log p(y_t|F_{t-1}) (JAX scalar).
        """
        # jax.debug.print("update >>> Inputs: pred_state={ps}, pred_info={pi}, obs={o}", ps=predicted_state, pi=predicted_info, o=observation)
        ## jax.debug.print("update: START - predicted_state: {ps}, predicted_info: {pi}, observation: {o}", ps=predicted_state, pi=predicted_info, o=observation)
        K = self.K
        N = self.N
        state_dim = self.state_dim
        lambda_r = params.lambda_r
        sigma2 = params.sigma2 # Assumed 1D JAX array
        jitter = 1e-6 # Jitter for information update

        # --- Add Parameter Checks ---
        # checkify.check(jnp.all(sigma2 > 0), "sigma2 must be positive, got {s2}", s2=sigma2)
        jax_observation = observation.flatten()

        # --- State Update ---
        # Initial guess for optimization is the predicted state
        alpha_init_guess = predicted_state.flatten()

        # Run block coordinate update (using the passed JITted function)
        # Note: block_coord_update_fn needs access to h_solver and build_covariance_fn
        # These should be configured during _setup_jax_functions and partially applied
        # or passed through if block_coord_update_fn is defined within _setup_jax_functions.
        # Assuming block_coord_update_fn is correctly configured.
        alpha_updated = block_coord_update_fn(
            lambda_r=lambda_r,
            sigma2=sigma2,
            alpha=alpha_init_guess,
            pred_state=predicted_state.flatten(),
            I_pred=predicted_info, # Pass predicted info Omega_{t|t-1}
            observation=jax_observation,
            max_iters=10 # Use a reasonable default or make configurable
            # h_solver, build_covariance_fn, log_posterior_fn are assumed bound
        )

        ## jax.debug.print("update: alpha_updated after block_coord: {x}", x=alpha_updated)
        # jax.debug.print("update --- alpha_updated: {x}", x=alpha_updated)
        # --- Information Update ---
        # Calculate Observed Fisher Information J_observed = -Hessian(log p(y_t|alpha_t))
        # Reuse the fisher_information_impl function (passed as fisher_info_fn)
        # Note: fisher_info_fn needs build_covariance_fn bound to it.
        J_observed = fisher_info_fn(lambda_r, sigma2, alpha_updated, observation)
        ## jax.debug.print("update: J_observed after fisher_info_fn: {x}", x=J_observed)
        # jax.debug.print("update --- J_observed: {x}", x=J_observed)

        # Compute updated information matrix Omega_{t|t} = Omega_{t|t-1} + J_observed
        updated_info = predicted_info + J_observed + jitter * jnp.eye(state_dim, dtype=jnp.float64)
        updated_info = (updated_info + updated_info.T) / 2 # Ensure symmetry

        # jax.debug.print("update --- updated_info: {x}", x=updated_info)
        # --- Log-Likelihood Contribution ---
        # Calculate fit term log p(y_t | alpha_{t|t})
        # Reuse log_posterior_impl (passed as log_posterior_fn)
        log_lik_fit = log_posterior_fn(lambda_r, sigma2, alpha_updated, jax_observation)
        ## jax.debug.print("update: log_lik_fit after log_posterior_fn: {x}", x=log_lik_fit)
        # jax.debug.print("update --- log_lik_fit: {x}", x=log_lik_fit)

        # Calculate KL-type penalty term (using the passed JITted function)
        kl_penalty = kl_penalty_fn(
            a_pred=predicted_state.flatten(),
            a_updated=alpha_updated,
            Omega_pred=predicted_info,
            Omega_post=updated_info
        )
        ## jax.debug.print("update: kl_penalty after kl_penalty_fn: {x}", x=kl_penalty)
        # jax.debug.print("update --- kl_penalty: {x}", x=kl_penalty)

        # Combine: log p(y_t|F_{t-1}) ≈ log p(y_t|alpha_{t|t}) - KL_penalty
        log_lik_contrib = log_lik_fit - kl_penalty
        ## jax.debug.print("update: log_lik_contrib final: {x}", x=log_lik_contrib)
        # jax.debug.print("update <<< Output: log_lik_contrib={llc}", llc=log_lik_contrib)

        # Return results as JAX arrays (reshaped state), keep log_lik as JAX scalar
        return alpha_updated.reshape(-1, 1), updated_info, log_lik_contrib

    # --- Filtering Methods (Adapted for Information Filter) ---
    def filter(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]: # Returns (filtered_states, filtered_infos, total_log_lik)
        """
        Run the Bellman Information Filter using a standard Python loop.

        Iterates through time steps, calling the JIT-compiled predict and update
        steps (`predict_jax_info_jit`, `update_jax_info_jit`). Stores results
        internally as JAX arrays and returns them converted to NumPy arrays.

        NOTE: This method uses a Python loop and involves conversions between
              NumPy and JAX arrays at each step (for observations and storing results).
              It may not be as performant as `filter_scan` and might be incompatible
              with JIT compilation if called within a JIT context itself.
              It's primarily useful for debugging or when `jax.lax.scan` causes issues
              (e.g., memory leaks with complex steps).

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed data with shape (T, N) (NumPy array).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                Filtered states alpha_{t|t} (NumPy array, shape (T, state_dim)),
                Filtered information matrices Omega_{t|t} (NumPy array, shape (T, state_dim, state_dim)),
                Total log-likelihood (float).
        """
        params_jax = self._process_params(params) # Ensure correct format with JAX arrays
        self._setup_jax_functions() # Ensure JIT functions are ready

        T = observations.shape[0]
        state_dim = self.state_dim

        # Initialize storage (JAX arrays)
        filtered_states_jax = jnp.zeros((T, state_dim), dtype=jnp.float64)
        filtered_infos_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        predicted_states_jax = jnp.zeros((T, state_dim, 1), dtype=jnp.float64)
        predicted_infos_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        log_likelihoods_jax = jnp.zeros(T, dtype=jnp.float64)

        # Initialization (t=0) - Use BIF initialization
        initial_state_jax, initial_info_jax = self.initialize_state_info(params_jax)

        # Start loop state with initial values
        state_post_prev = initial_state_jax
        info_post_prev = initial_info_jax

        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        # Filtering loop (using JAX arrays for storage and computation)
        for t in tqdm(range(T), desc="Bellman Information Filtering (Python Loop)"):
            # Predict step (predict state t based on t-1 posterior) -> returns JAX arrays
            pred_state_t_jax, pred_info_t_jax = self.predict_jax_info_jit(
                params_jax, state_post_prev, info_post_prev
            )

            # Store predicted results
            predicted_states_jax = predicted_states_jax.at[t].set(pred_state_t_jax)
            predicted_infos_jax = predicted_infos_jax.at[t].set(pred_info_t_jax)

            # Convert observation for this step to JAX array
            obs_t_jax = jnp.array(observations[t])

            # Update step (update state t using observation t) -> returns JAX arrays
            updated_state_t_jax, updated_info_t_jax, log_lik_t_jax = self.update_jax_info_jit(
                params_jax, pred_state_t_jax, pred_info_t_jax, obs_t_jax
            )

            # Store filtered results using JAX functional updates
            filtered_states_jax = filtered_states_jax.at[t].set(updated_state_t_jax.flatten()) # Flatten state before storing
            filtered_infos_jax = filtered_infos_jax.at[t].set(updated_info_t_jax)
            log_likelihoods_jax = log_likelihoods_jax.at[t].set(log_lik_t_jax) # Store JAX scalar

            # Update loop state for next iteration
            state_post_prev = updated_state_t_jax
            info_post_prev = updated_info_t_jax


        # Store results internally as JAX arrays
        self.filtered_states = filtered_states_jax
        self.filtered_infos = filtered_infos_jax
        self.predicted_states = predicted_states_jax
        self.predicted_infos = predicted_infos_jax
        self.log_likelihoods = log_likelihoods_jax
        self.total_log_likelihood = float(jnp.sum(log_likelihoods_jax)) # Sum JAX array, convert to float

        # Return NumPy arrays by calling getter methods (which will handle conversion)
        # Return NumPy arrays by calling getter methods
        return self.get_filtered_states(), self.get_filtered_information_matrices(), self.get_total_log_likelihood()

    def filter_scan(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, jnp.ndarray]: # Returns (filtered_states_np, filtered_infos_np, total_log_lik_jax)
        """
        Run the Bellman Information Filter using `jax.lax.scan`.

        This method leverages `jax.lax.scan` for potentially faster execution compared
        to the Python loop in `filter`, especially on accelerators (GPU/TPU).
        It performs the entire filtering loop within JAX's compiled computation graph.

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed data with shape (T, N) (NumPy array).

        Returns:
            Tuple[np.ndarray, np.ndarray, jnp.ndarray]:
                Filtered states alpha_{t|t} (NumPy array, shape (T, state_dim)),
                Filtered information matrices Omega_{t|t} (NumPy array, shape (T, state_dim, state_dim)),
                Total log-likelihood (JAX scalar).
        """
        params_jax = self._process_params(params) # Ensure correct format (contains JAX arrays)
        self._setup_jax_functions() # Ensure JIT functions are ready
        T = observations.shape[0]

        # Initialization (get initial state/info as JAX arrays)
        initial_state_jax, initial_info_jax = self.initialize_state_info(params_jax)
        # Ensure carry types are JAX compatible (float for sum)
        initial_carry = (initial_state_jax, initial_info_jax, jnp.array(0.0, dtype=jnp.float64)) # state, info, log_lik_sum

        # JAX observations
        jax_observations = jnp.array(observations)

        # Define the step function for lax.scan (operates purely on JAX types)
        # Note: params_jax is implicitly captured if not JITting the step function itself.
        # If JITting filter_step, params_jax would need to be static or passed differently.
        def filter_step(carry, obs_t):
            state_t_minus_1_jax, info_t_minus_1_jax, log_lik_sum_t_minus_1 = carry

            # Predict step (predict state t based on t-1 posterior) -> returns JAX arrays
            pred_state_t_jax, pred_info_t_jax = self.predict_jax_info_jit(
                params_jax, state_t_minus_1_jax, info_t_minus_1_jax
            )

            # Update step (update state t using observation t) -> returns JAX arrays
            updated_state_t_jax, updated_info_t_jax, log_lik_t_jax = self.update_jax_info_jit(
                params_jax, pred_state_t_jax, pred_info_t_jax, obs_t
            )

            # Prepare carry for next step (using JAX arrays)
            next_carry = (updated_state_t_jax, updated_info_t_jax, log_lik_sum_t_minus_1 + log_lik_t_jax)

            # What we store for this time step t (JAX arrays)
            scan_output = (pred_state_t_jax, pred_info_t_jax, updated_state_t_jax, updated_info_t_jax, log_lik_t_jax)
            return next_carry, scan_output

        # Run the scan
        final_carry, scan_results = jax.lax.scan(filter_step, initial_carry, jax_observations)

        # Unpack results (still JAX arrays)
        predicted_states_scan, predicted_infos_scan, filtered_states_scan, filtered_infos_scan, log_likelihoods_scan = scan_results

        # Assign final results directly as JAX arrays
        self.predicted_states = predicted_states_scan # Shape (T, state_dim, 1)
        self.predicted_infos = predicted_infos_scan   # Shape (T, state_dim, state_dim)
        # Reshape filtered states from (T, state_dim, 1) to (T, state_dim) before storing
        self.filtered_states = filtered_states_scan.reshape(T, self.state_dim)
        self.filtered_infos = filtered_infos_scan     # Shape (T, state_dim, state_dim)
        self.log_likelihoods = log_likelihoods_scan # Shape (T,)
        # Store final log-likelihood sum as JAX scalar
        self.total_log_likelihood = final_carry[2]


        # Return NumPy arrays for states/infos, JAX scalar for loglik
        # Return NumPy arrays for states/infos, JAX scalar for loglik by calling getter methods
        return self.get_filtered_states(), self.get_filtered_information_matrices(), self.get_total_log_likelihood()

    # --- Getter Methods (Adapted for Information Filter) ---
    # Convert internal JAX arrays to NumPy for external use

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

    def get_filtered_information_matrices(self) -> np.ndarray | None:
        """Returns the filtered state information matrices Omega_{t|t} (T, state_dim, state_dim) as a NumPy array."""
        infos_jax = getattr(self, 'filtered_infos', None)
        return np.asarray(infos_jax) if infos_jax is not None else None

    def get_predicted_states(self) -> np.ndarray | None:
        """Returns the predicted states alpha_{t|t-1} (T, state_dim, 1) as a NumPy array."""
        states_jax = getattr(self, 'predicted_states', None)
        return np.asarray(states_jax) if states_jax is not None else None

    def get_predicted_information_matrices(self) -> np.ndarray | None:
        """Returns the predicted state information matrices Omega_{t|t-1} (T, state_dim, state_dim) as a NumPy array."""
        infos_jax = getattr(self, 'predicted_infos', None)
        return np.asarray(infos_jax) if infos_jax is not None else None

    def get_log_likelihoods(self) -> np.ndarray | None:
        """Returns the log-likelihood contributions per step (T,) as a NumPy array."""
        lls_jax = getattr(self, 'log_likelihoods', None)
        return np.asarray(lls_jax) if lls_jax is not None else None

    def get_total_log_likelihood(self) -> float | jnp.ndarray | None:
        """
        Returns the total log-likelihood.
        Returns float if filter() was run, JAX scalar if filter_scan() was run.
        """
        return getattr(self, 'total_log_likelihood', None)
    # --- Methods to derive covariance from information ---

    @eqx.filter_jit
    def _invert_info_matrix(self, info_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Helper function to stably invert a single information matrix (Omega -> P).
        Uses Cholesky decomposition for potentially better stability than direct inv.
        Relies on JAX default error handling if Cholesky fails (e.g., matrix not SPD).
        """
        jitter = 1e-8 # Consistent jitter
        state_dim = self.state_dim
        info_jittered = info_matrix + jitter * jnp.eye(state_dim, dtype=jnp.float64)
        # Perform Cholesky-based inversion - let errors propagate if Cholesky fails
        # jax.debug.print("_invert --- info_jittered: {x}", x=info_jittered)
        ## jax.debug.print("_invert_info_matrix: input info_jittered: {x}", x=info_jittered)
        chol_info = jax.scipy.linalg.cholesky(info_jittered, lower=True)
        cov_matrix = jax.scipy.linalg.cho_solve((chol_info, True), jnp.eye(state_dim, dtype=jnp.float64))
        ## jax.debug.print("_invert_info_matrix: output cov_matrix: {x}", x=cov_matrix)
        # jax.debug.print("_invert --- cov_matrix: {x}", x=cov_matrix)
        # Ensure symmetry
        return (cov_matrix + cov_matrix.T) / 2

    def get_predicted_covariances(self) -> np.ndarray | None:
        """
        Calculates and returns the predicted state covariances P_{t|t-1}
        by inverting the stored predicted information matrices Omega_{t|t-1}.

        Note: This performs potentially expensive inversions and is intended for
              analysis after filtering, not during the filter loop itself.

        Returns:
            np.ndarray | None: Predicted state covariances (T, state_dim, state_dim) as NumPy array, or None.
        """
        pred_infos_jax = getattr(self, 'predicted_infos', None)
        if pred_infos_jax is None:
            return None

        # Use vmap to apply the inversion function across the time dimension (axis 0)
        # JIT the vmapped function for potential speedup
        # Let potential errors from vmap or _invert_info_matrix propagate.
        vmapped_inverter = jit(jax.vmap(self._invert_info_matrix, in_axes=0))
        pred_covs_jax = vmapped_inverter(pred_infos_jax)
        return np.asarray(pred_covs_jax)


    def get_predicted_variances(self) -> np.ndarray | None:
        """
        Calculates and returns the predicted state variances (diagonal of P_{t|t-1}).

        This is a convenience method that calls get_predicted_covariances()
        and extracts the diagonal elements.

        Returns:
            np.ndarray | None: Predicted state variances (T, state_dim) as NumPy array, or None.
        """
        pred_covs_np = self.get_predicted_covariances()
        if pred_covs_np is None:
            return None
        # Extract diagonal elements for each time step
        return np.diagonal(pred_covs_np, axis1=1, axis2=2)
    
    def get_filtered_covariances(self) -> np.ndarray | None:
        """
        Calculates and returns the filtered state covariances P_{t|t}
        by inverting the stored filtered information matrices Omega_{t|t}.

        Note: This performs potentially expensive inversions and is intended for
              analysis after filtering, not during the filter loop itself.

        Returns:
            np.ndarray | None: Filtered state covariances (T, state_dim, state_dim) as NumPy array, or None.
        """
        filtered_infos_jax = getattr(self, 'filtered_infos', None)
        if filtered_infos_jax is None:
            return None

        # Use vmap to apply the inversion function across the time dimension (axis 0)
        vmapped_inverter = jit(jax.vmap(self._invert_info_matrix, in_axes=0))
        filtered_covs_jax = vmapped_inverter(filtered_infos_jax)
        return np.asarray(filtered_covs_jax)

    def get_filtered_variances(self) -> np.ndarray | None:
        """
        Calculates and returns the filtered state variances (diagonal of P_{t|t}).

        This is a convenience method that calls get_filtered_covariances()
        and extracts the diagonal elements.

        Returns:
            np.ndarray | None: Filtered state variances (T, state_dim) as NumPy array, or None.
        """
        filtered_covs_np = self.get_filtered_covariances()
        if filtered_covs_np is None:
            return None
        # Extract diagonal elements for each time step
        return np.diagonal(filtered_covs_np, axis1=1, axis2=2)


    # --- Likelihood Calculation Methods (Adapted for BIF) ---

    def log_likelihood_of_params(
        self, params_dict: Dict[str, Any], observations: np.ndarray
    ) -> jnp.ndarray: # Return JAX scalar
        """
        Calculates the total pseudo log-likelihood for given parameters and observations using the BIF.

        This method serves as the public API for likelihood calculation. It handles
        parameter processing (dictionary to dataclass conversion) and calls the
        `filter_scan` method internally to get the total log-likelihood.
        It includes error handling for issues during filtering or parameter processing.

        Args:
            params_dict: Dictionary containing the model parameters.
            observations: Observed data (NumPy array, shape (T, N)).

        Returns:
            jnp.ndarray: Total pseudo log-likelihood (JAX scalar). Returns -inf if errors occur.
        """
        try:
            # Convert dict to dataclass, ensuring N and K are correct
            params_jax = self._process_params(params_dict)
            # filter_scan returns (filtered_states_np, filtered_infos_np, total_log_lik_jax)
            _, _, total_log_lik = self.filter_scan(params_jax, observations)
            # Handle potential NaN/Inf values from filtering (using JAX functions)
            return jnp.where(jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik), -jnp.inf, total_log_lik)
        except (ValueError, TypeError) as e: # Catch only pre-JAX processing errors
            # Handle errors during parameter processing or filtering
            print(f"Warning: Error calculating BIF likelihood: {e}")
            # Return JAX representation of -inf
            return jnp.array(-jnp.inf, dtype=jnp.float64)


    def _log_likelihood_of_params_impl(
        self, params: DFSVParamsDataclass, observations: jnp.ndarray
    ) -> jnp.ndarray: # Return JAX scalar
        """
        Internal JAX-compatible implementation for BIF log-likelihood calculation using scan.

        This method performs the core calculation using `jax.lax.scan` over the
        BIF predict and update steps. It's designed to be JIT-compiled via the
        `jit_log_likelihood_of_params` method. It assumes inputs are already
        correctly formatted JAX arrays (DFSVParamsDataclass, observations).
        It only returns the final log-likelihood sum, discarding intermediate filter states.

        Args:
            params: DFSVParamsDataclass instance with JAX arrays.
            observations: Observed returns (T, N) as JAX array.

        Returns:
            jnp.ndarray: Total pseudo log-likelihood (JAX scalar). Returns -inf if NaNs/Infs encountered.
        """
        # Ensure JIT functions are ready (important if this is called directly)
        # Note: Calling setup within a JITted function might have issues.
        # It's better to ensure setup is called once outside.
        # self._setup_jax_functions() # Removed this call from here

        T = observations.shape[0]

        # Initialization (use BIF JAX arrays)
        initial_state_jax, initial_info_jax = self.initialize_state_info(params)
        # Ensure carry types are JAX compatible
        initial_carry = (initial_state_jax, initial_info_jax, jnp.array(0.0, dtype=jnp.float64)) # state, info, log_lik_sum

        # Define the step function for lax.scan (operates purely on JAX types)
        def filter_step(carry, obs_t):
            state_t_minus_1_jax, info_t_minus_1_jax, log_lik_sum_t_minus_1 = carry
            # Predict step -> returns JAX arrays
            pred_state_t_jax, pred_info_t_jax = self.predict_jax_info_jit(
                params, state_t_minus_1_jax, info_t_minus_1_jax
            )
            # Update step -> returns JAX arrays
            updated_state_t_jax, updated_info_t_jax, log_lik_t_jax = self.update_jax_info_jit(
                params, pred_state_t_jax, pred_info_t_jax, obs_t
            )
            # Prepare carry for next step
            next_carry = (updated_state_t_jax, updated_info_t_jax, log_lik_sum_t_minus_1 + log_lik_t_jax)
            # We only need the carry for the final likelihood
            return next_carry, None # Don't store intermediate results

        # Run the scan
        final_carry, _ = jax.lax.scan(filter_step, initial_carry, observations)

        total_log_lik = final_carry[2] # JAX scalar

        # Replace NaN/Inf with -inf for optimization stability
        return jnp.where(jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik), -jnp.inf, total_log_lik)


    def jit_log_likelihood_of_params(self) -> callable:
        """
        Returns a JIT-compiled version of the BIF log-likelihood function.

        This method first ensures that all necessary internal JAX functions are set up
        (via `_setup_jax_functions`) and then returns a JIT-compiled version of the
        `_log_likelihood_of_params_impl` method.

        The returned function is suitable for use in gradient-based optimization routines
        (like those in `jax.scipy.optimize` or `optimistix`), as it takes parameters
        (as a DFSVParamsDataclass pytree) and observations (as a JAX array) and returns
        the total pseudo log-likelihood as a JAX scalar.

        Returns:
            callable: A JIT-compiled function `likelihood_fn(params: DFSVParamsDataclass, observations: jnp.ndarray) -> jnp.ndarray`.
        """
        # Ensure JIT functions are set up before returning the JITted likelihood function
        self._setup_jax_functions()
        # JIT the implementation method directly
        # Caller must ensure inputs are correct JAX types (DFSVParamsDataclass, jnp.ndarray)
        return eqx.filter_jit(self._log_likelihood_of_params_impl)

    # --- Helper Methods ---
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
