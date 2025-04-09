import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
from jax import jit
from functools import partial
from typing import Tuple, Union, Dict, Any, Callable

# Import base class and parameter dataclass
from .base import DFSVFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

# Import reusable components from _bellman_impl and _bellman_optim
from ._bellman_impl import (
    build_covariance_impl,
    log_posterior_impl,
    bif_likelihood_penalty_impl,
    observed_fim_impl
)
from ._bellman_optim import update_factors, update_h_bfgs, _block_coordinate_update_impl
import optimistix as optx
import jax.scipy.linalg
from jax.experimental import checkify


class DFSVBellmanInformationFilter(DFSVFilter):
    """Bellman Information Filter (BIF) for DFSV models.

    Implements a Bellman filter variant propagating the information state
    (state vector `alpha` and information matrix `Omega = P^-1`). This can improve numerical stability compared to
    covariance propagation.

    Uses JAX for JIT compilation and automatic differentiation. Estimates
    factors (f) and log-volatilities (h) via block coordinate descent.

    Attributes:
        N (int): Number of observed time series.
        K (int): Number of latent factors.
        filtered_states (Optional[jnp.ndarray]): Filtered states alpha_{t|t}
            (T, state_dim) stored internally as JAX array.
        filtered_infos (Optional[jnp.ndarray]): Filtered information matrices
            Omega_{t|t} (T, state_dim, state_dim) stored internally as JAX array.
        filtered_covs (Optional[np.ndarray]): Filtered covariance matrices P_{t|t}
            (T, state_dim, state_dim) stored internally as NumPy array after calling
            `get_filtered_covariances` or `smooth`.
        predicted_states (Optional[jnp.ndarray]): Predicted states alpha_{t|t-1}
            (T, state_dim, 1) stored internally as JAX array.
        predicted_infos (Optional[jnp.ndarray]): Predicted information matrices
            Omega_{t|t-1} (T, state_dim, state_dim) stored internally as JAX array.
        log_likelihoods (Optional[jnp.ndarray]): Log-likelihood contributions
            log p(y_t | Y_{1:t-1}) per step (T,) stored internally as JAX array.
        total_log_likelihood (Optional[Union[jnp.ndarray, float]]): Total
            log-likelihood after running filter (JAX scalar from scan, float
            otherwise).
        smoothed_states (Optional[np.ndarray]): Smoothed states alpha_{t|T}
            (T, state_dim) stored internally as NumPy array after smoothing.
        smoothed_covariances (Optional[np.ndarray]): Smoothed covariances P_{t|T}
            (T, state_dim, state_dim) stored internally as NumPy array after smoothing.
        h_solver (optx.AbstractMinimiser): Optimistix solver for 'h' update.
        build_covariance_jit (Callable): JIT-compiled covariance builder.
        fisher_information_jit (Callable): JIT-compiled Fisher info calculator.
        log_posterior_jit (Callable): JIT-compiled log posterior calculator.
        bif_penalty_jit (Callable): JIT-compiled BIF penalty calculator.
        block_coordinate_update_impl_jit (Callable): JIT-compiled block update.
        predict_jax_info_jit (Callable): JIT-compiled BIF prediction step.
        update_jax_info_jit (Callable): JIT-compiled BIF update step.
    """

    # Add storage for filtered covariances needed by smoother
    filtered_covs: Union[np.ndarray, None] = None

    def __init__(self, N: int, K: int):
        """Initializes the DFSVBellmanInformationFilter.

        Args:
            N: Number of observed time series.
            K: Number of latent factors.
        """
        super().__init__(N, K)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Initialize state storage for information form results
        self.filtered_states = None      # (T, state_dim) -> alpha_{t|t} (JAX)
        self.filtered_infos = None       # (T, state_dim, state_dim) -> Omega_{t|t} (JAX)
        self.filtered_covs = None        # (T, state_dim, state_dim) -> P_{t|t} (NumPy, populated by get_filtered_covariances/smooth)
        self.predicted_states = None     # (T, state_dim, 1) -> alpha_{t|t-1} (JAX)
        self.predicted_infos = None      # (T, state_dim, state_dim) -> Omega_{t|t-1} (JAX)
        self.log_likelihoods = None      # (T,) -> log p(y_t | Y_{1:t-1}) (JAX)
        self.total_log_likelihood = None # Scalar (JAX or float)
        self.smoothed_states = None      # (T, state_dim) -> alpha_{t|T} (NumPy)
        self.smoothed_covariances = None # (T, state_dim, state_dim) -> P_{t|T} (NumPy)


        # Setup JIT functions on initialization
        self._setup_jax_functions()


    # _process_params is inherited from DFSVFilter base class

    def _setup_jax_functions(self):
        """Sets up and JIT-compiles the core JAX functions used by the BIF.

        This includes the prediction and update steps specific to the information
        filter, reused helper functions (like Fisher information, log posterior),
        the BIF penalty function, and the block coordinate update optimization
        routine. It also initializes the Optimistix solver.
        """
        # --- JIT Helper Functions (Reused & New) ---
        self.build_covariance_jit = eqx.filter_jit(build_covariance_impl)
        self.fisher_information_jit = eqx.filter_jit(partial(observed_fim_impl,K=self.K))
        self.log_posterior_jit = eqx.filter_jit(partial(log_posterior_impl, K=self.K, build_covariance_fn=self.build_covariance_jit))
        self.bif_penalty_jit = eqx.filter_jit(bif_likelihood_penalty_impl)

        # --- Instantiate Optimistix Solver ---
        self.h_solver = optx.BFGS(rtol=1e-4, atol=1e-6)

        # --- JIT Core BIF Steps & Block Coordinate Update ---
        # JIT the imported _block_coordinate_update_impl
        self.block_coordinate_update_impl_jit = eqx.filter_jit(
            partial(
                _block_coordinate_update_impl, # Use imported function
                K=self.K, # Pass K explicitly
                h_solver=self.h_solver, # Pass solver instance
                build_covariance_fn=self.build_covariance_jit, # Pass JITted dependency
                log_posterior_fn=self.log_posterior_jit # Pass JITted dependency
            )
        )

        # JIT the BIF prediction step
        self.predict_jax_info_jit = eqx.filter_jit(self.__predict_jax_info)

        # JIT the BIF update step
        self.update_jax_info_jit = eqx.filter_jit(partial(
                self.__update_jax_info,
                block_coord_update_fn=self.block_coordinate_update_impl_jit,
                fisher_info_fn=self.fisher_information_jit,
                log_posterior_fn=self.log_posterior_jit,
                kl_penalty_fn=self.bif_penalty_jit
            ))

        # Checkify the matrix inversion helper (but don't JIT it here)
        # self._invert_info_matrix_checked = checkify.checkify(self._invert_info_matrix, errors=checkify.float_checks)


    def initialize_state(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initializes the BIF state (alpha_0) and information matrix (Omega_0).

        Calculates the initial state mean based on unconditional moments and the
        initial covariance P_0 using the discrete Lyapunov equation. The initial
        information matrix Omega_0 is then computed as the inverse of P_0.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).

        Returns:
            A tuple containing:
                - initial_state: Initial state vector alpha_0 (JAX array, shape
                  (state_dim, 1)).
                - initial_info: Initial information matrix Omega_0 (JAX array,
                  shape (state_dim, state_dim)).
        """
        params = self._process_params(params) # Ensure JAX arrays inside

        K = self.K
        initial_factors = jnp.zeros((K, 1), dtype=jnp.float64)
        initial_log_vols = params.mu.reshape(-1, 1) # mu is already 1D JAX array

        # Combine into state vector [f; h]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance
        P_f = jnp.eye(K, dtype=jnp.float64)

        # Solve discrete Lyapunov equation using the static helper from base class
        P_h = self._solve_discrete_lyapunov_jax(params.Phi_h, params.Q_h) # Use base class method

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs the BIF prediction step (internal JAX implementation).

        Calculates the predicted state mean alpha_{t|t-1} and predicted
        information matrix Omega_{t|t-1} based on the previous posterior estimates.

        State Prediction: alpha_{t|t-1} = F_t @ alpha_{t-1|t-1} (adjusted for mean-reverting h)
        Information Prediction: Uses the Joseph form / Woodbury identity:
            Q_inv = Q_t^{-1}
            M = Omega_{t-1|t-1} + F_t^T @ Q_inv @ F_t
            Omega_{t|t-1} = Q_inv - Q_inv @ F_t @ M^{-1} @ F_t^T @ Q_inv

        Args:
            params: Model parameters (DFSVParamsDataclass with JAX arrays).
            state_post: Posterior state estimate alpha_{t-1|t-1} (JAX array,
                shape (state_dim, 1)).
            info_post: Posterior information matrix Omega_{t-1|t-1} (JAX array,
                shape (state_dim, state_dim)).

        Returns:
            A tuple containing:
                - predicted_state: Predicted state alpha_{t|t-1} (JAX array,
                  shape (state_dim, 1)).
                - predicted_info: Predicted information Omega_{t|t-1} (JAX array,
                  shape (state_dim, state_dim)).
       """
        K = self.K
        state_dim = self.state_dim
        jitter = 1e-8 # Small jitter for numerical stability

        # --- State Prediction ---
        # Get transition matrix using base class method
        F_t = self._get_transition_matrix(params, self.K) # Use base class static method

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
        Q_f_inv = jnp.diag(jnp.exp(-predicted_log_vols))
        Q_f_inv = Q_f_inv + jitter * jnp.eye(K, dtype=jnp.float64) # Add jitter

        # Q_h_inv = params.Q_h^-1
        Q_h_jittered = params.Q_h + jitter * jnp.eye(K, dtype=jnp.float64)
        chol_Qh = jax.scipy.linalg.cholesky(Q_h_jittered, lower=True)
        Q_h_inv = jax.scipy.linalg.cho_solve((chol_Qh, True), jnp.eye(K, dtype=jnp.float64))
        Q_h_inv = (Q_h_inv + Q_h_inv.T) / 2 # Ensure symmetry

        # Construct block diagonal Q_t_inv
        Q_t_inv = jnp.block([
            [Q_f_inv,                   jnp.zeros((K, K), dtype=jnp.float64)],
            [jnp.zeros((K, K), dtype=jnp.float64), Q_h_inv]
        ])
        Q_t_inv = (Q_t_inv + Q_t_inv.T) / 2 # Ensure symmetry

        # 2. Calculate M = info_post + F_t.T @ Q_t_inv @ F_t
        M = info_post + F_t.T @ Q_t_inv @ F_t
        M_jittered = M + jitter * jnp.eye(state_dim, dtype=jnp.float64)

        # 3. Invert M stably
        chol_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)
        M_inv = jax.scipy.linalg.cho_solve((chol_M, True), jnp.eye(state_dim, dtype=jnp.float64))
        M_inv = (M_inv + M_inv.T) / 2 # Ensure symmetry

        # 4. Calculate predicted information: Omega_pred = Q_t_inv - Q_t_inv @ F_t @ M_inv @ F_t.T @ Q_t_inv
        term = Q_t_inv @ F_t @ M_inv @ F_t.T @ Q_t_inv
        predicted_info = Q_t_inv - term
        predicted_info = (predicted_info + predicted_info.T) / 2 # Ensure symmetry

        return predicted_state.reshape(-1, 1), predicted_info

    # _block_coordinate_update_impl is imported from _bellman_optim

    # Internal JAX version of update for Information Filter
    def __update_jax_info(
        self,
        params: DFSVParamsDataclass, # Expect JAX arrays inside
        predicted_state: jnp.ndarray, # Predicted state alpha_{t|t-1} (JAX array)
        predicted_info: jnp.ndarray,  # Predicted information Omega_{t|t-1} (JAX array)
        observation: jnp.ndarray,     # Observation y_t (JAX array)
        # Pass JITted functions required by block_coordinate_update and this method
        block_coord_update_fn: Callable,
        fisher_info_fn: Callable,
        log_posterior_fn: Callable,
        kl_penalty_fn: Callable
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Performs the BIF update step (internal JAX implementation).

        Calculates the updated state alpha_{t|t} and information matrix Omega_{t|t},
        along with the log-likelihood contribution for the current time step,
        based on the predicted state/information and the current observation.

        State Update: Performed by optimizing the posterior mode using `block_coord_update_fn`.
        Information Update: Omega_{t|t} = Omega_{t|t-1} + J_observed, where J_observed is the
                          observed Fisher Information calculated using `fisher_info_fn`.
        Log-Likelihood Contribution: Calculated using the formula from Lange (2024, Eq. 40),
                                   combining a fit term (`log_posterior_fn`) and a KL-type
                                   penalty (`kl_penalty_fn`).

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
            A tuple containing:
                - updated_state: Updated state alpha_{t|t} (JAX array, shape (state_dim, 1)).
                - updated_info: Updated information Omega_{t|t} (JAX array, shape (state_dim, state_dim)).
                - log_lik_contrib: Log-likelihood contribution log p(y_t|F_{t-1}) (JAX scalar).
       """
        K = self.K
        N = self.N
        state_dim = self.state_dim
        lambda_r = params.lambda_r
        sigma2 = params.sigma2 # Assumed 1D JAX array
        jitter = 1e-6 # Jitter for information update

        jax_observation = observation.flatten()

        # --- DEBUG PRINTS (Phase 1.3) ---
        # --- State Update ---
        # jax.debug.print("--- BIF Update Step ---")
        # # Initial guess for optimization is the predicted state
        # jax.debug.print("a_pred (predicted_state): {x}", x=predicted_state.flatten())
        alpha_init_guess = predicted_state.flatten()
        # jax.debug.print("Omega_pred (predicted_info): {x}", x=predicted_info)

        # Run block coordinate update (using the passed JITted function)
        # Note: block_coord_update_fn has h_solver, build_cov_fn, log_post_fn bound
        alpha_updated = block_coord_update_fn(
            lambda_r,                 # Pass individually
            sigma2,                  # Pass individually
            alpha_init_guess,        # Pass individually
            predicted_state.flatten(), # Pass individually
            predicted_info,          # Pass individually
            jax_observation,         # Pass individually
            max_iters=10             # Keyword argument remains
        )

        # --- Information Update ---
        # --- DEBUG PRINTS (Phase 1.4) ---
        # Calculate Observed Fisher Information J_observed = -Hessian(log p(y_t|alpha_t))
        # jax.debug.print("a_updated (alpha_updated): {x}", x=alpha_updated)
        J_observed = fisher_info_fn(lambda_r, sigma2, alpha_updated, observation)
        diff = alpha_updated - predicted_state.flatten()

        # jax.debug.print("diff (a_updated - a_pred): {x}", x=diff)
        # --- Regularize J_observed to ensure PSD ---
        # jax.debug.print("diff_h (h component): {x}", x=diff[K:])
        evals_j, evecs_j = jnp.linalg.eigh(J_observed)
        min_eigenvalue = 1e-8 # Small positive floor
        evals_j_clipped = jnp.maximum(evals_j, min_eigenvalue)
        J_observed_psd = evecs_j @ jnp.diag(evals_j_clipped) @ evecs_j.T
        J_observed_psd = (J_observed_psd + J_observed_psd.T) / 2 # Ensure symmetry
        # --- End Regularization ---


        # Compute updated information matrix Omega_{t|t} = Omega_{t|t-1} + J_observed_psd
        updated_info = predicted_info + J_observed_psd + jitter * jnp.eye(state_dim, dtype=jnp.float64)
        updated_info = (updated_info + updated_info.T) / 2 # Ensure symmetry

        # --- Log-Likelihood Contribution ---
        # Calculate fit term log p(y_t | alpha_{t|t})
        log_lik_fit = log_posterior_fn(lambda_r, sigma2, alpha_updated, jax_observation)

        # Calculate KL-type penalty term (using the passed JITted function)
        kl_penalty = kl_penalty_fn(
            a_pred=predicted_state.flatten(),
            a_updated=alpha_updated,
            Omega_pred=predicted_info,
            Omega_post=updated_info
        )

        # Combine: log p(y_t|F_{t-1}) ≈ log p(y_t|alpha_{t|t}) - KL_penalty
        log_lik_contrib = log_lik_fit - kl_penalty

        # Return results as JAX arrays (reshaped state), keep log_lik as JAX scalar
        # Always return components for potential use in scan
        return alpha_updated.reshape(-1, 1), updated_info, log_lik_contrib

    # --- Public API Methods (NumPy In/Out) ---

    def predict(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass],
        state: np.ndarray,
        info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the BIF prediction step.

        Accepts NumPy arrays for state and information, converts them to JAX,
        calls the internal JITted prediction function, and returns the results
        as NumPy arrays.

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            state: Posterior state estimate alpha_{t-1|t-1} (NumPy array,
                shape (state_dim,) or (state_dim, 1)).
            info: Posterior information matrix Omega_{t-1|t-1} (NumPy array,
                shape (state_dim, state_dim)).

        Returns:
            A tuple containing:
                - predicted_state: Predicted state alpha_{t|t-1} (NumPy array,
                  shape (state_dim, 1)).
                - predicted_info: Predicted information Omega_{t|t-1} (NumPy array,
                  shape (state_dim, state_dim)).
        """
        params_jax = self._process_params(params)
        state_jax = jnp.array(state, dtype=jnp.float64).reshape(-1, 1) # Ensure column vector
        info_jax = jnp.array(info, dtype=jnp.float64)

        pred_state_jax, pred_info_jax = self.predict_jax_info_jit(
            params_jax, state_jax, info_jax
        )

        return np.asarray(pred_state_jax), np.asarray(pred_info_jax)

    def update(
        self,
        params: Union[Dict[str, Any], DFSVParamsDataclass],
        predicted_state: np.ndarray,
        predicted_info: np.ndarray,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs the BIF update step.

        Accepts NumPy arrays for predicted state/info and observation, converts
        them to JAX, calls the internal JITted update function, and returns the
        results as NumPy arrays (state/info) and a float (log-likelihood).

        Args:
            params: Model parameters (Dict or DFSVParamsDataclass).
            predicted_state: Predicted state alpha_{t|t-1} (NumPy array,
                shape (state_dim,) or (state_dim, 1)).
            predicted_info: Predicted information matrix Omega_{t|t-1} (NumPy array,
                shape (state_dim, state_dim)).
            observation: Current observation y_t (NumPy array, shape (N,)).

        Returns:
            A tuple containing:
                - updated_state: Updated state alpha_{t|t} (NumPy array, shape (state_dim, 1)).
                - updated_info: Updated information Omega_{t|t} (NumPy array, shape (state_dim, state_dim)).
                - log_lik_contrib: Log-likelihood contribution log p(y_t|F_{t-1}) (float).
        """
        params_jax = self._process_params(params)
        pred_state_jax = jnp.array(predicted_state, dtype=jnp.float64).reshape(-1, 1)
        pred_info_jax = jnp.array(predicted_info, dtype=jnp.float64)
        obs_jax = jnp.array(observation, dtype=jnp.float64)

        updated_state_jax, updated_info_jax, log_lik_contrib_jax = self.update_jax_info_jit(
            params_jax, pred_state_jax, pred_info_jax, obs_jax
        )

        return np.asarray(updated_state_jax), np.asarray(updated_info_jax), float(log_lik_contrib_jax)


    # --- Filtering Methods (Adapted for Information Filter) ---
    def filter(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Runs the Bellman Information Filter using a standard Python loop.

        Iterates through time steps, calling the public `predict` and `update`
        methods which handle NumPy/JAX conversions internally. Stores results
        internally as JAX arrays and returns them converted to NumPy arrays.

        NOTE: This method uses a Python loop. Prefer `filter_scan` for
              performance, especially with JIT compilation.

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed data with shape (T, N) (NumPy array).

        Returns:
            A tuple containing:
                - filtered_states: Filtered states alpha_{t|t} (NumPy array,
                  shape (T, state_dim)).
                - filtered_infos: Filtered information matrices Omega_{t|t}
                  (NumPy array, shape (T, state_dim, state_dim)).
                - total_log_likelihood: Total log-likelihood (float).
        """
        params_jax = self._process_params(params) # Still useful to process once
        self._setup_jax_functions() # Ensure JIT functions are ready

        T = observations.shape[0]
        state_dim = self.state_dim

        # Initialize storage (JAX arrays)
        filtered_states_jax = jnp.zeros((T, state_dim), dtype=jnp.float64)
        filtered_infos_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        predicted_states_jax = jnp.zeros((T, state_dim, 1), dtype=jnp.float64)
        predicted_infos_jax = jnp.zeros((T, state_dim, state_dim), dtype=jnp.float64)
        log_likelihoods_jax = jnp.zeros(T, dtype=jnp.float64)

        # Initialization (t=0) - Use BIF initialization (returns JAX)
        initial_state_jax, initial_info_jax = self.initialize_state(params_jax)

        # Start loop state with initial values (NumPy for loop logic)
        state_post_prev_np = np.asarray(initial_state_jax)
        info_post_prev_np = np.asarray(initial_info_jax)

        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        # Filtering loop (using public predict/update with NumPy)
        for t in tqdm(range(T), desc="Bellman Information Filtering (Python Loop)"):
            # Predict step (predict state t based on t-1 posterior) -> returns NumPy
            pred_state_t_np, pred_info_t_np = self.predict(
                params_jax, state_post_prev_np, info_post_prev_np
            )

            # Store predicted results (convert back to JAX for internal storage)
            predicted_states_jax = predicted_states_jax.at[t].set(jnp.asarray(pred_state_t_np))
            predicted_infos_jax = predicted_infos_jax.at[t].set(jnp.asarray(pred_info_t_np))

            # Observation for this step (NumPy)
            obs_t_np = observations[t]

            # Update step (update state t using observation t) -> returns NumPy/float
            updated_state_t_np, updated_info_t_np, log_lik_t_float = self.update(
                params_jax, pred_state_t_np, pred_info_t_np, obs_t_np
            )

            # Store filtered results using JAX functional updates (convert back to JAX)
            filtered_states_jax = filtered_states_jax.at[t].set(jnp.asarray(updated_state_t_np).flatten()) # Flatten state before storing
            filtered_infos_jax = filtered_infos_jax.at[t].set(jnp.asarray(updated_info_t_np))
            log_likelihoods_jax = log_likelihoods_jax.at[t].set(jnp.array(log_lik_t_float)) # Store as JAX scalar

            # Update loop state for next iteration (NumPy)
            state_post_prev_np = updated_state_t_np
            info_post_prev_np = updated_info_t_np


        # Store results internally as JAX arrays
        self.filtered_states = filtered_states_jax
        self.filtered_infos = filtered_infos_jax
        self.predicted_states = predicted_states_jax
        self.predicted_infos = predicted_infos_jax
        self.log_likelihoods = log_likelihoods_jax
        self.total_log_likelihood = float(jnp.sum(log_likelihoods_jax)) # Sum JAX array, convert to float

        # Return NumPy arrays by calling getter methods
        return self.get_filtered_states(), self.get_filtered_information_matrices(), self.get_total_log_likelihood()

    def filter_scan(
        self, params: Union[Dict[str, Any], DFSVParamsDataclass], observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, jnp.ndarray]:
        """Runs the Bellman Information Filter using `jax.lax.scan`.

        This method leverages `jax.lax.scan` for potentially faster execution
        compared to the Python loop in `filter`, especially on accelerators.
        It performs the entire filtering loop within JAX's compiled graph.

        Args:
            params: Parameters of the DFSV model (Dict or DFSVParamsDataclass).
            observations: Observed data with shape (T, N) (NumPy array).

        Returns:
            A tuple containing:
                - filtered_states: Filtered states alpha_{t|t} (NumPy array,
                  shape (T, state_dim)).
                - filtered_infos: Filtered information matrices Omega_{t|t}
                  (NumPy array, shape (T, state_dim, state_dim)).
                - total_log_likelihood: Total log-likelihood (JAX scalar).
        """
        params_jax = self._process_params(params) # Ensure correct format (contains JAX arrays)
        self._setup_jax_functions() # Ensure JIT functions are ready
        T = observations.shape[0]

        # Initialization (get initial state/info as JAX arrays)
        initial_state_jax, initial_info_jax = self.initialize_state(params_jax) # Use renamed method
        # Ensure carry types are JAX compatible (float for sum)
        initial_carry = (initial_state_jax, initial_info_jax, jnp.array(0.0, dtype=jnp.float64)) # state, info, log_lik_sum

        # JAX observations
        jax_observations = jnp.array(observations)

        # Define the step function for lax.scan (operates purely on JAX types)
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

        # --- Store results for smoother ---
        # Convert filtered AND predicted information matrices to covariances
        # vmapped_inverter = jit(jax.vmap(self._invert_info_matrix, in_axes=0))
        # filtered_covs_scan = vmapped_inverter(filtered_infos_scan)
        # predicted_covs_scan = vmapped_inverter(predicted_infos_scan) # Compute predicted covs

        # Store NumPy versions for the smoother in the base class attributes
        self.filtered_states = np.asarray(filtered_states_scan.reshape(T, self.state_dim))
        # self.filtered_covs = np.asarray(filtered_covs_scan)
        # Store predicted states (already computed) and predicted covariances (newly computed)
        # Keep predicted states as (T, state_dim, 1) for potential consistency? Or flatten? Let's flatten for now.
        self.predicted_states = np.asarray(predicted_states_scan.reshape(T, self.state_dim))
        # self.predicted_covs = np.asarray(predicted_covs_scan) # Store predicted covs
        self.is_filtered = True # Mark filter as run

        # Return NumPy arrays for states/infos, JAX scalar for loglik by calling getter methods
        # Note: get_filtered_states() now returns the NumPy version stored above
        # Note: get_filtered_information_matrices() returns the NumPy version of filtered_infos_scan
        return self.get_filtered_states(), self.get_filtered_information_matrices(), self.get_total_log_likelihood()

    # --- Smoothing Method ---
    def smooth(self, params: DFSVParamsDataclass) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Rauch-Tung-Striebel (RTS) smoothing for the BIF.

        This method first computes the filtered covariances P_{t|t} by inverting
        the stored filtered information matrices Omega_{t|t}. It then calls the
        base class `smooth` method, which performs the standard RTS recursion
        using the filtered states (alpha_{t|t}) and the computed filtered
        covariances (P_{t|t}).

        Returns:
            A tuple containing:
                - smoothed_states: Smoothed states alpha_{t|T} (NumPy array, shape (T, state_dim)).
                - smoothed_covariances: Smoothed covariances P_{t|T} (NumPy array, shape (T, state_dim, state_dim)).

        Raises:
            RuntimeError: If the filter has not been run (i.e., `filtered_infos` or `filtered_states` is None).
        """
        # Check if filter results (JAX arrays) are available
        if getattr(self, 'filtered_states', None) is None or getattr(self, 'filtered_infos', None) is None:
            raise RuntimeError(
                "Filter must be run successfully (e.g., using filter_scan) "
                "before smoothing."
            )


        # Compute filtered covariances (P_t|t) from information matrices (Omega_t|t)
        # get_filtered_covariances() handles JAX->NumPy conversion and stores in self.filtered_covs
        filtered_covs_np = self.get_filtered_covariances()
        if filtered_covs_np is None:
            raise RuntimeError("Failed to compute filtered covariances needed for smoothing.")

        #Compute predicted covariances (P_t|t-1) from information matrices (Omega_t|t-1)
        predicted_covs_np = self.get_predicted_covariances()
        if predicted_covs_np is None:
            raise RuntimeError("Failed to compute predicted covariances needed for smoothing.")


        # Overwrite attributes temporarily with NumPy versions for the base class call
        self.predicted_covs = predicted_covs_np
        self.filtered_covs = filtered_covs_np # This is already set by get_filtered_covariances
        self.is_filtered = True # Ensure base class knows filter was run

        # Call the base class implementation which expects NumPy arrays and now params
        # Base class smooth returns 3 values: states, covs, lag1_covs
        smoothed_states, smoothed_covs, smoothed_lag1_covs_np = super().smooth(params)

        # Note: self.smoothed_states, self.smoothed_covs, and self.smoothed_lag1_covs are set
        #       by the base class smoother (as NumPy arrays).

        # Return only states and covs to match the method's type hint/docstring
        return smoothed_states, smoothed_covs


    # --- Getter Methods (Adapted for Information Filter) ---
    # Convert internal JAX arrays to NumPy for external use

    def get_filtered_states(self) -> np.ndarray | None:
        """Returns the filtered states alpha_{t|t} as a NumPy array."""
        states_jax = getattr(self, 'filtered_states', None)
        return np.asarray(states_jax) if states_jax is not None else None

    def get_filtered_factors(self) -> np.ndarray | None:
        """Returns the filtered factors f_{t|t} as a NumPy array."""
        states_np = self.get_filtered_states() # Gets NumPy array
        if states_np is not None:
            # Slicing on the NumPy array
            return states_np[:, :self.K]
        return None

    def get_filtered_volatilities(self) -> np.ndarray | None:
        """Returns the filtered log-volatilities h_{t|t} as a NumPy array."""
        states_np = self.get_filtered_states() # Gets NumPy array
        if states_np is not None:
            # Slicing on the NumPy array
            return states_np[:, self.K:]
        return None

    def get_filtered_information_matrices(self) -> np.ndarray | None:
        """Returns the filtered information matrices Omega_{t|t} as NumPy arrays."""
        infos_jax = getattr(self, 'filtered_infos', None)
        return np.asarray(infos_jax) if infos_jax is not None else None

    def get_predicted_states(self) -> np.ndarray | None:
        """Returns the predicted states alpha_{t|t-1} as a NumPy array with shape (T, state_dim)."""
        states_jax = getattr(self, 'predicted_states', None)
        if states_jax is not None:
            states_np = np.asarray(states_jax)
            # Ensure flat vector shape (T, state_dim)
            if states_np.ndim == 3:
                states_np = states_np.reshape(states_np.shape[0], states_np.shape[1])
            return states_np
        return None

    def get_predicted_information_matrices(self) -> np.ndarray | None:
        """Returns the predicted information matrices Omega_{t|t-1} as NumPy arrays."""
        infos_jax = getattr(self, 'predicted_infos', None)
        return np.asarray(infos_jax) if infos_jax is not None else None

    def get_log_likelihoods(self) -> np.ndarray | None:
        """Returns the log-likelihood contributions per step as a NumPy array."""
        lls_jax = getattr(self, 'log_likelihoods', None)
        return np.asarray(lls_jax) if lls_jax is not None else None

    def get_total_log_likelihood(self) -> float | jnp.ndarray | None:
        """Returns the total log-likelihood.

        Returns float if filter() was run, JAX scalar if filter_scan() was run.
        """
        return getattr(self, 'total_log_likelihood', None)

    # --- Methods to derive covariance from information ---

    @eqx.filter_jit
    def _invert_info_matrix(self, info_matrix: jnp.ndarray) -> jnp.ndarray:
        """Stably inverts a single information matrix (Omega -> P).

        Uses Cholesky decomposition for potentially better stability than direct inv.

        Args:
            info_matrix: The information matrix (state_dim, state_dim) to invert.

        Returns:
            The corresponding covariance matrix (state_dim, state_dim).
       """
        jitter = 1e-8 # Consistent jitter
        state_dim = self.state_dim
        info_jittered = info_matrix + jitter * jnp.eye(state_dim, dtype=jnp.float64)
        # Perform Cholesky-based inversion
        chol_info = jax.scipy.linalg.cholesky(info_jittered, lower=True)
        cov_matrix = jax.scipy.linalg.cho_solve((chol_info, True), jnp.eye(state_dim, dtype=jnp.float64))
        # Ensure symmetry
        return (cov_matrix + cov_matrix.T) / 2

    def get_predicted_covariances(self) -> np.ndarray | None:
        """Calculates predicted covariances P_{t|t-1} by inverting Omega_{t|t-1}.

        Note: This performs potentially expensive inversions and is intended for
              analysis after filtering, not during the filter loop itself.

        Returns:
            Predicted state covariances (T, state_dim, state_dim) as NumPy array,
            or None if predicted information matrices are not available.
        """
        pred_infos_jax = getattr(self, 'predicted_infos', None)
        if pred_infos_jax is None:
            return None

        # Use vmap to apply the inversion function across the time dimension
        vmapped_inverter = jit(jax.vmap(self._invert_info_matrix, in_axes=0))
        pred_covs_jax = vmapped_inverter(pred_infos_jax)
        return np.asarray(pred_covs_jax)


    def get_predicted_variances(self) -> np.ndarray | None:
        """Calculates predicted state variances (diagonal of P_{t|t-1}).

        Convenience method calling `get_predicted_covariances`.

        Returns:
            Predicted state variances (T, state_dim) as NumPy array, or None.
        """
        pred_covs_np = self.get_predicted_covariances()
        if pred_covs_np is None:
            return None
        # Extract diagonal elements for each time step
        return np.diagonal(pred_covs_np, axis1=1, axis2=2)

    def get_filtered_covariances(self) -> np.ndarray | None:
        """Calculates filtered covariances P_{t|t} by inverting Omega_{t|t}.

        Stores the result in `self.filtered_covs` as a side effect.

        Note: This performs potentially expensive inversions and is intended for
              analysis after filtering or before smoothing.

        Returns:
            Filtered state covariances (T, state_dim, state_dim) as NumPy array,
            or None if filtered information matrices are not available.
        """
        filtered_infos_jax = getattr(self, 'filtered_infos', None)
        if filtered_infos_jax is None:
            self.filtered_covs = None # Ensure consistency
            return None

        # Use vmap to apply the inversion function across the time dimension
        vmapped_inverter = jit(jax.vmap(self._invert_info_matrix, in_axes=0))
        filtered_covs_jax = vmapped_inverter(filtered_infos_jax)
        self.filtered_covs = np.asarray(filtered_covs_jax) # Store as NumPy
        return self.filtered_covs

    def get_filtered_variances(self) -> np.ndarray | None:
        """Calculates filtered state variances (diagonal of P_{t|t}).

        Convenience method calling `get_filtered_covariances`.

        Returns:
            Filtered state variances (T, state_dim) as NumPy array, or None.
        """
        filtered_covs_np = self.get_filtered_covariances() # This populates self.filtered_covs
        if filtered_covs_np is None:
            return None
        # Extract diagonal elements for each time step
        return np.diagonal(filtered_covs_np, axis1=1, axis2=2)


    # --- Likelihood Calculation Methods (Adapted for BIF) ---

    def log_likelihood_wrt_params(
        self, params_dict: Dict[str, Any], observations: np.ndarray
    ) -> jnp.ndarray:
        """Calculates the total BIF pseudo log-likelihood with respect to parameters.

        Public API for likelihood calculation. Handles parameter processing and
        calls `filter_scan` internally.

        Args:
            params_dict: Dictionary containing the model parameters.
            observations: Observed data (NumPy array, shape (T, N)).

        Returns:
            Total pseudo log-likelihood (JAX scalar). Returns -inf if errors occur.
        """
        try:
            # Convert dict to dataclass, ensuring N and K are correct
            params_jax = self._process_params(params_dict)
            # filter_scan returns (filtered_states_np, filtered_infos_np, total_log_lik_jax)
            _, _, total_log_lik = self.filter_scan(params_jax, observations)
            # Handle potential NaN/Inf values from filtering (using JAX functions)
            # Also handle extremely large positive values which can occur due to numerical issues
            # with the BIF penalty term
            is_invalid = jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik) | (total_log_lik > 1e10)
            return jnp.where(is_invalid, -jnp.inf, total_log_lik)
        except (ValueError, TypeError) as e: # Catch only pre-JAX processing errors
            # Handle errors during parameter processing or filtering
            print(f"Warning: Error calculating BIF likelihood: {e}")
            # Return JAX representation of -inf
            return jnp.array(-jnp.inf, dtype=jnp.float64)


    def _log_likelihood_wrt_params_impl(
        self, params: DFSVParamsDataclass, observations: jnp.ndarray
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Internal JAX implementation for BIF log-likelihood using scan.

        Designed to be JIT-compiled. Assumes inputs are JAX arrays. Can optionally
        return the sum of fit and penalty terms separately.

        Args:
            params: DFSVParamsDataclass instance with JAX arrays.
            observations: Observed returns (T, N) as JAX array.
            return_components: If True, return (total_lik, fit_sum, penalty_sum).
                               Otherwise, return only total_lik.

        Returns:
            Total pseudo log-likelihood (JAX scalar), or a tuple containing
            (total_lik, fit_sum, penalty_sum) if return_components is True.
            Returns -inf components if NaN/Inf encountered.
        """
        T = observations.shape[0]

        # Initialization (use BIF JAX arrays)
        initial_state_jax, initial_info_jax = self.initialize_state(params)
        # Ensure carry types are JAX compatible: state, info, total_lik_sum
        initial_carry = (initial_state_jax, initial_info_jax, jnp.array(0.0))

        # Define the step function for lax.scan (operates purely on JAX types)
        def filter_step(carry, obs_t):
            state_t_minus_1_jax, info_t_minus_1_jax, total_lik_sum= carry
            # Predict step -> returns JAX arrays
            pred_state_t_jax, pred_info_t_jax = self.predict_jax_info_jit(
                params, state_t_minus_1_jax, info_t_minus_1_jax
            )
            # Update step -> returns JAX arrays (state, info, total_lik
            updated_state_t_jax, updated_info_t_jax, log_lik_t_jax = self.update_jax_info_jit(
                params, pred_state_t_jax, pred_info_t_jax, obs_t
            )
            # Prepare carry for next step
            next_carry = (updated_state_t_jax, updated_info_t_jax,
                          total_lik_sum + log_lik_t_jax) # Note: penalty is subtracted later in update, so we sum it here
            # We only need the carry for the final likelihood(s)
            return next_carry, None # Don't store intermediate results

        # Run the scan
        final_carry, _ = jax.lax.scan(filter_step, initial_carry, observations)

        total_log_lik = final_carry[2] # JAX scalars

        # Replace NaN/Inf with -inf for optimization stability
        # Note: penalty_sum is the sum of KL terms, which are subtracted from fit_sum.
        # A large positive penalty sum means a large negative contribution to total likelihood.
        # Also handle extremely large positive values which can occur due to numerical issues
        is_invalid = jnp.isnan(total_log_lik) | jnp.isinf(total_log_lik) | (total_log_lik > 1e10)
        safe_total_log_lik = jnp.where(is_invalid, -jnp.inf, total_log_lik)

        return safe_total_log_lik


    def jit_log_likelihood_wrt_params(self) -> Callable:
        """Returns a JIT-compiled version of the BIF log-likelihood function w.r.t parameters.

        The returned function accepts an optional `return_components` argument (default False).
        If True, it returns (total_log_lik, fit_sum, penalty_sum).
        If False, it returns only total_log_lik.

        Returns:
            A JIT-compiled function `likelihood_fn(params, observations, return_components=False)`.
        """
        # Ensure JIT functions are set up before returning the JITted likelihood function
        self._setup_jax_functions()
        # JIT the implementation method directly. return_components is a runtime arg.
        return eqx.filter_jit(self._log_likelihood_wrt_params_impl)

    # _get_transition_matrix is inherited from DFSVFilter base class


