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

from functions.filters import DFSVFilter
# Update imports to use models.dfsv instead
from models.dfsv import DFSV_params, DFSVParamsDataclass
# We can keep the old imports for backward compatibility
import functions.jax_params  # For backward compatibility


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
    def _process_params(self, params: Union[DFSV_params, Dict[str, Any], DFSVParamsDataclass]) -> Union[DFSV_params, DFSVParamsDataclass]:
        """
        Convert different parameter formats to a consistent format.
        
        Args:
            params: Parameters in DFSV_params, DFSVParamsDataclass, or dictionary format
            
        Returns:
            Union[DFSV_params, DFSVParamsDataclass]: Parameters in a consistent format
        """
        if isinstance(params, dict):
            # Convert dictionary to DFSVParamsDataclass
            N = params.get('N', self.N)  # Use self.N if not in dict
            K = params.get('K', self.K)  # Use self.K if not in dict
            
            # Convert to DFSVParamsDataclass
            return DFSVParamsDataclass.from_dict(params, N, K)
        
        # If it's already a DFSV_params or DFSVParamsDataclass, return it
        return params

    def _setup_jax_functions(self):
        """
        Set up JIT-compiled JAX functions for efficiency.
        """
        # Build static versions of methods that don't reference self directly

        # Create build_covariance as a static function and JIT it
        self.build_covariance_jit = jit(self._build_covariance_impl)

        # Create fisher_information as a JIT-compiled function
        self.fisher_information_jit = jit(self._fisher_information_impl)

        # Create log_posterior as a JIT-compiled function
        self.log_posterior_jit = jit(self._log_posterior_impl)

        # Create KL penalty as a jit-function
        self.kl_penalty = jit(self._kl_penalty_impl)
        # Create block_coordinate_update
        self.block_coordinate_update_jit = jit(
            self._block_coordinate_update_impl,
            static_argnums=(6,),  # max_iters is a static argument
        )
        # Create solver
        # self.solver_h = jaxopt.LBFGS(
        #     fun=self.neg_log_post_h,
        #     maxiter=10,
        #     tol=1e-4,
        # )
        # Try to precompile
        try:
            self._precompile_jax_functions()
            print("JAX functions successfully precompiled")
        except Exception as e:
            print(f"Warning: JAX precompilation failed: {e}")
            print("Functions will be compiled during first filter run")

    def _build_covariance_impl(
        self, lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Static implementation of build_covariance.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            exp_h (jnp.ndarray): Exponentiated log-volatilities.
            sigma2 (jnp.ndarray): Idiosyncratic variances.

        Returns:
            jnp.ndarray: Covariance matrix.
        """
        K = lambda_r.shape[1]
        N = lambda_r.shape[0]

        if sigma2.ndim == 1:
            sigma2 = jnp.diag(sigma2)

        Sigma_f = jnp.diag(exp_h.flatten())
        lambda_r = lambda_r.reshape(N, K)
        A = lambda_r @ Sigma_f @ lambda_r.T
        A += jnp.diag(jnp.diag(sigma2)) + 1e-6 * jnp.eye(N)
        A = 0.5 * (A + A.T)
        return A

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

    def _fisher_information_impl(
        self, lambda_r: jnp.ndarray, sigma2: jnp.ndarray, alpha: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Static implementation of fisher_information.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): State vector [f, h].

        Returns:
            jnp.ndarray: Fisher information matrix.
        """
        K = self.K

        # f = alpha[:K] #not used
        h = alpha[K:]
        exp_h = jnp.exp(h)

        # Build covariance
        A = self.build_covariance(lambda_r, exp_h, sigma2)
        L = jax.scipy.linalg.cho_factor(A, lower=True)

        # Helper function to compute A^-1 @ x using Cholesky
        def A_inv(x):
            return jax.scipy.linalg.cho_solve(L, x)

        # ----------------------------------------------------------------
        # 2) Mean part: mu(alpha) = Lambda f
        #
        #    derivative w.r.t. f is Lambda_r,
        #    derivative w.r.t. h is 0.
        #
        #    So the (f,f) block is Lambda_r.T @ A_inv(Lambda_r).
        #    The (f,h) block is zero.
        # ----------------------------------------------------------------
        A_inv_lambda_r = A_inv(lambda_r)  # shape (N, K)
        mean_part = lambda_r.T @ A_inv_lambda_r  # shape (K, K)
        I_fh = jnp.zeros((K, K))  # shape (K, K), for cross terms

        # ----------------------------------------------------------------
        # 3) Covariance part:
        #    partial Sigma / partial h_k = e^{h_k} * outer(lambda_r[:,k], lambda_r[:,k])
        #
        #    Then the (h,h) block's (i,j)-element is
        #        0.5 * trace[ A_inv(dSigma_i) @ A_inv(dSigma_j) ].
        # ----------------------------------------------------------------
        # Build all partial-Sigma in one shot
        def partial_sigma(k):
            # dSigma_k = e^{h_k} * (lambda_r[:,k] outer lambda_r[:,k])
            lambda_r_k = jnp.take(lambda_r, k, axis=1)
            exp_h_k = jnp.take(exp_h, k)
            return exp_h_k * jnp.outer(lambda_r_k, lambda_r_k)

        # We gather partial derivatives in a list
        dSigmas = jax.vmap(partial_sigma)(jnp.arange(K))  # shape (K, N, N)

        # Apply A^{-1} to each partial derivative
        B = jax.vmap(A_inv)(dSigmas)  # shape (K, N, N)

        # Compute pairwise traces in a single einsum.
        # "kij,lji->kl" means sum_{i,j} over B[k,i,j]*B[l,j,i].
        # This yields a (K,K) result.
        cov_part_block = 0.5 * jnp.einsum("kij, lji -> kl", B, B)  # shape (K, K)

        # Now assemble the full 2K x 2K matrix
        top_left = mean_part  # (f,f)
        bottom_right = cov_part_block
        I_fisher = jnp.block([[top_left, I_fh], [I_fh, bottom_right]])
        return I_fisher

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

    def _log_posterior_impl(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        alpha: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> float:
        """
        Static implementation of log_posterior.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): State vector [f, h].
            observation (jnp.ndarray): Observation vector.

        Returns:
            float: Log posterior value.
        """
        # Unpack dimensions and flatten inputs
        K = self.K
        alpha = alpha.flatten()
        observation = observation.flatten()

        # Split state
        f = alpha[:K]
        log_vols = alpha[K:]

        # Innovation and covariance
        pred_obs = lambda_r @ f
        innovation = observation - pred_obs
        exp_log_vols = jnp.exp(log_vols)
        A = self.build_covariance(lambda_r, exp_log_vols, sigma2)

        # Compute log-likelihood with Cholesky decomposition
        L = jnp.linalg.cholesky(A)

        # Compute quadratic form via triangular solve
        alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
        quad_form = jnp.sum(alpha_vec**2)

        # Safe log determinant
        diag_L= jnp.diag(L)
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L, 1e-10)))

        # log-likelihood
        N_ = observation.shape[0]
        log_lik = -0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

        return log_lik

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
        max_iters: int = 5,
    ) -> jnp.ndarray:
        """
        Static implementation of block_coordinate_update.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            alpha (jnp.ndarray): Initial state vector [f, h].
            pred_state (jnp.ndarray): Predicted state vector.
            I_pred (jnp.ndarray): Predicted precision matrix.
            observation (jnp.ndarray): Observation vector.
            max_iters (int): Maximum number of iterations.

        Returns:
            jnp.ndarray: Updated state vector.
        """
        # Unpack dimensions and flatten inputs
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
        I_h = I_pred[K:, K:]
        I_fh = I_pred[:K, K:]
        I_hf = I_fh.T

        def update_factors(
            log_volatility: jnp.ndarray,
            factors_pred: jnp.ndarray,
            log_vols_pred: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Solve for factors given log-volatility is fixed.

            Args:
                log_volatility (jnp.ndarray): Log-volatility values.
                factors_pred (jnp.ndarray): Predicted factor values.
                log_vols_pred (jnp.ndarray): Predicted log-volatility values.

            Returns:
                jnp.ndarray: Updated factor values.
            """
            A = self.build_covariance(lambda_r, jnp.exp(log_volatility), sigma2)
            L = jax.scipy.linalg.cho_factor(A, lower=True)

            def A_inv(x):
                return jax.scipy.linalg.cho_solve(L, x)

            lhs_mat = jnp.dot(lambda_r.T, A_inv(lambda_r)) + I_f+1e-8 * jnp.eye(I_f.shape[0])
            rhs_vec = (
                jnp.dot(lambda_r.T, A_inv(observation))
                + jnp.dot(I_f, factors_pred)
                + jnp.dot(I_fh, (-log_vols_pred + log_volatility))
            )

            return jnp.linalg.solve(lhs_mat, rhs_vec)

        def update_h_bfgs(h_init: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            """
            Minimize J(h) = neg_log_post_h(h, f, ...) w.r.t. h using BFGS.

            Args:
                h_init (jnp.ndarray): Initial log-volatility values.
                f (jnp.ndarray): Factor values.

            Returns:
                jnp.ndarray: Updated log-volatility values.
            """
            # Create a wrapper function that matches the expected signature fn(y, args)
            def objective_fn(h, args):
                lambda_r, sigma2, factors, predicted_state, I_pred, observation = args
                return self.neg_log_post_h(
                    log_vols=h,
                    lambda_r=lambda_r,
                    sigma2=sigma2,
                    factors=factors,
                    predicted_state=predicted_state,
                    I_pred=I_pred,
                    observation=observation
                )
        
            # Instantiate Optimistix BFGS solver
            solver = optx.BFGS(rtol=1e-4, atol=1e-6) 
            
            # Prepare arguments for the objective function
            args = (lambda_r, sigma2, f, pred_state, I_pred, observation)
            
            # Run the minimization
            sol = optx.minimise(
                fn=objective_fn,
                solver=solver,
                y0=h_init,
                args=args,
                options={},
                max_steps=max_iters,
                throw=False
            )

            # Check if optimization was successful
            def true_fun(result_params):
                return result_params

            def false_fun(initial_params):
                return initial_params

            # Use sol.result instead of checking status directly
            is_successful = bool(sol.result)

            return jax.lax.cond(
                is_successful,
                true_fun,
                false_fun,
                sol.value
            )

        # Update loop
        def body_fn(
            _, carry: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Single iteration of the block-coordinate update.
            `carry` is (f, h).

            Args:
                _ (int): Loop index.
                carry (Tuple[jnp.ndarray, jnp.ndarray]): Current state (factors, log-vols).

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: Updated state (factors, log-vols).
            """
            f_current, h_current = carry
            f_new = update_factors(h_current, factors_pred, log_vols_pred)
            h_new = update_h_bfgs(h_current, f_new)
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
            max_iters (int): Maximum number of iterations.

        Returns:
            jnp.ndarray: Updated state vector.
        """
        return self.block_coordinate_update_jit(
            lambda_r, sigma2, alpha, pred_alpha, I_pred, observation, max_iters
        )

    def _precompile_jax_functions(self):
        """Precompile JIT functions with dummy data"""
        # Create dummy data
        K = self.K
        N = self.N

        # Create dummy arrays of appropriate shapes
        dummy_lambda_r = jnp.ones((N, K))
        dummy_sigma2 = jnp.ones(N)
        dummy_exp_h = jnp.ones(K)
        dummy_alpha = jnp.zeros(2 * K)
        dummy_pred_alpha = jnp.zeros(2 * K)
        dummy_I_pred = jnp.eye(2 * K)
        dummy_observation = jnp.zeros(N)

        # Create both standard and PyTree parameter versions
        dummy_params = DFSV_params(
            N=N,
            K=K,
            lambda_r=dummy_lambda_r,
            Phi_f=jnp.eye(K),
            Phi_h=jnp.eye(K),
            mu=jnp.zeros(K),
            sigma2=dummy_sigma2,
            Q_h=jnp.eye(K),
        )

        # Warm up individual functions
        _ = self.build_covariance(dummy_lambda_r, dummy_exp_h, dummy_sigma2)
        _ = self.fisher_information(dummy_lambda_r, dummy_sigma2, dummy_alpha)
        _ = self.log_posterior(
            dummy_lambda_r, dummy_sigma2, dummy_alpha, dummy_observation
        )
        _ = self.block_coordinate_update(
            dummy_lambda_r,
            dummy_sigma2,
            dummy_alpha,
            dummy_pred_alpha,
            dummy_I_pred,
            dummy_observation,
            max_iters=2,
        )

        # Also precompile functions that accept pytree params
        # _ = self.predict(dummy_params, dummy_alpha, jnp.eye(2 * K))

    @partial(jit, static_argnums=(0,))  # Mark self as static argument
    def neg_log_post_h(
        self,
        log_vols: jnp.ndarray,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        factors: jnp.ndarray,
        predicted_state: jnp.ndarray,
        I_pred: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> float:
        """
        Negative log-likelihood + prior penalty wrt h (with f fixed).
        J(h) = 0.5[ log det A(h) + (y - Lambda f)^T A(h)^-1 (y - Lambda f ) ]
            + 0.5[ (f - f_pred), (h - h_pred) ]^T I_pred [ (f - f_pred), (h - h_pred) ].

        Args:
            log_vols (jnp.ndarray): Log-volatilities.
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            factors (jnp.ndarray): Factor values.
            predicted_state (jnp.ndarray): Predicted state vector.
            I_pred (jnp.ndarray): Predicted precision matrix.
            observation (jnp.ndarray): Observation vector.

        Returns:
            float: Negative log-posterior value.
        """
        # Unpack dimensions and flatten inputs
        predicted_state = predicted_state.flatten()
        observation = observation.flatten()
        alpha = jnp.concatenate([factors, log_vols]).flatten()

        # Innovation and covariance
        pred_obs = lambda_r @ factors
        innovation = observation - pred_obs
        exp_log_vols = jnp.exp(log_vols)
        A = self.build_covariance(lambda_r, exp_log_vols, sigma2)

        L = jnp.linalg.cholesky(A)

        # Compute quadratic form via triangular solve
        alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
        quad_form = jnp.sum(alpha_vec**2)

        # Safe log determinant
        diag_L= jnp.diag(L)
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L, 1e-10)))

        # Negative log-likelihood
        N_ = observation.shape[0]
        neg_log_lik = 0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

        # Prior penalty
        state_diff = alpha - predicted_state
        penalty = 0.5 * (state_diff @ (I_pred @ state_diff))

        # Return negative log-posterior
        return neg_log_lik + penalty

    def _kl_penalty_impl(self, a_pred, a_updated, I_pred, I_updated):
        """
        Compute KL penalty for the updated state. This is used for the approximate MLE for the static parameters.

        Args:
            a_updated (jnp.ndarray): Updated state vector.
            a_pred (jnp.ndarray): Predicted state vector.
            I_updated (jnp.ndarray): Updated precision matrix.
            I_pred (jnp.ndarray): Predicted precision matrix.

        Returns:
            jnp.ndarray: KL penalty value.
        """
        # Log-determinant
        logdet_I = logdet_I = 0.5 * (
            jnp.linalg.slogdet(I_updated)[1] - jnp.linalg.slogdet(I_pred)[1]
        )
        # Quadratic penalty
        diff = a_updated.flatten() - a_pred.flatten()
        quad_penalty = 0.5 * (jnp.dot(diff, jnp.dot(I_pred, diff)))
        return logdet_I + quad_penalty

    @jit
    def obj_and_grad_fn(
        self,
        lambda_r: jnp.ndarray,
        sigma2: jnp.ndarray,
        x: jnp.ndarray,
        predicted_state: jnp.ndarray,
        I_pred: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> Tuple[float, jnp.ndarray]:
        """
        Combined objective and gradient function for optimization.

        Args:
            lambda_r (jnp.ndarray): Factor loading matrix.
            sigma2 (jnp.ndarray): Idiosyncratic variances.
            x (jnp.ndarray): Current state estimate [f, log_vols].
            predicted_state (jnp.ndarray): Predicted state.
            I_pred (jnp.ndarray): Predicted precision matrix.
            observation (jnp.ndarray): Observation vector.

        Returns:
            Tuple[float, jnp.ndarray]: Objective value and gradient.
        """
        # Unpack dimensions and flatten inputs
        K = lambda_r.shape[1]
        N_ = lambda_r.shape[0]

        x = x.flatten()
        predicted_state = predicted_state.flatten()
        observation = observation.flatten()

        # Split state
        f = x[:K]
        h = x[K:]
        exp_h = jnp.exp(h)

        # Innovation and covariance
        pred_obs = lambda_r @ f
        innovation = observation - pred_obs

        # Build covariance matrix
        A = self.build_covariance(lambda_r, exp_h, sigma2)

        # Compute objective and intermediate values for gradient using Cholesky
        L = jnp.linalg.cholesky(A)
        alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
        quad_form = jnp.sum(alpha_vec**2)
        
        diag_L= jnp.diag(L)
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.maximum(diag_L, 1e-10)))

        # Negative log-likelihood
        neg_log_lik = 0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

        # Prior penalty
        state_diff = x - predicted_state
        penalty = 0.5 * (state_diff @ (I_pred @ state_diff))

        # Total objective
        obj = neg_log_lik + penalty

        # Compute gradient components using the already computed Cholesky factor
        A_inv_innovation = jax.scipy.linalg.cho_solve((L, True), innovation)
        A_inv_lambda = jax.scipy.linalg.cho_solve((L, True), lambda_r)

        # Gradient w.r.t factors
        grad_f = -lambda_r.T @ A_inv_innovation

        # Gradient w.r.t log-volatilities
        term1 = jnp.diag(lambda_r.T @ A_inv_lambda)
        proj = lambda_r.T @ A_inv_innovation
        term2 = proj**2
        grad_h = 0.5 * exp_h * (term1 - term2)

        # Add prior penalty gradient
        penalty_grad = I_pred @ state_diff

        # Combine gradients
        grad = jnp.concatenate([grad_f, grad_h]) + penalty_grad

        return obj, grad.flatten()

    def initialize_state(
        self, params: Union[DFSV_params, Dict[str, Any], DFSVParamsDataclass]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state and covariance.

        Args:
            params: Model parameters in any supported format (DFSV_params, Dictionary, or DFSVParamsDataclass).

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
        params: Union[DFSV_params, Dict[str, Any], DFSVParamsDataclass],
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

        return predicted_state, predicted_cov

    def update(
        self,
        params: DFSV_params,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the Bellman update step using JAX-based optimization.

        Args:
            params: Parameters of the DFSV model (either DFSV_params or DFSVParamsPytree).
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
            max_iters=5,
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

        return updated_state, updated_cov, log_likelihood

    def _get_transition_matrix(
        self, params: DFSV_params,
    ) -> jnp.ndarray:
        """
        Get the state transition matrix for the DFSV model.

        Args:
            params: Parameters of the DFSV model (either DFSV_params or DFSVParamsPytree).

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
        self, params: DFSV_params, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter on the provided data with JAX optimization.

        This implementation replaces the base class version with a more JAX-efficient
        approach that minimizes host-device transfers and leverages JAX's functional
        programming model.

        Args:
            params: Parameters of the DFSV model (either DFSV_params).
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
        self, params: DFSV_params, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter using JAX's scan operation for maximum efficiency.

        This implementation uses JAX's scan operation for efficient filtering without a
        progress bar. Ideal for production use or when performance is critical.

        Args:
            params: Parameters of the DFSV model (either DFSV_params or DFSVParamsPytree).
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
        pytree_params: Union[DFSV_params, Dict[str, Any], DFSVParamsDataclass],
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
        pytree_params: DFSV_params,
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
        pytree_params: DFSV_params,
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
