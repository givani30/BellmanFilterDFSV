import jax.scipy.optimize
import numpy as np
from typing import Tuple
from functions.filters import DFSVFilter
from functions.simulation import DFSV_params
import jax
import jax.numpy as jnp
from jax import hessian, jit
import jaxopt
from functools import partial


class DFSVBellmanFilter_BlockDiag(DFSVFilter):
    """
    Bellman Filter implementation for DFSV models using Block Diagonal optimization.

    This class implements a Bellman filter for state estimation in DFSV models,
    using dynamic programming principles to recursively compute the optimal
    state estimates and covariances. JAX is used for automatic differentiation,
    eliminating the need to manually compute gradients and Hessians.
    """

    def __init__(self, params: DFSV_params):
        """
        Initialize the JAX-based Bellman Filter with DFSV model parameters.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        """
        super().__init__(params)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Convert model parameters to JAX arrays and setup computation functions
        self._setup_jax_params()
        self._compile_jax_functions()

    def _setup_jax_params(self):
        """
        Convert numpy parameters to JAX arrays for use in JAX functions.
        """
        # Convert model parameters to JAX arrays
        self.jax_lambda_r = jnp.array(self.params.lambda_r)

        # Handle sigma2 - ensure it's properly formatted
        if np.isscalar(self.params.sigma2) or (
            isinstance(self.params.sigma2, np.ndarray) and self.params.sigma2.ndim == 1
        ):
            self.jax_sigma2 = jnp.array(self.params.sigma2)
        else:
            self.jax_sigma2 = jnp.array(self.params.sigma2)

        # Handle mu - ensure it has the right shape
        if self.params.mu.ndim == 1:
            self.jax_mu = jnp.array(self.params.mu.reshape(-1, 1))
        else:
            self.jax_mu = jnp.array(self.params.mu)

        # Other model parameters
        self.jax_Phi_f = jnp.array(self.params.Phi_f)
        self.jax_Phi_h = jnp.array(self.params.Phi_h)
        self.jax_Q_h = jnp.array(self.params.Q_h)

    def _compile_jax_functions(self):
        """
        JIT-compile JAX functions for efficiency.
        """

        # Build covariance matrix function
        def build_covariance(lambda_r, exp_h, sigma2):
            """
            Build observation covariance A = Lambda diag(exp_h) Lambda^T + diag(sigma2).

            Parameters:
            lambda_r: (N,K) - factor loadings
            exp_h: (K,) - exponentiated log-volatilities
            sigma2: (N,) - idiosyncratic variances

            Returns:
            A: (N,N) - observation covariance matrix
            """
            Sigma_f = jnp.diag(exp_h)  # (K,K)
            A = lambda_r @ Sigma_f @ lambda_r.T  # (N,N)
            A += jnp.diag(sigma2) + 1e-8 * jnp.eye(
                lambda_r.shape[0]
            )  # Add regularization
            A = 0.5 * (A + A.T)  # Ensure symmetry
            return A

        @jit
        def neg_log_post_h(log_vols, factors, predicted_state, I_pred, observation):
            """
            Negative log-likelihood + prior penalty wrt h (with f fixed).
            J(h) = 0.5[ log det A(h) + (y - Lambda f)^T A(h)^-1 (y - Lambda f ) ]
                + 0.5[ (f - f_pred), (h - h_pred) ]^T I_pred [ (f - f_pred), (h - h_pred) ].
            """
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
            K = lambda_r.shape[1]
            predicted_state = predicted_state.flatten()
            observation = observation.flatten()
            alpha = jnp.concatenate([factors, log_vols]).flatten()
            # Innovation and covariance
            pred_obs = lambda_r @ factors
            innovation = observation - pred_obs
            exp_log_vols = jnp.exp(log_vols)
            A = build_covariance(lambda_r, exp_log_vols, sigma2)

            # Compute negative log-likelihood with Cholesky decomposition
            try:
                L = jnp.linalg.cholesky(A)

                # Compute quadratic form via triangular solve
                alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
                quad_form = jnp.sum(alpha_vec**2)

                # Log determinant
                logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

                # Negative log-likelihood
                N_ = observation.shape[0]
                neg_log_lik = 0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

                # Prior penalty
                state_diff = alpha - predicted_state
                penalty = 0.5 * (state_diff @ (I_pred @ state_diff))

                # Return negative log-posterior
                return neg_log_lik + penalty

            except:
                # In case of Cholesky failure, return a large value
                return jnp.array(1e10)

        self.solver_h = jaxopt.LBFGS(
            fun=neg_log_post_h,
            value_and_grad=False,
            # maxiter=100,
            verbose=False,
        )

        @partial(jit, static_argnames=["max_iters"])
        def block_coordinate_update(
            alpha, pred_state, I_pred, observation, max_iters=10
        ):
            """
            Perform block-coordinate updates on [f, h], accounting for cross terms
            in the predicted precision matrix I_pred.

            Parameters
            ----------
            f_init, h_init : jnp.ndarray, shape (K,)
                Initial guesses for factors and log-vols. Often set to (f_pred, h_pred).
            f_pred, h_pred : jnp.ndarray, shape (K,)
                Predicted means (the prior's center).
            I_pred : jnp.ndarray, shape ((2K), (2K))
                Predicted precision for [f, h]. Partitioned as:
                    [ I_ff   I_fh ]
                    [ I_hf   I_hh ]
                each block is KxK.
            y_obs : jnp.ndarray, shape (N,)
                Observation vector at time t.
            lambda_r : jnp.ndarray, shape (N,K)
            sigma2   : jnp.ndarray, shape (N,)
            max_iters : int
                Number of coordinate iterations.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, float]
                Updated state, updated covariance, and log-likelihood contribution
            """
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
            K = lambda_r.shape[1]
            N_ = lambda_r.shape[0]
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

            def update_factors(log_volatility):
                """
                Solve for factors given log-volatility is fixed.

                We set gradient wrt f of [0.5*(y - Lambda f)^T A^-1 (y - Lambda f)
                + 0.5*(f - f_pred, h - h_pred)^T I_pred (f - f_pred, h - h_pred)] = 0.

                 => (Lambda^T A^-1 Lambda + I_ff) f = Lambda^T A^-1 y + I_ff f_pred
                                            + I_fh * (h_pred - h).

                """
                A = build_covariance(lambda_r, jnp.exp(log_volatility), sigma2)
                L = jax.scipy.linalg.cho_factor(A, lower=True)

                # Helper function to compute A^-1 @ x using Cholesky
                def A_inv(x):
                    return jax.scipy.linalg.cho_solve(L, x)

                # Left-hand side:
                #    (Lambda^T A^-1 Lambda + I_ff)
                lhs_mat = jnp.dot(lambda_r.T, A_inv(lambda_r)) + I_f  # shape (K,K)
                # Right-hand side:
                #    Lambda^T A^-1 y + I_ff f_pred + I_fh (h_pred - h)

                Lambda_inv_y = jnp.dot(lambda_r.T, A_inv(observation))  # shape (K,)
                rhs_vec = (
                    Lambda_inv_y
                    + jnp.dot(I_f, factors_pred)
                    + jnp.dot(I_fh, (-log_vols_pred + log_volatility))
                )

                # Solve linear system
                factors_new = jnp.linalg.solve(lhs_mat, rhs_vec)

                return factors_new

            # Update one step of h given f is fixed.
            # def update_log_volatility(factors, log_volatility):
            #     grad_h = jax.grad(neg_log_post_h)(factors, log_volatility)
            #     hess_h = jax.hessian(neg_log_post_h)(
            #         factors, log_volatility
            #     )  # shape (K,K)
            #     # One Newton step: h_new = h - H^-1 grad
            #     # (in practice, you might do line-search or damping)
            #     h_new = log_volatility - jnp.linalg.solve(hess_h, grad_h)
            #     return h_new
            #     # return log_volatility
            #     # Bellman objective function (negative log-posterior)

            def update_h_bfgs(h_init, f):
                """
                Minimize J(h) = neg_log_post_h(h, f, ...) w.r.t. h using BFGS.
                We'll do a short run with jax.scipy.optimize.minimize.
                """

                # The objective function
                result = self.solver_h.run(
                    init_params=h_init,
                    factors=f,
                    predicted_state=pred_state,
                    I_pred=I_pred,
                    observation=observation,
                )
                # Return the optimized h
                return result.params

            # Update loop
            f, h = factors_guess, log_vols_guess

            for _ in range(max_iters):
                f = update_factors(h)
                # print("f", f)
                h = update_h_bfgs(h, f)
                # print("h", h)
            # Return updated state
            return jnp.concatenate([f, h])

        # log-posterior without the penalty term (used for the hessian)
        @jax.jit
        def log_posterior(alpha, observation):
            """
            JAX implementation of the Bellman objective function (negative log-posterior).

            Parameters:
            alpha: (2K,) - current state estimate [f, log_vols]
            observation: (N,) - observation vector
            Returns:
            Negative log posterior value (scalar)
            """
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
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
            A = build_covariance(lambda_r, exp_log_vols, sigma2)

            # Compute negative log-likelihood with Cholesky decomposition
            L = jnp.linalg.cholesky(A)

            # Compute quadratic form via triangular solve
            alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
            quad_form = jnp.sum(alpha_vec**2)

            # Log determinant
            logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            # log-likelihood
            N_ = observation.shape[0]
            log_lik = -0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

            # Return negative log-posterior
            return -log_lik

        # Combined objective and gradient function for optimization
        @jax.jit
        def obj_and_grad_fn(x, predicted_state, I_pred, observation):
            """Combined objective and gradient function for optimization that computes both in one pass."""
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
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
            Sigma_f = jnp.diag(exp_h)
            A = lambda_r @ Sigma_f @ lambda_r.T
            A += jnp.diag(sigma2) + 1e-8 * jnp.eye(N_)
            A = 0.5 * (A + A.T)

            # Compute objective and intermediate values for gradient using Cholesky
            try:
                L = jnp.linalg.cholesky(A)
                alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
                quad_form = jnp.sum(alpha_vec**2)
                logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

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

            except:
                # In case of Cholesky failure, return a large value and zero gradient
                return jnp.array(1e10), jnp.zeros_like(x)

        # Store the compiled functions
        self.obj_and_grad_fn = obj_and_grad_fn
        self.jax_bellman_objective = jit(
            lambda x, p, i, o: obj_and_grad_fn(x, p, i, o)[0]
        )
        self.jax_gradient = jit(lambda x, p, i, o: obj_and_grad_fn(x, p, i, o)[1])
        self.jax_hessian = jit(hessian(log_posterior, argnums=0))
        self.log_posterior = log_posterior

        self.block_coordinate_update = block_coordinate_update

    def initialize_state(self, y):
        """
        Initialize state and covariance by calling parent method.

        Parameters
        ----------
        y : np.ndarray
            Observed data

        Returns
        -------
        tuple
            Initial state and covariance
        """
        return super().initialize_state(y)

    def predict(
        self, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Bellman prediction step using Schur complement for information matrix.

        Parameters
        ----------
        state : np.ndarray
            Current state estimate with shape (state_dim, 1)
        cov : np.ndarray
            Current state covariance with shape (state_dim, state_dim)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state and covariance
        """
        # Extract parameters
        K = self.K
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h
        mu = (
            self.params.mu.reshape(-1, 1)
            if self.params.mu.ndim == 1
            else self.params.mu
        )

        # Extract current state components
        factors = state[:K]
        log_vols = state[K:]

        # Convert to information form (precision matrix)
        I_curr = jnp.linalg.inv(cov)

        # Predict factors: E[f_{t+1}|t] = Phi_f @ f_t
        predicted_factors = Phi_f @ factors

        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h @ (log_vols - mu)
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)

        # Combine predicted state
        predicted_state = np.vstack([predicted_factors, predicted_log_vols])

        # Get transition matrix
        F_t = self._get_transition_matrix(state)

        # Process noise covariance (using state-dependent factor noise)
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_f = np.diag(np.exp(log_vols.flatten()))  # Factor process noise
        Q_h = self.params.Q_h  # Log-volatility process noise
        Q_t[:K, :K] = Q_f
        Q_t[K:, K:] = Q_h
        predicted_cov = F_t @ cov @ F_t.T + Q_t

        # Ensure the covariance is symmetric
        predicted_cov = (predicted_cov + predicted_cov.T) / 2

        return predicted_state, predicted_cov

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the Bellman update step using JAX-based optimization.

        Parameters
        ----------
        predicted_state : np.ndarray
            Predicted state with shape (state_dim, 1)
        predicted_cov : np.ndarray
            Predicted state covariance with shape (state_dim, state_dim)
        observation : np.ndarray
            Current observation with shape (N, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Updated state, updated covariance, and log-likelihood contribution
        """
        # Convert inputs to JAX arrays
        jax_observation = jnp.array(observation)

        # Compute information matrix (inverse of predicted covariance)
        try:
            # Use Cholesky for numerical stability
            jax_predicted_cov = jnp.array(predicted_cov)
            L = jax.scipy.linalg.cholesky(jax_predicted_cov, lower=True)
            jax_I_pred = jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.state_dim))
        except:
            # Fallback to regularized pseudoinverse
            jax_I_pred = jnp.linalg.pinv(
                jnp.array(predicted_cov) + 1e-8 * jnp.eye(self.state_dim)
            )

        # Ensure symmetry
        jax_I_pred = (jax_I_pred + jax_I_pred.T) / 2.0

        # Initial guess is the predicted state
        alpha = jnp.array(predicted_state.flatten())

        # Run block diagonal update
        updated_state = self.block_coordinate_update(
            alpha, predicted_state, jax_I_pred, observation, max_iters=10
        ).reshape(-1, 1)

        # Compute the Hessian at the optimum for covariance estimation
        hessian = np.array(
            self.jax_hessian(
                jnp.array(updated_state),
                jax_observation,
            )
        ).reshape(self.state_dim, self.state_dim)
        updated_cov = np.linalg.inv(
            hessian + jax_I_pred + np.eye(self.state_dim) * 1e-8
        )

        # Ensure symmetry
        updated_cov = (updated_cov + updated_cov.T) / 2.0

        val = self.log_posterior(updated_state, jax_observation)
        log_likelihood = -float(
            val
        )  # Negate because we minimize negative log-posterior

        return updated_state, updated_cov, log_likelihood

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the state transition matrix for the DFSV model.

        Parameters
        ----------
        state : np.ndarray
            Current state vector with shape (state_dim, 1)

        Returns
        -------
        np.ndarray
            Transition matrix with shape (state_dim, state_dim)
        """
        K = self.K
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h

        # Initialize transition matrix
        F_t = np.zeros((self.state_dim, self.state_dim))

        # Top-left block: factor transition
        F_t[:K, :K] = Phi_f

        # Bottom-right block: log-volatility transition
        F_t[K:, K:] = Phi_h

        return F_t

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter on the provided data.

        Parameters
        ----------
        y : np.ndarray
            Observed returns with shape (T, N) or (N, T)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states with shape (T, state_dim),
            filtered covariances with shape (T, state_dim, state_dim),
            and total log-likelihood
        """
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            # If tqdm is not installed, create a simple pass-through iterator
            def tqdm(iterable, **kwargs):
                return iterable

            print("Warning: tqdm not installed. No progress bar will be shown.")

        # Ensure y is in (T, N) format
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]

        # Storage for filtered states and covariances
        filtered_states = np.zeros((T, self.state_dim))
        filtered_covs = np.zeros((T, self.state_dim, self.state_dim))

        # Initialize
        state, cov = self.initialize_state(y.T)  # Note: initialize_state expects (N, T)
        log_likelihood = 0.0

        # Forward pass
        for t in tqdm(range(T), desc="Bellman Filter Progress"):
            # Prediction step
            predicted_state, predicted_cov = self.predict(state, cov)
            # eigs = np.linalg.eigvals(predicted_cov)
            # print("eigs", eigs)
            # Update step - reshape observation to (N, 1)
            observation = y[t : t + 1, :].T.reshape(-1, 1)
            state, cov, ll_contrib = self.update(
                predicted_state, predicted_cov, observation
            )

            # Store results (convert state from column vector to row vector)
            filtered_states[t, :] = state.flatten()
            filtered_covs[t, :, :] = cov
            log_likelihood += ll_contrib

        # Store results in object
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        return filtered_states, filtered_covs, log_likelihood
