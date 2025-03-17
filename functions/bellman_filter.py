import jax.scipy.optimize
import numpy as np
from typing import Tuple
from functions.filters import DFSVFilter
from functions.simulation import DFSV_params
import jax
import jax.numpy as jnp
from jax import jit
import jaxopt
from functools import partial


class DFSVBellmanFilter(DFSVFilter):
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
        self._setup_jax_functions()

    def _setup_jax_params(self):
        """
        Convert numpy parameters to JAX arrays for use in JAX functions.
        """
        # Convert model parameters to JAX arrays
        self.jax_lambda_r = jnp.array(self.params.lambda_r)

        # Handle scalar or vector sigma2
        if np.isscalar(self.params.sigma2) or self.params.sigma2.ndim == 1:
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

    def _setup_jax_functions(self):
        """
        Set up and JIT-compile JAX functions for efficiency.
        """
        # Create the Fisher information function
        self.fisher_information = self._create_fisher_information_fn()

        # Create log posterior function for Hessian computation
        self.log_posterior = self._create_log_posterior_fn()

        # Create negative log posterior function for h optimization
        self.neg_log_post_h = self._create_neg_log_post_h_fn()

        # Create the solver for h updates
        self.solver_h = jaxopt.LBFGS(
            fun=self.neg_log_post_h,
            value_and_grad=False,
            tol=1e-4,
            verbose=False,
        )

        # Create the block coordinate update function
        self.block_coordinate_update = self._create_block_coordinate_update_fn()

        # Create the objective and gradient function
        self.obj_and_grad_fn = self._create_obj_and_grad_fn()

        # Store derived functions
        self.jax_bellman_objective = jit(
            lambda x, p, i, o: self.obj_and_grad_fn(x, p, i, o)[0]
        )
        self.jax_gradient = jit(lambda x, p, i, o: self.obj_and_grad_fn(x, p, i, o)[1])
        self.jax_hessian = jit(jax.hessian(self.log_posterior, argnums=0))

        # Precompile the functions with dummy data to warm up JIT compilation
        self._precompile_jax_functions()

    def _precompile_jax_functions(self):
        """
        Precompile JIT functions with dummy data to avoid compilation during filtering.
        Executes a complete filter cycle (predict + update) to warm up all JAX functions.
        """
        try:
            # Create dummy data of the right shapes
            K = self.K
            N = self.params.lambda_r.shape[0]

            # Create dummy state, covariance, and observation
            dummy_state = np.zeros((2 * K, 1))
            dummy_cov = np.eye(2 * K)
            dummy_obs = np.zeros((N, 1))

            # Precompile individual functions first (for diagnostics)
            dummy_alpha = jnp.zeros(2 * K)
            dummy_state_jax = jnp.zeros((2 * K, 1))
            dummy_I_pred = jnp.eye(2 * K)
            dummy_obs_jax = jnp.zeros((N, 1))

            # Fisher information
            _ = self.fisher_information(dummy_alpha)

            # Log posterior
            _ = self.log_posterior(dummy_alpha, dummy_obs_jax.flatten())

            # Negative log posterior for h
            _ = self.neg_log_post_h(
                dummy_alpha[K:],
                dummy_alpha[:K],
                dummy_state_jax.flatten(),
                dummy_I_pred,
                dummy_obs_jax.flatten(),
            )

            # Block coordinate update
            _ = self.block_coordinate_update(
                dummy_alpha,
                dummy_state_jax.flatten(),
                dummy_I_pred,
                dummy_obs_jax.flatten(),
                max_iters=1,
            )

            # Objective and gradient
            _, _ = self.obj_and_grad_fn(
                dummy_alpha,
                dummy_state_jax.flatten(),
                dummy_I_pred,
                dummy_obs_jax.flatten(),
            )

            # Hessian
            _ = self.jax_hessian(dummy_alpha, dummy_obs_jax.flatten())

            # Now run a complete filter cycle (predict + update)
            print("Precompiling complete filter cycle...")

            # Prediction step
            pred_state, pred_cov = self.predict(dummy_state, dummy_cov)

            # Update step
            _, _, _ = self.update(pred_state, pred_cov, dummy_obs)

            print("JAX functions and filter cycle successfully precompiled.")
        except Exception as e:
            print(f"Warning: Error during JAX function precompilation: {e}")
            print("Functions will be compiled during the first filter iteration.")
            import traceback

            traceback.print_exc()

    @staticmethod
    @jit
    def build_covariance(lambda_r, exp_h, sigma2):
        """
        Build observation covariance A = Lambda diag(exp_h) Lambda^T + diag(sigma2).

        Parameters:
        lambda_r: (N,K) - factor loadings
        exp_h: (K,) - exponentiated log-volatilities
        sigma2: (N,) or (N,N) - idiosyncratic variances

        Returns:
        A: (N,N) - observation covariance matrix
        """
        K = lambda_r.shape[1]
        N = lambda_r.shape[0]

        if sigma2.ndim == 1:
            sigma2 = jnp.diag(sigma2)

        Sigma_f = jnp.diag(exp_h.flatten())  # (K,K)
        lambda_r = lambda_r.reshape(N, K)  # (N,K)
        A = lambda_r @ Sigma_f @ lambda_r.T  # (N,N)
        A += jnp.diag(jnp.diag(sigma2)) + 1e-8 * jnp.eye(N)  # Add regularization
        A = 0.5 * (A + A.T)  # Ensure symmetry
        return A

    def _create_fisher_information_fn(self):
        """Create the Fisher information function"""

        @jit
        def fisher_information(alpha):
            """
            Compute the Fisher information of the *likelihood*
            for r ~ N(Lambda f, Lambda diag(e^h) Lambda^T + Sigma_e),
            w.r.t. alpha = [f, h].

            Returns: I_F, shape (2K, 2K)
            """
            # Unpack
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
            K = lambda_r.shape[1]

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
                return exp_h[k] * jnp.outer(lambda_r[:, k], lambda_r[:, k])

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

        return fisher_information

    def _create_log_posterior_fn(self):
        """Create the log posterior function (without penalty term)"""

        @jit
        def log_posterior(alpha, observation):
            """
            JAX implementation of the log-posterior (without the penalty term).

            Parameters:
            alpha: (2K,) - current state estimate [f, log_vols]
            observation: (N,) - observation vector
            Returns:
            Log posterior value (scalar)
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
            A = self.build_covariance(lambda_r, exp_log_vols, sigma2)

            # Compute log-likelihood with Cholesky decomposition
            L = jnp.linalg.cholesky(A)

            # Compute quadratic form via triangular solve
            alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
            quad_form = jnp.sum(alpha_vec**2)

            # Log determinant
            logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            # log-likelihood
            N_ = observation.shape[0]
            log_lik = -0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

            return log_lik

        return log_posterior

    def _create_neg_log_post_h_fn(self):
        """Create the negative log posterior function for h optimization"""

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
            A = self.build_covariance(lambda_r, exp_log_vols, sigma2)

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

        return neg_log_post_h

    def _create_block_coordinate_update_fn(self):
        """Create the block coordinate update function"""

        @partial(jit, static_argnames=["max_iters"])
        def block_coordinate_update(
            alpha, pred_state, I_pred, observation, max_iters=5
        ):
            """
            Perform block-coordinate updates on [f, h], accounting for cross terms
            in the predicted precision matrix I_pred.

            Parameters
            ----------
            alpha : jnp.ndarray
                Initial state guess [f, h]
            pred_state : jnp.ndarray
                Predicted state
            I_pred : jnp.ndarray
                Predicted precision matrix
            observation : jnp.ndarray
                Observation vector
            max_iters : int
                Number of coordinate iterations

            Returns
            -------
            jnp.ndarray
                Updated state [f, h]
            """
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
            K = lambda_r.shape[1]
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
                A = self.build_covariance(lambda_r, jnp.exp(log_volatility), sigma2)
                L = jax.scipy.linalg.cho_factor(A, lower=True)

                # Helper function to compute A^-1 @ x using Cholesky
                def A_inv(x):
                    return jax.scipy.linalg.cho_solve(L, x)

                # Left-hand side: (Lambda^T A^-1 Lambda + I_ff)
                lhs_mat = jnp.dot(lambda_r.T, A_inv(lambda_r)) + I_f  # shape (K,K)

                # Right-hand side: Lambda^T A^-1 y + I_ff f_pred + I_fh (h_pred - h)
                Lambda_inv_y = jnp.dot(lambda_r.T, A_inv(observation))  # shape (K,)
                rhs_vec = (
                    Lambda_inv_y
                    + jnp.dot(I_f, factors_pred)
                    + jnp.dot(I_fh, (-log_vols_pred + log_volatility))
                )

                # Solve linear system
                factors_new = jnp.linalg.solve(lhs_mat, rhs_vec)
                return factors_new

            def update_h_bfgs(h_init, f):
                """
                Minimize J(h) = neg_log_post_h(h, f, ...) w.r.t. h using BFGS.
                """
                # The objective function
                result = self.solver_h.run(
                    init_params=h_init,
                    factors=f,
                    predicted_state=pred_state,
                    I_pred=I_pred,
                    observation=observation,
                )

                # Check if optimization was successful
                def true_fun(h):
                    return h

                def false_fun(h):
                    return h_init

                return jax.lax.cond(
                    result.state.error < 1e-4, true_fun, false_fun, result.params
                )

            # Update loop
            f, h = factors_guess, log_vols_guess
            for _ in range(max_iters):
                f = update_factors(h)
                h = update_h_bfgs(h, f)

            # Return updated state
            return jnp.concatenate([f, h])

        return block_coordinate_update

    def _create_obj_and_grad_fn(self):
        """Create the objective and gradient function for optimization"""

        @jit
        def obj_and_grad_fn(x, predicted_state, I_pred, observation):
            """
            Combined objective and gradient function for optimization.

            Parameters:
            x: (2K,) - current state estimate [f, log_vols]
            predicted_state: (2K,) - predicted state
            I_pred: (2K,2K) - predicted precision matrix
            observation: (N,) - observation vector

            Returns:
            (obj, grad) - objective value and gradient
            """
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
            A = self.build_covariance(lambda_r, exp_h, sigma2)

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

        return obj_and_grad_fn

    # ... rest of the class implementation remains the same
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
        fisher_info = self.fisher_information(updated_state)
        updated_cov = np.linalg.inv(fisher_info + jax_I_pred)

        # Ensure symmetry
        updated_cov = (updated_cov + updated_cov.T) / 2.0

        val = self.log_posterior(updated_state, jax_observation)
        # TODO: Likelihood is almost certainly incorrect currently.
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
        return super().filter(y)
