import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults, MLEModelResultsWrapper
from scipy.linalg import solve_discrete_lyapunov # If stationary initialization were needed
import warnings

# Ensure calculations use float64 for stability
# np.set_printoptions(precision=8, suppress=True) # Optional: for printing

class FactorCVModel(MLEModel):
    """
    Custom State Space Model for Factor-CV (N=95, K=5).

    Assumes diagonal Sigma_epsilon and Sigma_nu.
    Enforces stationarity of Phi_f via eigenvalue penalty.
    """
    # Constants for stationarity penalty (from thesis [cite: 175])
    STABILITY_PENALTY_WEIGHT = 1e4
    STABILITY_EIG_BUFFER = 1e-6

    def __init__(self, endog, k_factors):
        # Basic setup
        self.k_factors = k_factors
        n_obs = endog.shape[1]
        k_states = k_factors # State vector is just the factors f_t
        k_posdef = k_factors # Dimension of state covariance matrix Sigma_nu (assumed diagonal)

        # Initialize the state space model
        super().__init__(endog, k_states=k_states, k_posdef=k_posdef,
                         initialization='approximate_diffuse', # Use diffuse prior for latent factors
                         loglikelihood_burn=k_factors) # Burn-in for diffuse

        # Determine number of parameters and their indices
        self._initialize_parameter_indices(n_obs, k_factors)

        # Set fixed components of the state space model
        # Selection matrix R = Identity (maps state innovations nu_t to state equation)
        self.ssm['selection'] = np.eye(k_states)
        # Intercepts are zero as data is demeaned
        self.ssm['state_intercept'] = np.zeros((k_states, 1))
        self.ssm['obs_intercept'] = np.zeros((n_obs, 1))

        print(f"Initialized FactorCVModel: N={n_obs}, K={k_factors}")
        print(f"Number of parameters: {self.k_params}")

    def _initialize_parameter_indices(self, n_obs, k_factors):
        """Helper to calculate indices for slicing the parameter vector."""
        self.n_obs = n_obs
        idx = 0

        # Indices for Sigma_epsilon (N diagonal elements -> N parameters)
        self.param_indices_sigma_eps = slice(idx, idx + n_obs)
        idx += n_obs

        # Indices for Lambda (factor loadings) - only free elements
        # Identification: First KxK block is lower triangular with unit diagonal
        # Free elements: K*(K-1)/2 below diagonal in first K rows
        #             + (N-K)*K elements in the remaining rows
        n_lambda_free = k_factors * (k_factors - 1) // 2 + (n_obs - k_factors) * k_factors
        self.param_indices_lambda = slice(idx, idx + n_lambda_free)
        idx += n_lambda_free

        # Indices for Phi_f (K*K elements -> K^2 parameters)
        self.param_indices_phi_f = slice(idx, idx + k_factors**2)
        idx += k_factors**2

        # Indices for Sigma_nu (K diagonal elements -> K parameters)
        self.param_indices_sigma_nu = slice(idx, idx + k_factors)
        idx += k_factors

        self.k_params = idx # Total number of parameters

    @property
    def start_params(self):
        """ Define starting values for the optimizer (unconstrained space). """
        params = np.zeros(self.k_params)

        # 1. log(Sigma_epsilon diagonals): Guess based on sample variance * 0.5
        sample_vars = np.maximum(np.var(self.endog, axis=0), 1e-8) # Avoid log(0)
        sigma_eps_guess = 0.5 * sample_vars
        params[self.param_indices_sigma_eps] = np.log(sigma_eps_guess)

        # 2. Lambda free elements: Start near zero (unconstrained)
        n_lambda_free = self.param_indices_lambda.stop - self.param_indices_lambda.start
        params[self.param_indices_lambda] = 0.01 * np.random.randn(n_lambda_free)

        # 3. Phi_f elements (vectorized): Start near diagonal 0.8 (unconstrained)
        phi_f_guess = 0.8 * np.eye(self.k_factors)
        # Add small random noise to off-diagonals
        phi_f_guess += 0.01 * np.random.randn(self.k_factors, self.k_factors) * (1-np.eye(self.k_factors))
        params[self.param_indices_phi_f] = phi_f_guess.flatten() # Store vectorized

        # 4. log(Sigma_nu diagonals): Start with small positive variance, e.g., 0.05^2
        sigma_nu_guess = np.full(self.k_factors, 0.0025) # 0.05^2
        params[self.param_indices_sigma_nu] = np.log(sigma_nu_guess)

        return params

    @property
    def param_names(self):
        """ Define names for the parameters for result summaries. """
        names = []
        # Sigma_epsilon names (log scale)
        names += [f'log_sigma_eps.{i}' for i in range(self.n_obs)]
        # Lambda names (free elements, unconstrained)
        names += [f'lambda_free.{i}' for i in range(self.param_indices_lambda.stop - self.param_indices_lambda.start)]
        # Phi_f names (vectorized, unconstrained)
        names += [f'phi_f.{i}.{j}' for i in range(self.k_factors) for j in range(self.k_factors)]
        # Sigma_nu names (log scale)
        names += [f'log_sigma_nu.{i}' for i in range(self.k_factors)]
        return names

    def transform_params(self, unconstrained):
        """ Apply transformations: unconstrained -> constrained space. """
        constrained = np.array(unconstrained) # Ensure it's a NumPy array
        # 1. Sigma_epsilon: exp transform for positivity
        constrained[self.param_indices_sigma_eps] = np.exp(unconstrained[self.param_indices_sigma_eps])
        # 2. Lambda: No transformation needed for free elements
        # constrained[self.param_indices_lambda] = unconstrained[self.param_indices_lambda]
        # 3. Phi_f: No transformation needed for elements themselves (stationarity handled via penalty)
        # constrained[self.param_indices_phi_f] = unconstrained[self.param_indices_phi_f]
        # 4. Sigma_nu: exp transform for positivity
        constrained[self.param_indices_sigma_nu] = np.exp(unconstrained[self.param_indices_sigma_nu])
        return constrained

    def untransform_params(self, constrained):
        """ Reverse transformations: constrained -> unconstrained space. """
        unconstrained = np.array(constrained) # Ensure it's a NumPy array
        # 1. Sigma_epsilon: log transform
        # Add small epsilon to prevent log(0)
        unconstrained[self.param_indices_sigma_eps] = np.log(np.maximum(constrained[self.param_indices_sigma_eps], 1e-10))
        # 2. Lambda: No transformation
        # unconstrained[self.param_indices_lambda] = constrained[self.param_indices_lambda]
        # 3. Phi_f: No transformation
        # unconstrained[self.param_indices_phi_f] = constrained[self.param_indices_phi_f]
        # 4. Sigma_nu: log transform
        unconstrained[self.param_indices_sigma_nu] = np.log(np.maximum(constrained[self.param_indices_sigma_nu], 1e-10))
        return unconstrained

    def update(self, params, **kwargs):
        """
        Update state space system matrices based on constrained parameters.

        Args:
            params (ndarray): Array of parameters in the *constrained* space.
            **kwargs: Additional keyword arguments.

        Returns:
            None: Modifies the `self.ssm` object in place.
        """
        # Note: `statsmodels` calls this with constrained params after transform_params
        # So, 'params' here are already transformed (e.g., variances are positive)

        # --- Update Observation Equation Matrices ---
        # obs_cov (H_t): Sigma_epsilon (diagonal, constant)
        # Ensure positivity just in case, though transform_params should handle it
        sigma_eps_diag = np.maximum(params[self.param_indices_sigma_eps], 1e-10)
        self.ssm['obs_cov', 0, 0] = np.diag(sigma_eps_diag)

        # design (Z_t): Lambda (factor loadings, constant)
        lambda_free_params = params[self.param_indices_lambda]
        lambda_mat = np.zeros((self.n_obs, self.k_factors))
        current_idx = 0
        # Fill the first KxK block (lower triangular with unit diagonal)
        for i in range(self.k_factors):
            lambda_mat[i, i] = 1.0 # Diagonal element
            # Fill lower triangle elements for row i from free params
            n_elements_in_row = i
            if n_elements_in_row > 0:
                row_elements = lambda_free_params[current_idx : current_idx + n_elements_in_row]
                lambda_mat[i, :i] = row_elements
                current_idx += n_elements_in_row
        # Fill the remaining N-K rows
        n_elements_remaining = (self.n_obs - self.k_factors) * self.k_factors
        if n_elements_remaining > 0:
            remaining_rows_flat = lambda_free_params[current_idx:]
            if len(remaining_rows_flat) != n_elements_remaining:
                 raise ValueError(f"Lambda parameter size mismatch. Expected {n_elements_remaining} for remaining rows, got {len(remaining_rows_flat)}")
            lambda_mat[self.k_factors:, :] = remaining_rows_flat.reshape(
                 (self.n_obs - self.k_factors, self.k_factors)
             )
        self.ssm['design', 0, 0] = lambda_mat

        # --- Update State Equation Matrices ---
        # transition (T_t): Phi_f (constant)
        phi_f_mat = params[self.param_indices_phi_f].reshape(self.k_factors, self.k_factors)
        # Stationarity is handled via penalty in loglike, not by adjusting the matrix here
        self.ssm['transition', 0, 0] = phi_f_mat

        # state_cov (Q_t): Sigma_nu (diagonal, constant)
        sigma_nu_diag = np.maximum(params[self.param_indices_sigma_nu], 1e-10)
        self.ssm['state_cov', 0, 0] = np.diag(sigma_nu_diag)

    def _calculate_stationarity_penalty(self, phi_f_mat):
        """Calculates penalty for Phi_f violating stationarity."""
        try:
            eigenvalues = np.linalg.eigvals(phi_f_mat)
            max_abs_eig = np.max(np.abs(eigenvalues))
            # Penalty increases sharply if max eigenvalue magnitude > (1 - buffer)
            # Using ReLU like function: max(0, value - threshold)
            penalty = np.sum(np.maximum(0, np.abs(eigenvalues) - (1.0 - self.STABILITY_EIG_BUFFER)))
            #penalty = np.maximum(0, max_abs_eig - (1.0 - self.STABILITY_EIG_BUFFER))
            return self.STABILITY_PENALTY_WEIGHT * penalty
        except np.linalg.LinAlgError:
            # Penalize heavily if eigenvalue computation fails
            warnings.warn("Eigenvalue computation failed for Phi_f. Applying large penalty.")
            return self.STABILITY_PENALTY_WEIGHT * self.k_factors * 10 # Arbitrarily large penalty

    def loglike(self, params, *args, **kwargs):
        """
        Calculate the log-likelihood, adding a penalty for non-stationarity.

        Args:
            params (ndarray): Array of parameters in the *unconstrained* space.
            *args, **kwargs: Additional arguments passed to the superclass loglike.

        Returns:
            float: Penalized log-likelihood value.
        """
        # 1. Call superclass update to transform params and update ssm matrices
        #    This ensures self.ssm reflects the current *constrained* parameters
        #    corresponding to the unconstrained input `params`.
        #    super().update() calls self.transform_params and then self.update()
        try:
             super().update(params, **kwargs)
        except ValueError as e:
            # Catch potential errors during parameter transformation or matrix update
             warnings.warn(f"Error during model update in loglike: {e}. Returning -inf.")
             return -np.inf


        # 2. Calculate the base log-likelihood using the Kalman filter
        #    The ssm object now contains the constrained matrices needed by the filter
        try:
            base_loglike = super().loglike(params, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError) as e:
             # Handle potential numerical issues in the Kalman filter itself
             warnings.warn(f"Error calculating base log-likelihood: {e}. Returning -inf.")
             base_loglike = -np.inf

        # If base loglike calculation failed, return -inf immediately
        if not np.isfinite(base_loglike):
            return -np.inf

        # 3. Extract the *constrained* Phi_f matrix from the ssm object
        phi_f_mat_constrained = self.ssm['transition', 0, 0]

        # 4. Calculate the stationarity penalty based on constrained Phi_f
        penalty = self._calculate_stationarity_penalty(phi_f_mat_constrained)

        # 5. Return penalized log-likelihood
        penalized_loglike = base_loglike - penalty # Subtract penalty

        # Check for non-finite results before returning
        if not np.isfinite(penalized_loglike):
             warnings.warn(f"Non-finite penalized log-likelihood ({penalized_loglike}). Base LL: {base_loglike}, Penalty: {penalty}. Returning -inf.")
             return -np.inf

        return penalized_loglike
