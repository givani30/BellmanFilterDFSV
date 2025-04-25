import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults
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
    # Constants for stationarity penalty and numerical stability
    STABILITY_PENALTY_WEIGHT = 100
    STABILITY_EIG_BUFFER = 1e-6
    MIN_VARIANCE = 1e-4  # Minimum allowed variance
    JITTER = 1e-4  # Jitter for numerical stability

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

        # 1. log(Sigma_epsilon diagonals): Start with actual sample variances
        sample_vars = np.var(self.endog, axis=0)
        # Ensure minimum variance and use full variance
        sigma_eps_guess = np.maximum(sample_vars, self.MIN_VARIANCE)
        params[self.param_indices_sigma_eps] = np.log(sigma_eps_guess)

        # 2. Lambda free elements: Start with small positive values (unconstrained)
        n_lambda_free = self.param_indices_lambda.stop - self.param_indices_lambda.start
        # Use a small positive value instead of random values centered at zero
        params[self.param_indices_lambda] = 0.1 + 0.01 * np.random.randn(n_lambda_free)

        # 3. Phi_f elements (vectorized): Start with more moderate persistence (unconstrained)
        phi_f_guess = 0.5 * np.eye(self.k_factors)  # Reduced from 0.8 to 0.5
        # Add small random noise to off-diagonals
        phi_f_guess += 0.01 * np.random.randn(self.k_factors, self.k_factors) * (1-np.eye(self.k_factors))
        params[self.param_indices_phi_f] = phi_f_guess.flatten() # Store vectorized

        # 4. log(Sigma_nu diagonals): Start with state variances based on factor analysis
        # Use larger initial state variance to encourage factor dynamics
        sigma_nu_guess = np.full(self.k_factors, 0.1)  # Increased from 0.05 to 0.1
        params[self.param_indices_sigma_nu] = np.log(np.maximum(sigma_nu_guess, self.MIN_VARIANCE))

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
        # 1. Sigma_epsilon: exp transform with minimum bound
        constrained[self.param_indices_sigma_eps] = np.maximum(
            np.exp(unconstrained[self.param_indices_sigma_eps]),
            self.MIN_VARIANCE
        )
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
        unconstrained[self.param_indices_sigma_eps] = np.log(
            np.maximum(constrained[self.param_indices_sigma_eps],
            self.MIN_VARIANCE)
        )
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

        # In statsmodels, for diagonal obs_cov, we need to set just the diagonal elements
        # not the full matrix
        self.ssm['obs_cov'] = np.diag(sigma_eps_diag)

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

            # Reshape with explicit dtype to ensure consistency
            reshaped_rows = remaining_rows_flat.reshape((self.n_obs - self.k_factors, self.k_factors))
            lambda_mat[self.k_factors:, :] = reshaped_rows

        # Set the design matrix - use a copy to ensure no reference issues
        self.ssm['design'] = lambda_mat.copy()

        # --- Update State Equation Matrices ---
        # transition (T_t): Phi_f (constant)
        phi_f_mat = params[self.param_indices_phi_f].reshape(self.k_factors, self.k_factors)
        # Stationarity is handled via penalty in loglike, not by adjusting the matrix here
        self.ssm['transition'] = phi_f_mat.copy()

        # state_cov (Q_t): Sigma_nu (diagonal, constant)
        sigma_nu_diag = np.maximum(params[self.param_indices_sigma_nu], 1e-10)
        # Set diagonal elements directly
        self.ssm['state_cov'] = np.diag(sigma_nu_diag)

    def _calculate_stationarity_penalty(self, phi_f_mat):
        """Calculates penalty for Phi_f violating stationarity."""
        try:
            eigenvalues = np.linalg.eigvals(phi_f_mat)
            max_abs_eig = np.max(np.abs(eigenvalues))
            # Penalty increases sharply if max eigenvalue magnitude > (1 - buffer)
            # Using ReLU like function: max(0, value - threshold)
            # Only penalize the maximum eigenvalue to create a smoother optimization surface
            penalty = np.maximum(0, max_abs_eig - (1.0 - self.STABILITY_EIG_BUFFER))
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
        # Enable detailed diagnostics
        debug = True
        if debug:
            print("\n==== DIAGNOSTIC: loglike method ====")
            print(f"Input params shape: {params.shape}")
            print(f"Input params range: [{np.min(params):.6f}, {np.max(params):.6f}]")
            print(f"Input params mean: {np.mean(params):.6f}")
            print(f"Input params has NaN: {np.isnan(params).any()}")
            print(f"Input params has Inf: {np.isinf(params).any()}")

            # Print specific parameter groups
            print(f"log_sigma_eps params: {params[self.param_indices_sigma_eps][:5]}...")
            print(f"lambda_free params: {params[self.param_indices_lambda][:5]}...")
            print(f"phi_f params: {params[self.param_indices_phi_f]}")
            print(f"log_sigma_nu params: {params[self.param_indices_sigma_nu]}")

        # 1. Call superclass update to transform params and update ssm matrices
        #    This ensures self.ssm reflects the current *constrained* parameters
        #    corresponding to the unconstrained input `params`.
        #    super().update() calls self.transform_params and then self.update()
        try:
             # First transform the parameters
             constrained_params = self.transform_params(params)

             if debug:
                 print("\nAfter transform_params:")
                 print(f"Constrained params shape: {constrained_params.shape}")
                 print(f"Constrained params range: [{np.min(constrained_params):.6f}, {np.max(constrained_params):.6f}]")
                 print(f"Constrained params mean: {np.mean(constrained_params):.6f}")
                 print(f"Constrained params has NaN: {np.isnan(constrained_params).any()}")
                 print(f"Constrained params has Inf: {np.isinf(constrained_params).any()}")

                 # Print specific parameter groups
                 print(f"sigma_eps params: {constrained_params[self.param_indices_sigma_eps][:5]}...")
                 print(f"lambda_free params: {constrained_params[self.param_indices_lambda][:5]}...")
                 print(f"phi_f params: {constrained_params[self.param_indices_phi_f]}")
                 print(f"sigma_nu params: {constrained_params[self.param_indices_sigma_nu]}")

             # Then update the state space matrices
             self.update(constrained_params, transformed=True)

             if debug:
                 print("\nAfter update:")
                 print(f"design matrix shape: {self.ssm['design'].shape}")
                 print(f"design matrix range: [{np.min(self.ssm['design']):.6f}, {np.max(self.ssm['design']):.6f}]")
                 print(f"obs_cov matrix shape: {self.ssm['obs_cov'].shape}")
                 print(f"obs_cov matrix range: [{np.min(self.ssm['obs_cov']):.6f}, {np.max(self.ssm['obs_cov']):.6f}]")
                 print(f"transition matrix shape: {self.ssm['transition'].shape}")
                 print(f"transition matrix range: [{np.min(self.ssm['transition']):.6f}, {np.max(self.ssm['transition']):.6f}]")
                 print(f"state_cov matrix shape: {self.ssm['state_cov'].shape}")
                 print(f"state_cov matrix range: [{np.min(self.ssm['state_cov']):.6f}, {np.max(self.ssm['state_cov']):.6f}]")

        except Exception as e:
            # Catch potential errors during parameter transformation or matrix update
             warnings.warn(f"Error during model update in loglike: {e}. Returning -inf.")
             import traceback
             traceback.print_exc()
             return -np.inf


        # 2. Calculate the base log-likelihood using the Kalman filter
        #    The ssm object now contains the constrained matrices needed by the filter
        try:
            # Add a small jitter to the observation and state covariance matrices
            # to improve numerical stability
            # Add jitter using class constant for numerical stability
            self.ssm['obs_cov'] = self.ssm['obs_cov'] + self.JITTER * np.eye(self.n_obs)
            self.ssm['state_cov'] = self.ssm['state_cov'] + self.JITTER * np.eye(self.k_factors)

            if debug:
                print("\nAfter adding jitter:")
                print(f"obs_cov matrix range: [{np.min(self.ssm['obs_cov']):.6f}, {np.max(self.ssm['obs_cov']):.6f}]")
                print(f"state_cov matrix range: [{np.min(self.ssm['state_cov']):.6f}, {np.max(self.ssm['state_cov']):.6f}]")
                print(f"obs_cov matrix condition number: {np.linalg.cond(self.ssm['obs_cov']):.6e}")
                print(f"state_cov matrix condition number: {np.linalg.cond(self.ssm['state_cov']):.6e}")
                print(f"obs_cov matrix determinant: {np.linalg.det(self.ssm['obs_cov']):.6e}")
                print(f"state_cov matrix determinant: {np.linalg.det(self.ssm['state_cov']):.6e}")

                # Check for positive definiteness
                try:
                    np.linalg.cholesky(self.ssm['obs_cov'])
                    print("obs_cov is positive definite")
                except np.linalg.LinAlgError:
                    print("WARNING: obs_cov is not positive definite!")

                try:
                    np.linalg.cholesky(self.ssm['state_cov'])
                    print("state_cov is positive definite")
                except np.linalg.LinAlgError:
                    print("WARNING: state_cov is not positive definite!")

            # Call the superclass loglike method to compute the base log-likelihood
            base_loglike = super().loglike(params, *args, **kwargs)

            if debug:
                print(f"\nBase log-likelihood: {base_loglike:.6f}")
                print(f"Base log-likelihood per observation: {base_loglike / self.nobs:.6f}")
                print(f"Base log-likelihood per variable: {base_loglike / (self.nobs * self.n_obs):.6f}")

        except Exception as e:
             # Handle potential numerical issues in the Kalman filter itself
             warnings.warn(f"Error calculating base log-likelihood: {e}. Returning -inf.")
             import traceback
             traceback.print_exc()
             base_loglike = -np.inf

        # If base loglike calculation failed, return -inf immediately
        if not np.isfinite(base_loglike):
            return -np.inf

        # 3. Extract the *constrained* Phi_f matrix from the ssm object
        phi_f_mat_constrained = self.ssm['transition']

        if debug:
            print("\nStationarity check:")
            print(f"Phi_f matrix from ssm:")
            print(phi_f_mat_constrained)
            try:
                eigenvalues = np.linalg.eigvals(phi_f_mat_constrained)
                print(f"Phi_f eigenvalues: {eigenvalues}")
                print(f"Phi_f max abs eigenvalue: {np.max(np.abs(eigenvalues)):.6f}")
                print(f"Phi_f is {'stationary' if np.max(np.abs(eigenvalues)) < 1.0 else 'non-stationary'}")
            except Exception as e:
                print(f"Error computing eigenvalues: {e}")

        # 4. Calculate the stationarity penalty based on constrained Phi_f
        penalty = self._calculate_stationarity_penalty(phi_f_mat_constrained)

        if debug:
            print(f"Stationarity penalty: {penalty:.6f}")

        # 5. Return penalized log-likelihood
        penalized_loglike = base_loglike - penalty # Subtract penalty

        if debug:
            print(f"\nFinal penalized log-likelihood: {penalized_loglike:.6f}")
            if not np.isfinite(penalized_loglike):
                print("WARNING: Final log-likelihood is not finite!")

        # Check for non-finite results before returning
        if not np.isfinite(penalized_loglike):
             warnings.warn(f"Non-finite penalized log-likelihood ({penalized_loglike}). Base LL: {base_loglike}, Penalty: {penalty}. Returning -inf.")
             return -np.inf

        return penalized_loglike
