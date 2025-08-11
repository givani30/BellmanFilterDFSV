"""Parameter error calculation utilities for simulation study analysis.

This module provides utilities for calculating various error metrics between true and
estimated parameters in the DFSV model. It includes functions for computing comprehensive
error metrics, element-wise errors for matrices, and aggregating errors across replicates.
"""

import logging
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np
import polars as pl

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass


def calculate_comprehensive_param_errors(
    true_params: DFSVParamsDataclass,
    estimated_params: DFSVParamsDataclass
) -> Dict[str, float]:
    """Calculate comprehensive error metrics between true and estimated parameters.

    Args:
        true_params: True parameter values
        estimated_params: Estimated parameter values

    Returns:
        Dictionary containing all error metrics
    """
    if true_params is None or estimated_params is None:
        return {}  # Return empty dict if either params object is None

    errors = {}

    def safe_array(x: Optional[np.ndarray]) -> np.ndarray:
        """Convert None to zeros array of appropriate shape."""
        return np.zeros_like(x) if x is None else np.array(x)

    def calc_elementwise_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate element-wise error metrics."""
        diff = est - true
        if len(true.shape) == 2 and true.shape[0] == true.shape[1]:  # Square matrix
            K = true.shape[0]
            for i in range(K):
                for j in range(K):
                    errors[f"param_{name}_element_{i}_{j}_bias"] = float(diff[i, j])
            logging.debug(f"Calculated element-wise metrics for {name} with K={K}")

        errors[f"param_{name}_bias"] = float(np.mean(diff))
        errors[f"param_{name}_mae"] = float(np.mean(np.abs(diff)))
        errors[f"param_{name}_rmse"] = float(np.sqrt(np.mean(diff**2)))

    def calc_matrix_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate matrix-specific error metrics."""
        # Frobenius norm differences
        frob_diff = float(jnp.linalg.norm(est - true))
        errors[f"param_{name}_frob_diff"] = frob_diff
        true_norm = float(jnp.linalg.norm(true))
        if true_norm > 1e-10:  # Avoid division by zero
            errors[f"param_{name}_frob_rel_diff"] = frob_diff / true_norm
        else:
            errors[f"param_{name}_frob_rel_diff"] = np.nan

    def calc_square_matrix_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate square matrix-specific error metrics."""
        try:
            true_eigs = jnp.linalg.eigvals(true)
            est_eigs = jnp.linalg.eigvals(est)

            # Sort by magnitude for comparison
            true_eigs = jnp.sort(jnp.abs(true_eigs))
            est_eigs = jnp.sort(jnp.abs(est_eigs))

            eig_diff = est_eigs - true_eigs
            errors[f"param_{name}_eig_bias"] = float(jnp.mean(eig_diff))
            errors[f"param_{name}_eig_mae"] = float(jnp.mean(jnp.abs(eig_diff)))
            errors[f"param_{name}_eig_rmse"] = float(jnp.sqrt(jnp.mean(eig_diff**2)))
        except Exception as e:
            logging.warning(f"Failed to calculate eigenvalue metrics for {name}: {str(e)}")
            errors[f"param_{name}_eig_bias"] = np.nan
            errors[f"param_{name}_eig_mae"] = np.nan
            errors[f"param_{name}_eig_rmse"] = np.nan

    def calc_covariance_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate covariance matrix-specific error metrics."""
        try:
            # Log determinant difference using slogdet for stability
            true_sign, true_logdet = jnp.linalg.slogdet(true)
            est_sign, est_logdet = jnp.linalg.slogdet(est)
            if true_sign > 0 and est_sign > 0:
                errors[f"param_{name}_logdet_diff"] = float(est_logdet - true_logdet)
            else:
                errors[f"param_{name}_logdet_diff"] = np.nan

            # Trace difference
            errors[f"param_{name}_trace_diff"] = float(jnp.trace(est) - jnp.trace(true))
        except Exception as e:
            logging.warning(f"Failed to calculate covariance metrics for {name}: {str(e)}")
            errors[f"param_{name}_logdet_diff"] = np.nan
            errors[f"param_{name}_trace_diff"] = np.nan

    # Calculate metrics for each parameter
    # lambda_r (vector)
    true_lambda_r = safe_array(true_params.lambda_r)
    est_lambda_r = safe_array(estimated_params.lambda_r)
    calc_elementwise_metrics(true_lambda_r, est_lambda_r, "lambda_r")

    # Phi_f (square matrix)
    true_phi_f = safe_array(true_params.Phi_f)
    est_phi_f = safe_array(estimated_params.Phi_f)
    calc_elementwise_metrics(true_phi_f, est_phi_f, "Phi_f")
    calc_matrix_metrics(true_phi_f, est_phi_f, "Phi_f")
    calc_square_matrix_metrics(true_phi_f, est_phi_f, "Phi_f")

    # Phi_h (square matrix)
    true_phi_h = safe_array(true_params.Phi_h)
    est_phi_h = safe_array(estimated_params.Phi_h)
    calc_elementwise_metrics(true_phi_h, est_phi_h, "Phi_h")
    calc_matrix_metrics(true_phi_h, est_phi_h, "Phi_h")
    calc_square_matrix_metrics(true_phi_h, est_phi_h, "Phi_h")

    # Q_h (covariance matrix)
    true_q_h = safe_array(true_params.Q_h)
    est_q_h = safe_array(estimated_params.Q_h)
    calc_elementwise_metrics(true_q_h, est_q_h, "Q_h")
    calc_matrix_metrics(true_q_h, est_q_h, "Q_h")
    calc_square_matrix_metrics(true_q_h, est_q_h, "Q_h")
    calc_covariance_metrics(true_q_h, est_q_h, "Q_h")

    # mu (scalar)
    if hasattr(true_params, 'mu') and hasattr(estimated_params, 'mu'):
        true_mu = np.array([true_params.mu]) if true_params.mu is not None else np.array([0.0])
        est_mu = np.array([estimated_params.mu]) if estimated_params.mu is not None else np.array([0.0])
        calc_elementwise_metrics(true_mu, est_mu, "mu")

    # sigma2 (scalar)
    if hasattr(true_params, 'sigma2') and hasattr(estimated_params, 'sigma2'):
        true_sigma2 = np.array([true_params.sigma2]) if true_params.sigma2 is not None else np.array([0.0])
        est_sigma2 = np.array([estimated_params.sigma2]) if estimated_params.sigma2 is not None else np.array([0.0])
        calc_elementwise_metrics(true_sigma2, est_sigma2, "sigma2")

    return errors


def calculate_param_errors_for_replicate(
    unique_id: str,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> Optional[Dict[str, float]]:
    """Calculate parameter errors for a single replicate.

    Args:
        unique_id: The unique identifier for this replicate
        true_params_dict: Dictionary of true parameter objects
        estimated_params_dict: Dictionary of estimated parameter objects

    Returns:
        Dictionary with error metrics or None if parameters not found
    """
    true_params = true_params_dict.get(unique_id)
    estimated_params = estimated_params_dict.get(unique_id)

    if true_params is None or estimated_params is None:
        logging.warning(f"Missing parameters for unique_id: {unique_id}")
        return None

    errors = calculate_comprehensive_param_errors(true_params, estimated_params)
    return {'unique_id': unique_id, **errors}


def create_param_errors_df(
    df_success: pl.DataFrame,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> pl.DataFrame:
    """Create a DataFrame with parameter error metrics for all successful runs.

    Args:
        df_success: DataFrame with successful runs
        true_params_dict: Dictionary of true parameter objects
        estimated_params_dict: Dictionary of estimated parameter objects

    Returns:
        DataFrame with parameter error metrics
    """
    param_errors_list = []

    for row in df_success.iter_rows(named=True):
        unique_id = row['unique_id']
        errors = calculate_param_errors_for_replicate(
            unique_id, true_params_dict, estimated_params_dict
        )
        if errors is not None:
            param_errors_list.append(errors)

    if not param_errors_list:
        raise ValueError("No valid parameter error calculations found")

    # Convert to Polars DataFrame
    df_param_errors = pl.from_dicts(param_errors_list)
    logging.info("Created parameter errors DataFrame with shape: %s", df_param_errors.shape)

    return df_param_errors


def calculate_matrix_element_errors(
    true_params: DFSVParamsDataclass,
    estimated_params: DFSVParamsDataclass,
    param_name: str
) -> Dict[str, float]:
    """Calculate element-wise errors for matrix parameters.

    Args:
        true_params: True parameter values
        estimated_params: Estimated parameter values
        param_name: Name of the parameter (e.g., 'Phi_f')

    Returns:
        Dictionary containing errors for each matrix element
    """
    true_matrix = getattr(true_params, param_name)
    est_matrix = getattr(estimated_params, param_name)

    if true_matrix is None or est_matrix is None:
        return {}

    errors = {}
    K = true_matrix.shape[0]  # Get actual matrix dimension
    for i in range(K):
        for j in range(K):
            key = f"param_{param_name}_element_{i}_{j}_bias"
            errors[key] = float(est_matrix[i, j] - true_matrix[i, j])

    logging.debug(f"Calculated element-wise errors for {param_name} with K={K}")
    return errors