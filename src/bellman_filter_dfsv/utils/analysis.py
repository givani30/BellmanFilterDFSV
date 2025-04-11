"""Analysis utilities for DFSV models.

Provides accuracy metrics for comparing true and estimated values.
"""

import numpy as np
import jax.numpy as jnp

def calculate_accuracy(true_values, estimated_values):
    """Calculate RMSE and correlation between true and estimated values.

    Args:
        true_values: Ground truth array (T, K) or (T,).
        estimated_values: Estimated array (T, K) or (T,).

    Returns:
        Tuple of (rmse, correlation), each np.ndarray of shape (K,).
    """
    if isinstance(true_values, jnp.ndarray):
        true_values = np.array(true_values)
    if isinstance(estimated_values, jnp.ndarray):
        estimated_values = np.array(estimated_values)

    if true_values.shape != estimated_values.shape:
        min_T = min(true_values.shape[0], estimated_values.shape[0])
        if abs(true_values.shape[0] - estimated_values.shape[0]) <= 1 and true_values.shape[1:] == estimated_values.shape[1:]:
            true_values = true_values[:min_T]
            estimated_values = estimated_values[:min_T]
        else:
            num_cols = true_values.shape[1] if true_values.ndim > 1 else 1
            return np.full(num_cols, np.nan), np.full(num_cols, np.nan)

    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
        estimated_values = estimated_values.reshape(-1, 1)

    rmse = np.sqrt(np.nanmean((true_values - estimated_values) ** 2, axis=0))
    correlations = []
    for k in range(true_values.shape[1]):
        valid_mask = ~np.isnan(true_values[:, k]) & ~np.isnan(estimated_values[:, k])
        if np.sum(valid_mask) < 2:
            corr = np.nan
        else:
            if np.std(true_values[valid_mask, k]) < 1e-10 or np.std(estimated_values[valid_mask, k]) < 1e-10:
                corr = np.nan
            else:
                try:
                    corr_matrix = np.corrcoef(true_values[valid_mask, k], estimated_values[valid_mask, k])
                    if corr_matrix.shape == (2, 2):
                        corr = corr_matrix[0, 1]
                    else:
                        corr = np.nan if np.isnan(corr_matrix) else float(corr_matrix)
                except Exception:
                    corr = np.nan
        correlations.append(corr)
    return rmse, np.array(correlations)