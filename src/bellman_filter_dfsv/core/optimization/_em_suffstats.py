"""
Sufficient statistics for EM algorithm on DFSV models.

This module provides the EMSufficientStats dataclass that stores all
sufficient statistics computed during the E-step. These are used by
the M-step for closed-form parameter updates.

The statistics are organized by which parameter update they support:
- Observation equation: λ_r, σ² updates
- Factor dynamics: Φ_f update (with weighted LS for SV)
- Log-volatility dynamics: μ, Φ_h, Q_h updates
"""

from typing import NamedTuple

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float


@jdc.pytree_dataclass
class EMSufficientStats:
    """
    Sufficient statistics for EM parameter updates in DFSV models.

    All statistics are computed from the smoothed posterior q(f_{1:T}, h_{1:T} | r_{1:T})
    obtained via BIF filtering + RTS smoothing.

    The naming convention uses subscripts to indicate:
    - sum_*: Summation over time index t
    - *_f, *_h: Which state variable (factors vs log-vols)
    - *_prev: Time t-1 (lagged)
    - *_sq: Squared terms (diagonal of outer product)

    Attributes:
        # --- Observation Equation Statistics (for λ_r, σ²) ---
        sum_r_f: Σ_{t=1}^T r_t E[f_t]', shape (N, K)
        sum_f_f: Σ_{t=1}^T E[f_t f_t'], shape (K, K)
        sum_r_r_diag: Σ_{t=1}^T r_t ⊙ r_t (element-wise), shape (N,)

        # --- Factor Dynamics Statistics (for Φ_f) ---
        sum_f_fprev: Σ_{t=2}^T E[f_t f_{t-1}'], shape (K, K)
        sum_fprev_fprev: Σ_{t=2}^T E[f_{t-1} f_{t-1}'], shape (K, K)
        sum_exp_neg_h: Σ_{t=2}^T E[exp(-h_t)], shape (K,)
        sum_exp_neg_h_f_fprev_diag: Σ_{t=2}^T E[exp(-h_kt)] E[f_kt f_{k,t-1}], shape (K,)
        sum_exp_neg_h_fprev_sq: Σ_{t=2}^T E[exp(-h_kt)] E[f_{k,t-1}²], shape (K,)

        # --- Log-Volatility Dynamics Statistics (for μ, Φ_h, Q_h) ---
        sum_h: Σ_{t=2}^T E[h_t], shape (K,)  # Note: t=2..T for dynamics
        sum_hprev: Σ_{t=2}^T E[h_{t-1}], shape (K,)
        sum_h_h: Σ_{t=2}^T E[h_t h_t'], shape (K, K)
        sum_h_hprev: Σ_{t=2}^T E[h_t h_{t-1}'], shape (K, K)
        sum_hprev_hprev: Σ_{t=2}^T E[h_{t-1} h_{t-1}'], shape (K, K)

        # --- Counts ---
        T: Total number of time steps (static, for computing averages)
    """

    # === Observation Equation Statistics ===
    # For updating λ_r and σ²

    sum_r_f: Float[Array, "N K"]  # Σ_{t=1}^T r_t E[f_t]'
    sum_f_f: Float[Array, "K K"]  # Σ_{t=1}^T E[f_t f_t']
    sum_r_r_diag: Float[Array, "N"]  # Σ_{t=1}^T r_t ⊙ r_t

    # === Factor Dynamics Statistics ===
    # For updating Φ_f (diagonal) with weighted least squares

    sum_f_fprev: Float[Array, "K K"]  # Σ_{t=2}^T E[f_t f_{t-1}']
    sum_fprev_fprev: Float[Array, "K K"]  # Σ_{t=2}^T E[f_{t-1} f_{t-1}']
    sum_exp_neg_h: Float[Array, "K"]  # Σ_{t=2}^T E[exp(-h_t)]
    sum_exp_neg_h_f_fprev_diag: Float[
        Array, "K"
    ]  # Σ_{t=2}^T E[exp(-h_kt)] E[f_kt f_{k,t-1}]
    sum_exp_neg_h_fprev_sq: Float[Array, "K"]  # Σ_{t=2}^T E[exp(-h_kt)] E[f_{k,t-1}²]

    # === Log-Volatility Dynamics Statistics ===
    # For updating μ, Φ_h (diagonal), Q_h (diagonal)

    sum_h: Float[Array, "K"]  # Σ_{t=2}^T E[h_t]
    sum_hprev: Float[Array, "K"]  # Σ_{t=2}^T E[h_{t-1}]
    sum_h_h: Float[Array, "K K"]  # Σ_{t=2}^T E[h_t h_t']
    sum_h_hprev: Float[Array, "K K"]  # Σ_{t=2}^T E[h_t h_{t-1}']
    sum_hprev_hprev: Float[Array, "K K"]  # Σ_{t=2}^T E[h_{t-1} h_{t-1}']

    # === Counts ===
    T: jdc.Static[int]  # Total number of time steps

    @classmethod
    def zeros(cls, N: int, K: int, T: int) -> "EMSufficientStats":
        """
        Create a zero-initialized EMSufficientStats.

        Used as initial accumulator for jax.lax.scan accumulation.

        Args:
            N: Number of observed series
            K: Number of latent factors
            T: Total number of time steps

        Returns:
            EMSufficientStats with all arrays initialized to zeros
        """
        return cls(
            # Observation stats
            sum_r_f=jnp.zeros((N, K)),
            sum_f_f=jnp.zeros((K, K)),
            sum_r_r_diag=jnp.zeros(N),
            # Factor dynamics stats
            sum_f_fprev=jnp.zeros((K, K)),
            sum_fprev_fprev=jnp.zeros((K, K)),
            sum_exp_neg_h=jnp.zeros(K),
            sum_exp_neg_h_f_fprev_diag=jnp.zeros(K),
            sum_exp_neg_h_fprev_sq=jnp.zeros(K),
            # Log-vol dynamics stats
            sum_h=jnp.zeros(K),
            sum_hprev=jnp.zeros(K),
            sum_h_h=jnp.zeros((K, K)),
            sum_h_hprev=jnp.zeros((K, K)),
            sum_hprev_hprev=jnp.zeros((K, K)),
            # Count
            T=T,
        )

    def __add__(self, other: "EMSufficientStats") -> "EMSufficientStats":
        """
        Element-wise addition of sufficient statistics.

        Useful for combining stats from parallel processing or minibatches.

        Args:
            other: Another EMSufficientStats to add

        Returns:
            New EMSufficientStats with summed values
        """
        return EMSufficientStats(
            sum_r_f=self.sum_r_f + other.sum_r_f,
            sum_f_f=self.sum_f_f + other.sum_f_f,
            sum_r_r_diag=self.sum_r_r_diag + other.sum_r_r_diag,
            sum_f_fprev=self.sum_f_fprev + other.sum_f_fprev,
            sum_fprev_fprev=self.sum_fprev_fprev + other.sum_fprev_fprev,
            sum_exp_neg_h=self.sum_exp_neg_h + other.sum_exp_neg_h,
            sum_exp_neg_h_f_fprev_diag=self.sum_exp_neg_h_f_fprev_diag
            + other.sum_exp_neg_h_f_fprev_diag,
            sum_exp_neg_h_fprev_sq=self.sum_exp_neg_h_fprev_sq
            + other.sum_exp_neg_h_fprev_sq,
            sum_h=self.sum_h + other.sum_h,
            sum_hprev=self.sum_hprev + other.sum_hprev,
            sum_h_h=self.sum_h_h + other.sum_h_h,
            sum_h_hprev=self.sum_h_hprev + other.sum_h_hprev,
            sum_hprev_hprev=self.sum_hprev_hprev + other.sum_hprev_hprev,
            T=self.T,  # T should be the same
        )


class SmoothedMoments(NamedTuple):
    """
    Smoothed moments at a single time step.

    Used as intermediate representation during E-step computation
    before accumulation into sufficient statistics.

    Attributes:
        f_mean: E[f_t | r_{1:T}], shape (K,)
        h_mean: E[h_t | r_{1:T}], shape (K,)
        P_ff: Cov[f_t | r_{1:T}], shape (K, K)
        P_hh: Cov[h_t | r_{1:T}], shape (K, K)
        P_fh: Cov[f_t, h_t | r_{1:T}], shape (K, K)
    """

    f_mean: Float[Array, "K"]
    h_mean: Float[Array, "K"]
    P_ff: Float[Array, "K K"]
    P_hh: Float[Array, "K K"]
    P_fh: Float[Array, "K K"]


class SmoothedLagMoments(NamedTuple):
    """
    Smoothed lag-1 cross-moments for consecutive time steps.

    Used for computing statistics like E[f_t f_{t-1}'] needed
    for AR coefficient updates.

    Attributes:
        P_ff_lag: Cov[f_t, f_{t-1} | r_{1:T}], shape (K, K)
        P_hh_lag: Cov[h_t, h_{t-1} | r_{1:T}], shape (K, K)
    """

    P_ff_lag: Float[Array, "K K"]
    P_hh_lag: Float[Array, "K K"]
