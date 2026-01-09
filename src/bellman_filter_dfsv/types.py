from typing import NamedTuple

from jax import Array
from jaxtyping import Float


class DFSVParams(NamedTuple):
    """Parameters for the Dynamic Factor Stochastic Volatility Model.

    Attributes:
        lambda_r: Factor loadings Λ (N, K).
        Phi_f: Factor autoregression matrix Φ_f (K, K).
        Phi_h: Log-volatility autoregression matrix Φ_h (K, K).
        mu: Long-run mean of log-volatilities μ (K,).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        Q_h: Log-volatility innovation covariance Q_h (K, K).
    """

    lambda_r: Float[Array, "N K"]
    Phi_f: Float[Array, "K K"]
    Phi_h: Float[Array, "K K"]
    mu: Float[Array, "K"]
    sigma2: Float[Array, "N"]
    Q_h: Float[Array, "K K"]


class BIFState(NamedTuple):
    """State container for the Bellman Information Filter.

    Attributes:
        mean: Information state mean vector α_{t|t} (2K,).
        info: Information matrix Ω_{t|t} (2K, 2K).
    """

    mean: Float[Array, "2K"]
    info: Float[Array, "2K 2K"]


class FilterResult(NamedTuple):
    """Result container for a full filter run.

    Attributes:
        means: Filtered state means α_{t|t} (T, 2K).
        infos: Filtered information matrices Ω_{t|t} (T, 2K, 2K).
        log_likelihood: Total log-likelihood scalar.
    """

    means: Float[Array, "T 2K"]
    infos: Float[Array, "T 2K 2K"]
    log_likelihood: Float[Array, ""]


class ParticleState(NamedTuple):
    """State container for Particle Filter.

    Attributes:
        particles: Particle states (2K, num_particles).
        log_weights: Log-weights for each particle (num_particles,).
    """

    particles: Float[Array, "2K P"]
    log_weights: Float[Array, "P"]


class ParticleFilterResult(NamedTuple):
    """Result container for Particle Filter run.

    Attributes:
        means: Weighted mean estimates (T, 2K).
        covs: Weighted covariance estimates (T, 2K, 2K).
        log_likelihood: Total log-likelihood scalar.
    """

    means: Float[Array, "T 2K"]
    covs: Float[Array, "T 2K 2K"]
    log_likelihood: Float[Array, ""]


class EMSufficientStats(NamedTuple):
    """Sufficient statistics for EM Algorithm (M-step)."""

    sum_r_f: Float[Array, "N K"]
    sum_f_f: Float[Array, "K K"]
    sum_r_r_diag: Float[Array, "N"]

    sum_f_fprev: Float[Array, "K K"]
    sum_fprev_fprev: Float[Array, "K K"]
    sum_exp_neg_h: Float[Array, "K"]
    sum_exp_neg_h_f_fprev_diag: Float[Array, "K"]
    sum_exp_neg_h_fprev_sq: Float[Array, "K"]

    sum_h: Float[Array, "K"]
    sum_hprev: Float[Array, "K"]
    sum_h_h: Float[Array, "K K"]
    sum_h_hprev: Float[Array, "K K"]
    sum_hprev_hprev: Float[Array, "K K"]

    T: int


class RBParticleState(NamedTuple):
    """
    State for Rao-Blackwellized Particle Filter.
    """

    # h-particles (K, num_particles)
    h_particles: Float[Array, "K P"]

    # Conditional f-statistics (for each particle)
    # Mean f_{t|t} (K, num_particles)
    f_means: Float[Array, "K P"]
    # Covariance P_{t|t} (K, K, num_particles) - batched covariance
    f_covs: Float[Array, "K K P"]

    # Weights
    log_weights: Float[Array, "P"]


class RBPSResult(NamedTuple):
    """
    Result of Rao-Blackwellized Particle Smoothing.
    Contains M sampled trajectories and their conditional f-statistics.
    """

    # Sampled h trajectories (M, T, K)
    h_samples: Float[Array, "M T K"]

    # Conditional f smoothed means (M, T, K)
    f_smooth_means: Float[Array, "M T K"]

    # Conditional f smoothed covariances (M, T, K, K)
    f_smooth_covs: Float[Array, "M T K K"]

    # Conditional f smoothed lag-1 covariances (M, T-1, K, K)
    f_smooth_lag1_covs: Float[Array, "M T_minus_1 K K"]
