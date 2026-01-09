"""EM Algorithm optimizer for DFSV models."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..filters.bellman_information import DFSVBellmanInformationFilter
from ..models import DFSVParamsDataclass
from ._em_estep import accumulate_sufficient_stats
from ._em_mstep import m_step_full


@dataclass
class EMHistory:
    """Tracks EM algorithm convergence."""

    log_likelihoods: list[float] = field(default_factory=list)
    converged: bool = False
    n_iters: int = 0


class EMOptimizer:
    """
    EM algorithm for DFSV parameter estimation using BIF + RTS smoothing.

    Uses Gaussian approximation via Bellman Information Filter for the E-step
    and closed-form updates for the M-step.
    """

    def __init__(
        self,
        N: int,
        K: int,
        max_iters: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ):
        self.N = N
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

    def e_step(
        self,
        params: DFSVParamsDataclass,
        observations: Float[Array, "T N"],
    ):
        """Run E-step: filter + smooth + accumulate sufficient statistics."""
        bif = DFSVBellmanInformationFilter(N=self.N, K=self.K)
        _, _, log_likelihood = bif.filter_scan(params, np.asarray(observations))

        smoothed_states, smoothed_covs, smoothed_lag1_covs = bif.smooth(params)

        stats = accumulate_sufficient_stats(
            observations=jnp.asarray(observations),
            smoothed_states=jnp.asarray(smoothed_states),
            smoothed_covs=jnp.asarray(smoothed_covs),
            smoothed_lag1_covs=jnp.asarray(smoothed_lag1_covs),
            K=self.K,
        )

        return stats, float(log_likelihood)

    def m_step(self, stats) -> DFSVParamsDataclass:
        """Run M-step: update all parameters from sufficient statistics."""
        lambda_r, sigma2, Phi_f, mu, Phi_h, Q_h = m_step_full(stats)

        return DFSVParamsDataclass(
            lambda_r=np.asarray(lambda_r),
            Phi_f=np.asarray(Phi_f),
            Phi_h=np.asarray(Phi_h),
            mu=np.asarray(mu),
            Q_h=np.asarray(Q_h),
            sigma2=np.asarray(sigma2),
        )

    def fit(
        self,
        observations: Float[Array, "T N"],
        initial_params: DFSVParamsDataclass,
    ) -> tuple[DFSVParamsDataclass, EMHistory]:
        """
        Fit DFSV model using EM algorithm.

        Args:
            observations: Observed returns (T, N)
            initial_params: Starting parameter values

        Returns:
            (fitted_params, history)
        """
        params = initial_params
        history = EMHistory()

        for iteration in range(self.max_iters):
            stats, ll = self.e_step(params, observations)
            history.log_likelihoods.append(ll)

            if self.verbose:
                print(f"EM iter {iteration + 1}: log-likelihood = {ll:.4f}")

            if iteration > 0:
                ll_change = ll - history.log_likelihoods[-2]
                if abs(ll_change) < self.tol:
                    history.converged = True
                    history.n_iters = iteration + 1
                    if self.verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break

            params = self.m_step(stats)

        if not history.converged:
            history.n_iters = self.max_iters
            if self.verbose:
                print(f"Did not converge after {self.max_iters} iterations")

        return params, history
