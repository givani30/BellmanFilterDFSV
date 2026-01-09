import jax
import jax.numpy as jnp

from bellman_filter_dfsv.filters import BellmanFilter
from bellman_filter_dfsv.smoothing import rts_smoother
from bellman_filter_dfsv.types import DFSVParams


def test_bellman_filter_v2_shapes():
    jax.config.update("jax_enable_x64", True)

    N, K, T = 3, 1, 50
    key = jax.random.PRNGKey(42)

    # Random parameters
    params = DFSVParams(
        lambda_r=jnp.ones((N, K)),
        Phi_f=0.9 * jnp.eye(K),
        Phi_h=0.95 * jnp.eye(K),
        mu=jnp.array([-1.0]),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.1 * jnp.eye(K),
    )

    # Random observations
    observations = jax.random.normal(key, (T, N))

    # Initialize filter
    bf = BellmanFilter(params)

    # Run filter
    result = bf.filter(observations)

    # Check shapes
    assert result.means.shape == (T, 2 * K)
    assert result.infos.shape == (T, 2 * K, 2 * K)
    assert result.log_likelihood.shape == ()

    # Run smoother
    smoothed = rts_smoother(params, result.means, result.infos)

    # Check smoother shapes
    assert smoothed.smoothed_means.shape == (T, 2 * K)
    assert smoothed.smoothed_covs.shape == (T, 2 * K, 2 * K)
    assert smoothed.smoothed_lag1_covs.shape == (T, 2 * K, 2 * K)

    print("V2 Filter Test Passed!")


if __name__ == "__main__":
    test_bellman_filter_v2_shapes()
