import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Callable, Any
from jaxtyping import Float, Array

from .types import DFSVParams
from .filters import BellmanFilter


def constrain_params_default(p_unc: DFSVParams) -> DFSVParams:
    """Default transformation from unconstrained to constrained parameters."""
    # 1. Lambda: Unconstrained
    lambda_r = p_unc.lambda_r

    # 2. Phi: (-1, 1) via tanh
    Phi_f = jnp.tanh(p_unc.Phi_f)
    Phi_h = jnp.tanh(p_unc.Phi_h)

    # 3. Mu: Unconstrained
    mu = p_unc.mu

    # 4. Variances: Positive via softplus
    sigma2 = jax.nn.softplus(p_unc.sigma2)
    Q_h = jnp.diag(jax.nn.softplus(jnp.diag(p_unc.Q_h)))

    return DFSVParams(lambda_r, Phi_f, Phi_h, mu, sigma2, Q_h)


def unconstrain_params_default(p: DFSVParams) -> DFSVParams:
    """Default transformation from constrained to unconstrained parameters."""

    def inv_softplus(y):
        return jnp.log(jnp.exp(y) - 1.0)

    # Clip tanh inputs to avoid infinities
    clip_tanh = lambda x: jnp.arctanh(jnp.clip(x, -0.999, 0.999))

    return DFSVParams(
        lambda_r=p.lambda_r,
        Phi_f=clip_tanh(p.Phi_f),
        Phi_h=clip_tanh(p.Phi_h),
        mu=p.mu,
        sigma2=inv_softplus(p.sigma2),
        Q_h=jnp.diag(inv_softplus(jnp.diag(p.Q_h))),
    )


def fit_mle(
    start_params: DFSVParams,
    observations: Float[Array, "T N"],
    learning_rate: float = 0.01,
    num_steps: int = 100,
    optimizer: optax.GradientTransformation = None,
    constrain_fn: Callable[[DFSVParams], DFSVParams] = constrain_params_default,
    unconstrain_fn: Callable[[DFSVParams], DFSVParams] = unconstrain_params_default,
    verbose: bool = True,
) -> tuple[DFSVParams, list[float]]:
    """Fits DFSV parameters using Maximum Likelihood Estimation (MLE).

    Args:
        start_params: Initial guess for parameters.
        observations: Observed data matrix (Time x N).
        learning_rate: Learning rate for Adam optimizer (default: 0.01).
        num_steps: Number of optimization steps.
        optimizer: Custom Optax optimizer (optional). If None, uses Adam.
        constrain_fn: Function to map unconstrained -> constrained params.
        unconstrain_fn: Function to map constrained -> unconstrained params.
        verbose: Whether to print progress.

    Returns:
        tuple: (optimized_params, loss_history)
    """

    # 1. Setup Optimizer
    if optimizer is None:
        optimizer = optax.adam(learning_rate=learning_rate)

    # 2. Unconstrain Initial Parameters
    params_unc = unconstrain_fn(start_params)
    opt_state = optimizer.init(params_unc)

    # 3. Define Loss Function
    def loss_fn(p_u, obs):
        p_c = constrain_fn(p_u)
        # Create filter dynamically (zero cost in JIT)
        bf = BellmanFilter(p_c)
        return -bf.filter(obs).log_likelihood

    # 4. JIT Compile Step
    @jax.jit
    def step(p_u, opt_s, obs):
        loss, grads = jax.value_and_grad(loss_fn)(p_u, obs)
        updates, opt_s = optimizer.update(grads, opt_s, p_u)
        p_u = eqx.apply_updates(p_u, updates)
        return p_u, opt_s, loss

    # 5. Optimization Loop
    loss_history = []
    current_params = params_unc

    if verbose:
        print(f"Starting MLE Optimization ({num_steps} steps)...")

    for i in range(num_steps):
        current_params, opt_state, loss = step(current_params, opt_state, observations)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if verbose and (i % (num_steps // 10) == 0 or i == num_steps - 1):
            print(f"Step {i:4d} | Log-Likelihood: {-loss_val:.4f}")

    # 6. Return Constrained Parameters
    final_params = constrain_fn(current_params)
    return final_params, loss_history
