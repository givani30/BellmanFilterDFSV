import jax
import jax.numpy as jnp
from bellman_filter_dfsv.next.types import DFSVParams
from bellman_filter_dfsv.next import fit_em
from bellman_filter_dfsv.core.models.simulation import simulate_DFSV
from bellman_filter_dfsv.core.models.dfsv import DFSVParamsDataclass as OldParams

jax.config.update("jax_enable_x64", True)


def run_next_em_experiment():
    print("=" * 60)
    print("Experiment 09: Testing 'next' EM Implementation")
    print("=" * 60)

    # 1. Simulation (Reuse existing simulation utility for now)
    T, N, K = 300, 3, 1
    true_params_old = OldParams(
        N=N,
        K=K,
        lambda_r=jnp.array([[0.9], [0.7], [0.8]]),
        Phi_f=jnp.array([[0.7]]),
        Phi_h=jnp.array([[0.95]]),
        mu=jnp.array([-0.5]),
        sigma2=jnp.array([0.1, 0.15, 0.12]),
        Q_h=jnp.array([[0.04]]),
    )

    returns, _, _ = simulate_DFSV(true_params_old, T=T, seed=42)
    returns = jnp.array(returns)

    print(f"Data shape: {returns.shape}")

    # 2. Init Params (New Type)
    init_params = DFSVParams(
        lambda_r=true_params_old.lambda_r + 0.2,
        Phi_f=true_params_old.Phi_f,
        Phi_h=true_params_old.Phi_h,
        mu=true_params_old.mu + 0.5,
        sigma2=true_params_old.sigma2 + 0.1,
        Q_h=true_params_old.Q_h,
    )

    print("\nInitial Parameters:")
    print(f"  Lambda[0,0]: {init_params.lambda_r[0, 0]:.4f}")
    print(f"  Sigma2[0]:   {init_params.sigma2[0]:.4f}")

    # 3. Run EM using the new 'next' API
    final_params, history = fit_em(
        returns,
        init_params,
        num_particles=200,
        num_trajectories=20,
        max_iters=30,
        verbose=True,
    )

    # 4. Results
    print("\nFinal Results:")
    print("True Lambda:\n", true_params_old.lambda_r)
    print("Est Lambda:\n", final_params.lambda_r)

    print("True Sigma2:\n", true_params_old.sigma2)
    print("Est Sigma2:\n", final_params.sigma2)

    print("True Mu:\n", true_params_old.mu)
    print("Est Mu:\n", final_params.mu)

    # Simple check
    err_lambda = jnp.mean(jnp.abs(true_params_old.lambda_r - final_params.lambda_r))
    if err_lambda < 0.2:
        print("\nSUCCESS: Parameters recovered reasonably well.")
    else:
        print("\nFAILURE: Parameter recovery poor.")


if __name__ == "__main__":
    run_next_em_experiment()
