# BellmanFilterDFSV Examples (v2.0.0)

This directory contains example scripts demonstrating the v2.0.0 architecture of the BellmanFilterDFSV package.

**Note**: v2.0.0 is a complete rewrite with breaking changes. These examples showcase the new Equinox-based functional architecture.

## Available Examples

### 1. DFSV Model Simulation (`01_dfsv_simulation.py`)

**Demonstrates:**
- Creating DFSV model parameters using `DFSVParams`
- Simulating data with `simulate_dfsv()`
- Analyzing simulated returns, factors, and log-volatilities
- Visualizing time series

**Key APIs:**
```python
from bellman_filter_dfsv import DFSVParams, simulate_dfsv

params = DFSVParams(lambda_r=..., Phi_f=..., Phi_h=..., mu=..., sigma2=..., Q_h=...)
returns, factors, log_vols = simulate_dfsv(params, T=1000, key=42)
```

### 2. Basic Filtering (`02_basic_filtering.py`)

**Demonstrates:**
- Applying filters to estimate latent states:
  - `BellmanFilter` (Bellman Information Filter)
  - `ParticleFilter` (Bootstrap Particle Filter)
- Comparing filter performance (accuracy, speed)
- Visualizing filtered estimates vs. true states

**Key APIs:**
```python
from bellman_filter_dfsv import BellmanFilter, ParticleFilter

bf = BellmanFilter(params)
result = bf.filter(returns)  # Returns FilterResult(means, infos, log_likelihood)

pf = ParticleFilter(params, num_particles=1000)
result = pf.filter(returns)  # Returns ParticleFilterResult(means, covs, log_likelihood)
```

### 3. Parameter Estimation with MLE (`03_parameter_optimization.py`)

**Demonstrates:**
- Maximum Likelihood Estimation using `fit_mle()`
- Creating initial parameter guesses
- Tracking optimization progress
- Comparing estimated vs. true parameters

**Key APIs:**
```python
from bellman_filter_dfsv import fit_mle

estimated_params, loss_history = fit_mle(
    start_params=initial_guess,
    observations=returns,
    num_steps=50,
    learning_rate=0.01,
)
```

### 4. EM Algorithm (`04_em_algorithm.py`)

**Demonstrates:**
- Expectation-Maximization algorithm using `fit_em()`
- Uses Rao-Blackwellized Particle Smoother for E-step
- Closed-form M-step updates

**Key APIs:**
```python
from bellman_filter_dfsv import fit_em

estimated_params = fit_em(
    start_params=initial_guess,
    observations=returns,
    num_em_steps=10,
    num_particles=500,
    num_trajectories=50,
)
```

### 5. Particle Cloud Visualization (`05_particle_cloud.py`)

**Demonstrates:**
- Rao-Blackwellized Particle Smoother (`run_rbps()`)
- "Uncertainty collapse" phenomenon
- Simulating volatility shocks
- Visualizing particle distributions over time

**Key APIs:**
```python
from bellman_filter_dfsv import run_rbps

rbps_result = run_rbps(
    params=params,
    observations=returns,
    num_particles=500,
    num_trajectories=100,
    seed=42,
)
# rbps_result.h_samples: (num_trajectories, T, K) - sampled log-vol paths
# rbps_result.f_smooth_means: (num_trajectories, T, K) - conditional factor means
```

**Output**: Creates `particle_cloud_visualization.png` showing how particle uncertainty narrows after informative shocks.

## Running the Examples

```bash
# From the project root directory
uv run python examples/01_dfsv_simulation.py
uv run python examples/02_basic_filtering.py
uv run python examples/03_parameter_optimization.py
uv run python examples/04_em_algorithm.py
uv run python examples/05_particle_cloud.py
```

## Dependencies

Core dependencies (installed with the package):
- `jax`, `jaxlib` - Array operations and automatic differentiation
- `jaxtyping` - Type annotations for JAX arrays
- `equinox` - Neural network library (used for module structure)
- `optax` - Optimization library
- `optimistix` - Advanced optimizers

Visualization dependencies (install separately):
```bash
uv pip install matplotlib numpy
```

## Key Differences from v1.x

| Feature | v1.x (Old) | v2.0.0 (New) |
|---------|------------|--------------|
| **Architecture** | OOP class hierarchy | Functional Core + Equinox modules |
| **Parameters** | `DFSVParamsDataclass(N, K, ...)` | `DFSVParams(lambda_r, ...)` (NamedTuple) |
| **Filters** | `DFSVBellmanInformationFilter(N, K)` | `BellmanFilter(params)` |
| **Filter Call** | `filter.filter(params, data)` | `filter.filter(data)` |
| **Optimization** | `run_optimization(...)` | `fit_mle(...)` or `fit_em(...)` |
| **Simulation** | `simulate_DFSV(params, T, seed)` | `simulate_dfsv(params, T, key=seed)` |
| **Type Safety** | Partial | Full `jaxtyping` annotations |
| **JIT Compatibility** | Limited | Full support |

## Migration from v1.x

If you have v1.x code, here's a quick migration guide:

**Old (v1.x):**
```python
from bellman_filter_dfsv.core.models import DFSVParamsDataclass
from bellman_filter_dfsv.core.filters import DFSVBellmanInformationFilter
from bellman_filter_dfsv.core.models.simulation import simulate_DFSV

params = DFSVParamsDataclass(N=3, K=1, lambda_r=..., ...)
bif = DFSVBellmanInformationFilter(N=3, K=1)
states, covs, ll = bif.filter(params, returns)
returns, factors, h = simulate_DFSV(params, T=500, seed=42)
```

**New (v2.0.0):**
```python
from bellman_filter_dfsv import DFSVParams, BellmanFilter, simulate_dfsv

params = DFSVParams(lambda_r=..., Phi_f=..., Phi_h=..., mu=..., sigma2=..., Q_h=...)
bf = BellmanFilter(params)
result = bf.filter(returns)  # result.means, result.infos, result.log_likelihood
returns, factors, h = simulate_dfsv(params, T=500, key=42)
```

## Further Reading

- **API Documentation**: See the full API reference in `/docs/build/html/index.html`
- **Academic Paper**: Coming soon
- **Research Repository**: [BellmanFilterDFSV-ThesisResearch](https://github.com/givani30/BellmanFilterDFSV-ThesisResearch)

## Need Help?

- Open an issue on GitHub
- Email: givaniboek@hotmail.com
