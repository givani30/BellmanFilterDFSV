# BellmanFilterDFSV

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-orange.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/givani30/BellmanFilterDFSV/workflows/Tests/badge.svg)](https://github.com/givani30/BellmanFilterDFSV/actions)
[![Documentation](https://github.com/givani30/BellmanFilterDFSV/workflows/Build%20and%20Deploy%20Documentation/badge.svg)](https://givani30.github.io/BellmanFilterDFSV/)

**High-performance JAX-based filtering for Dynamic Factor Stochastic Volatility (DFSV) models**

> **Note**: v2.0.0 is a complete rewrite with breaking changes. See [examples/README.md](examples/README.md) for migration guide.

BellmanFilterDFSV is a Python package that provides efficient implementations of filtering algorithms for Dynamic Factor Stochastic Volatility models using JAX for automatic differentiation and JIT compilation.

## 🚀 Key Features (v2.0.0)

- **Functional Architecture**: Built with Equinox for clean, composable JAX code
- **Multiple Filtering Algorithms**: Bellman Information Filter and Particle Filter
- **Advanced Smoothing**: RTS Smoother and Rao-Blackwellized Particle Smoother
- **Parameter Estimation**: MLE (via gradient descent) and EM algorithm
- **JAX-Powered Performance**: Automatic differentiation, JIT compilation, and vectorization
- **Full Type Safety**: Complete `jaxtyping` annotations for all array operations
- **Numerical Stability**: Advanced techniques for robust parameter estimation
- **Comprehensive Testing**: 93% test coverage with 69 tests including property-based tests

## 📦 Installation

### Basic Installation

```bash
pip install bellman-filter-dfsv
```

### With Optional Dependencies

```bash
# For data analysis and visualization
pip install bellman-filter-dfsv[analysis]

# For cloud computing and batch processing
pip install bellman-filter-dfsv[cloud]

# For notebook development
pip install bellman-filter-dfsv[notebooks]

# For econometric extensions
pip install bellman-filter-dfsv[econometrics]

# Everything
pip install bellman-filter-dfsv[all]
```

### Development Installation

```bash
git clone https://github.com/givani30/BellmanFilterDFSV.git
cd BellmanFilterDFSV
pip install -e .[dev,all]
```

Or using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/givani30/BellmanFilterDFSV.git
cd BellmanFilterDFSV
uv sync
uv run pytest  # Run tests
```

## 🚀 Quick Start (v2.0.0)

```python
import jax.numpy as jnp
from bellman_filter_dfsv import DFSVParams, BellmanFilter, simulate_dfsv

# Define model parameters
params = DFSVParams(
    lambda_r=jnp.array([[0.8], [0.7], [0.9]]),  # Factor loadings (N×K)
    Phi_f=jnp.array([[0.7]]),  # Factor AR matrix (K×K)
    Phi_h=jnp.array([[0.95]]),  # Log-vol AR matrix (K×K)
    mu=jnp.array([-1.2]),  # Long-run mean of log-vols (K,)
    sigma2=jnp.array([0.3, 0.25, 0.35]),  # Idiosyncratic variances (N,)
    Q_h=jnp.array([[0.01]])  # Log-vol innovation covariance (K×K)
)

# Simulate data
returns, factors, log_vols = simulate_dfsv(params, T=500, key=42)

# Create and run filter
bf = BellmanFilter(params)
result = bf.filter(returns)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Filtered states shape: {result.means.shape}")  # (T, 2K)
```

### Parameter Estimation

```python
from bellman_filter_dfsv import fit_mle

# Estimate parameters using MLE
estimated_params, loss_history = fit_mle(
    start_params=initial_guess,
    observations=returns,
    num_steps=100,
    learning_rate=0.01,
)

# Or use EM algorithm
from bellman_filter_dfsv import fit_em

estimated_params = fit_em(
    start_params=initial_guess,
    observations=returns,
    num_em_steps=10,
    num_particles=500,
    num_trajectories=50,
)
```

## 📊 Examples

The `examples/` directory contains 5 comprehensive examples demonstrating v2.0.0 features:

1. **01_dfsv_simulation.py** - Simulate DFSV models and analyze properties
2. **02_basic_filtering.py** - Compare BellmanFilter and ParticleFilter performance
3. **03_parameter_optimization.py** - Maximum likelihood estimation with `fit_mle`
4. **04_em_algorithm.py** - EM algorithm with Rao-Blackwellized Particle Smoother
5. **05_particle_cloud.py** - Visualize "uncertainty collapse" phenomenon

```bash
# Run examples
uv run python examples/01_dfsv_simulation.py
uv run python examples/02_basic_filtering.py
uv run python examples/03_parameter_optimization.py
```

See [examples/README.md](examples/README.md) for detailed documentation and v1→v2 migration guide.

## 🏗️ Architecture

### DFSV Model

The Dynamic Factor Stochastic Volatility model consists of three key equations:

**Observation Equation:**
```
r_t = λ_r f_t + e_t,  where e_t ~ N(0, Σ)
```

**Factor Dynamics:**
```
f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t,  where ε_t ~ N(0, I_K)
```

**Log-Volatility Dynamics:**
```
h_t = μ + Φ_h (h_{t-1} - μ) + η_t,  where η_t ~ N(0, Q_h)
```

Where:
- `r_t`: Observed returns (N×1)
- `f_t`: Latent factors with stochastic volatility (K×1)
- `h_t`: Log-volatilities of factors (K×1)
- `λ_r`: Factor loading matrix (N×K)
- `Φ_f`: Factor autoregression matrix (K×K)
- `Φ_h`: Log-volatility autoregression matrix (K×K)
- `μ`: Long-run mean of log-volatilities (K×1)
- `Σ`: Idiosyncratic error covariance (N×N)
- `Q_h`: Log-volatility innovation covariance (K×K)

### Filtering & Smoothing Algorithms (v2.0.0)

1. **Bellman Information Filter (BIF)**: Information-form Kalman filter for numerical stability
2. **Particle Filter**: Bootstrap particle filter with systematic resampling
3. **RTS Smoother**: Rauch-Tung-Striebel smoother (Direct Information Form)
4. **Rao-Blackwellized Particle Smoother (RBPS)**: Marginalizes out linear states analytically

## 📁 Project Structure (v2.0.0)

```text
BellmanFilterDFSV/
├── src/bellman_filter_dfsv/     # Core package (v2 architecture)
│   ├── filters.py              # BellmanFilter, ParticleFilter (Equinox modules)
│   ├── _bellman_math.py        # Pure math kernels for BIF
│   ├── _particle_math.py       # Pure math kernels for PF
│   ├── estimation.py           # fit_mle, fit_em
│   ├── smoothing.py            # rts_smoother, run_rbps
│   ├── simulation.py           # simulate_dfsv
│   ├── types.py                # DFSVParams, FilterResult, etc. (NamedTuples)
│   └── utils/                  # Utility functions
├── examples/                   # 5 v2 examples
├── tests/                      # 69 tests, 93% coverage
└── pyproject.toml             # Package configuration
```

## 🧪 Testing (v2.0.0)

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage (93% coverage achieved)
uv run pytest --cov=bellman_filter_dfsv --cov-report=term-missing

# Run specific module tests
uv run pytest tests/test_filters.py
uv run pytest tests/test_estimation.py
uv run pytest tests/test_smoothing.py

# Run property-based tests only
uv run pytest -m property
```

**Test Statistics (v2.0.0):**
- 69 tests total (68 passing, 1 skipped)
- 93% code coverage
- 7 property-based tests using Hypothesis
- All type checks passing (basedpyright: 0 errors)

## 📚 Documentation

**📖 [Full Documentation](https://givani30.github.io/BellmanFilterDFSV/)** - Complete API reference, tutorials, and examples

**Local Documentation Build:**

```bash
cd docs/
make html
# Open docs/build/html/index.html
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/source/contributing.rst) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research

This package was developed as part of a quantitative finance thesis on Dynamic Factor Stochastic Volatility models.

**📚 [Complete Research Materials](https://github.com/givani30/BellmanFilterDFSV-ThesisResearch)**

All thesis research materials, simulation studies, and experimental code are available in the dedicated research repository. This includes:

- 🎯 Monte Carlo simulation studies
- 📊 Empirical analysis with real financial data
- 🔬 Experimental implementations and prototypes
- 📈 Complete thesis results and figures
- 📝 Research notes and development logs

See [THESIS_RESEARCH.md](THESIS_RESEARCH.md) for detailed information.

## 📞 Contact

- **Author**: Givani Boekestijn
- **Email**: givaniboek@hotmail.com
- **GitHub**: [@givani30](https://github.com/givani30)

---

**Citation**: If you use this package in your research, please cite:

```bibtex
@software{boekestijn2025bellman,
  title={BellmanFilterDFSV: JAX Implementation of Bellman Filter for Dynamic Factor Stochastic Volatility Models},
  author={Boekestijn, Givani},
  year={2025},
  url={https://github.com/givani30/BellmanFilterDFSV},
  note={Research materials: https://github.com/givani30/BellmanFilterDFSV-ThesisResearch}
}
```
