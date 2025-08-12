# BellmanFilterDFSV

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-orange.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/givani30/BellmanFilterDFSV/workflows/Tests/badge.svg)](https://github.com/givani30/BellmanFilterDFSV/actions)
[![Documentation](https://github.com/givani30/BellmanFilterDFSV/workflows/Build%20and%20Deploy%20Documentation/badge.svg)](https://givani30.github.io/BellmanFilterDFSV/)

**High-performance JAX-based filtering for Dynamic Factor Stochastic Volatility (DFSV) models**

BellmanFilterDFSV is a Python package that provides efficient implementations of filtering algorithms for Dynamic Factor Stochastic Volatility models using JAX for automatic differentiation and JIT compilation.

## 🚀 Key Features

- **Multiple Filtering Algorithms**: Bellman Information Filter (BIF), Bellman Filter, and Particle Filter
- **JAX-Powered Performance**: Automatic differentiation, JIT compilation, and vectorization
- **Numerical Stability**: Advanced techniques for robust parameter estimation
- **Clean API**: Intuitive interface for research and applications
- **Extensible Design**: Easy to adapt for other state-space models
- **Comprehensive Testing**: Full test suite with 76+ tests

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

## 🚀 Quick Start

```python
import jax.numpy as jnp
from bellman_filter_dfsv.core.models import DFSVParamsDataclass, simulate_DFSV
from bellman_filter_dfsv.core.filters import DFSVBellmanInformationFilter

# Define model parameters
params = DFSVParamsDataclass(
    N=3,  # Number of observed series
    K=1,  # Number of factors
    Lambda=jnp.array([[0.8], [0.7], [0.9]]),
    phi_f=jnp.array([[0.7]]),
    phi_h=jnp.array([0.95]),
    sigma_f=jnp.array([1.0]),
    sigma_h=jnp.array([0.1]),
    sigma_eps=jnp.array([0.3, 0.25, 0.35]),
    mu=jnp.array([-1.2])
)

# Simulate data
returns, factors, log_vols = simulate_DFSV(params, T=500, key=42)

# Create and run filter
bif = DFSVBellmanInformationFilter(N=3, K=1)
states, covs, loglik = bif.filter(params, returns)

print(f"Log-likelihood: {loglik:.2f}")
print(f"Filtered states shape: {states.shape}")
```

## 📊 Examples

The package includes several comprehensive examples:

- **Basic Simulation**: Simulate DFSV models and analyze properties
- **Filter Comparison**: Compare BIF, Bellman, and Particle filters
- **Parameter Estimation**: Maximum likelihood estimation with optimization
- **Real Data Application**: Apply to financial time series

```bash
# Run examples
python examples/01_dfsv_simulation.py
python examples/02_basic_filtering.py
python examples/03_parameter_optimization.py
```

## 🏗️ Architecture

### DFSV Model

The Dynamic Factor Stochastic Volatility model represents observed returns as:

```
y_t = Λf_t + ε_t
```

Where:
- `y_t`: Observed returns (N×1)
- `f_t`: Latent factors (K×1)
- `Λ`: Factor loading matrix (N×K)
- `ε_t`: Idiosyncratic errors with stochastic volatility

### Filtering Algorithms

1. **Bellman Information Filter (BIF)**: Information-form implementation for numerical stability
2. **Bellman Filter**: Traditional covariance-form implementation
3. **Particle Filter**: Bootstrap particle filter for non-linear/non-Gaussian cases

## 📁 Project Structure

```text
BellmanFilterDFSV/
├── src/bellman_filter_dfsv/     # Core package
│   ├── core/                    # Main functionality
│   │   ├── filters/            # Filtering algorithms
│   │   ├── models/             # DFSV model definitions
│   │   └── optimization/       # Parameter estimation
│   └── utils/                  # Utility functions
├── examples/                   # Usage examples
├── tests/                     # Test suite
├── docs/                      # Documentation
├── thesis_artifacts/          # Research materials
└── pyproject.toml            # Package configuration
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=bellman_filter_dfsv

# Run specific test
uv run pytest tests/test_unified_filters.py
```

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

This package was developed as part of a quantitative finance thesis on Dynamic Factor Stochastic Volatility models. The research materials and thesis-specific code are archived in the `thesis_artifacts/` directory.

## 📞 Contact

- **Author**: Givani Boekestijn
- **Email**: givaniboek@hotmail.com
- **GitHub**: [@givani30](https://github.com/givani30)

---

**Citation**: If you use this package in your research, please cite the original thesis and this software package.
