# Package Template Guide

This document explains how to use BellmanFilterDFSV as a template for creating new JAX-based scientific computing packages.

## 🎯 Template Features

This package structure provides:

- **Modern Python packaging** with `pyproject.toml`
- **JAX-based scientific computing** with proper type hints
- **Modular architecture** with clean separation of concerns
- **Comprehensive testing** with pytest
- **Professional documentation** with Sphinx
- **Optional dependencies** for different use cases
- **Development tools** with ruff, mypy, and pre-commit hooks

## 🏗️ Package Structure Template

```text
your-package/
├── src/your_package/           # Core package
│   ├── __init__.py            # Public API exports
│   ├── core/                  # Main functionality
│   │   ├── __init__.py
│   │   ├── models/            # Model definitions
│   │   ├── algorithms/        # Core algorithms
│   │   └── optimization/      # Parameter estimation
│   └── utils/                 # Utility functions
├── examples/                  # Usage examples
├── tests/                     # Test suite
├── docs/                      # Sphinx documentation
├── pyproject.toml            # Package configuration
├── README.md                 # Project overview
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## 📋 Adaptation Checklist

### 1. Package Metadata (`pyproject.toml`)

```toml
[project]
name = "your-package-name"
version = "1.0.0"
description = "Your package description"
authors = [{name = "Your Name", email = "your.email@example.com"}]
keywords = ["jax", "scientific-computing", "your-domain"]
```

### 2. Core Module Structure

**Replace `bellman_filter_dfsv` with your package name:**

```bash
# Rename directories
mv src/bellman_filter_dfsv src/your_package

# Update imports throughout codebase
find . -name "*.py" -exec sed -i 's/bellman_filter_dfsv/your_package/g' {} \;
```

**Adapt core modules:**

- `core/models/` → Your domain models (e.g., `neural_networks/`, `optimization_problems/`)
- `core/filters/` → Your algorithms (e.g., `solvers/`, `estimators/`)
- `core/optimization/` → Your optimization routines

### 3. Dependencies

**Core dependencies to keep:**
- `jax`, `jaxlib` - JAX ecosystem
- `jax-dataclasses` - Structured parameters
- `jaxtyping` - Type hints
- `equinox` - Neural networks (if needed)
- `optimistix` - Optimization
- `numpy`, `scipy` - Scientific computing

**Domain-specific dependencies:**
- Replace econometrics packages with your domain packages
- Update optional dependency groups in `pyproject.toml`

### 4. Testing Structure

**Keep the testing framework:**
- `tests/conftest.py` - Common fixtures
- `tests/test_*.py` - Test modules
- Pytest configuration in `pyproject.toml`

**Adapt test content:**
- Replace DFSV-specific tests with your domain tests
- Keep numerical stability and JAX compatibility tests
- Maintain test coverage standards

### 5. Documentation

**Update documentation files:**
- `docs/source/index.rst` - Package overview
- `docs/source/installation.rst` - Installation instructions
- `docs/source/usage.rst` - Usage examples
- `docs/source/api/` - API documentation

**Adapt examples:**
- Replace `examples/01_dfsv_simulation.py` with your domain examples
- Keep the progressive complexity structure
- Maintain comprehensive docstrings

## 🔧 Development Setup Template

### Environment Configuration

```bash
# Clone and setup
git clone your-repo-url
cd your-package
uv sync  # or pip install -e .[dev,all]

# Development tools
uv run ruff check .     # Linting
uv run ruff format .    # Formatting
uv run mypy src/        # Type checking
uv run pytest          # Testing
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

## 📦 JAX Best Practices

### 1. Function Design

```python
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx

def your_algorithm(
    params: YourParamsClass,
    data: Float[Array, "time features"]
) -> Float[Array, "time outputs"]:
    """Your algorithm with proper type hints."""
    # Use JAX transformations
    return jax.vmap(your_step_function)(data)

# JIT compilation
your_algorithm = eqx.filter_jit(your_algorithm)
```

### 2. Parameter Classes

```python
from jax_dataclasses import pytree_dataclass
import jax.numpy as jnp

@pytree_dataclass
class YourParamsClass:
    """Parameters for your model."""
    param1: jnp.ndarray
    param2: float
    param3: jnp.ndarray
    
    @classmethod
    def create_default(cls, dim: int) -> "YourParamsClass":
        """Create default parameters."""
        return cls(
            param1=jnp.ones(dim),
            param2=1.0,
            param3=jnp.eye(dim)
        )
```

### 3. Numerical Stability

```python
# Use stable implementations
def stable_computation(x):
    # Avoid numerical issues
    return jnp.where(
        jnp.abs(x) < 1e-8,
        0.0,  # Stable fallback
        your_computation(x)
    )

# Error checking in JIT
import equinox as eqx

def your_function(x):
    eqx.error_if(
        jnp.any(jnp.isnan(x)),
        "NaN detected in input"
    )
    return your_computation(x)
```

## 🚀 Deployment

### PyPI Publishing

```bash
# Build package
uv build

# Upload to PyPI
uv publish
```

### Documentation Hosting

```bash
# Build docs
cd docs/
make html

# Deploy to GitHub Pages or ReadTheDocs
```

## 📚 Additional Resources

- **JAX Documentation**: https://jax.readthedocs.io/
- **Equinox Documentation**: https://docs.kidger.site/equinox/
- **Python Packaging**: https://packaging.python.org/
- **Sphinx Documentation**: https://www.sphinx-doc.org/

## 🤝 Contributing to Template

If you improve this template structure, please contribute back:

1. Fork the BellmanFilterDFSV repository
2. Make improvements to the template structure
3. Submit a pull request with your enhancements

---

This template provides a solid foundation for JAX-based scientific computing packages with professional standards and best practices.
