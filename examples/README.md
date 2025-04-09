# Bellman Filter DFSV Examples

This directory contains example scripts demonstrating the functionality of the Bellman Filter DFSV package. These examples showcase various aspects of the package, from basic model simulation to advanced parameter optimization.

## Example Scripts

### 1. DFSV Model Simulation (`01_dfsv_simulation.py`)

This example demonstrates how to:
- Create a Dynamic Factor Stochastic Volatility (DFSV) model
- Simulate data from this model
- Visualize the results

The DFSV model combines factor models with stochastic volatility, allowing for time-varying volatility in the latent factors that drive returns.

### 2. Basic Filtering (`02_basic_filtering.py`)

This example demonstrates how to:
- Create a DFSV model and simulate data
- Apply different filters to estimate the latent states:
  - Bellman Filter (BF)
  - Bellman Information Filter (BIF)
  - Particle Filter (PF)
- Compare filter performance and visualize results

### 3. Parameter Optimization (`03_parameter_optimization.py`)

This example demonstrates how to:
- Create a DFSV model and simulate data
- Estimate model parameters using different filters
- Compare estimated parameters to true parameters
- Visualize optimization results

### 4. Parameter Transformation (`04_parameter_transformation.py`)

This example demonstrates how to:
- Transform DFSV model parameters between constrained and unconstrained space
- Compare optimization with and without parameter transformations
- Visualize the benefits of parameter transformations for numerical stability

Parameter transformations map constrained parameters (e.g., positive definite matrices) to unconstrained space, which can improve optimization stability and convergence.

### 5. Optimizer Comparison (`05_optimizer_comparison.py`)

This example demonstrates how to:
- Create a DFSV model and simulate data
- Compare different optimizers for parameter estimation:
  - AdamW
  - DampedTrustRegionBFGS (default)
- Analyze optimizer performance in terms of:
  - Convergence speed
  - Final parameter accuracy
  - Numerical stability

### 6. Real Data Application (`06_real_data_application.py`)

This example demonstrates how to:
- Load and preprocess real financial data
- Estimate DFSV model parameters using different filters
- Filter latent states (factors and volatilities)
- Analyze and visualize the results
- Evaluate model performance

Note: This example requires pandas and yfinance packages for data handling.
Install them with: `pip install pandas yfinance`

## Running the Examples

To run any of the examples, navigate to the project root directory and execute:

```bash
python examples/01_dfsv_simulation.py
```

Replace the filename with the example you want to run.

## Dependencies

These examples require the following packages:
- numpy
- matplotlib
- jax
- jaxlib
- equinox
- optimistix
- pandas (for real data example)
- yfinance (for real data example)

Make sure you have installed all the required dependencies before running the examples.
