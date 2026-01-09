
# API Comparison: BellmanFilterDFSV (v1 vs v2)

This document compares the usage of the legacy (v1) API with the new, refactored (v2) API.

## 1. Basic Filtering

### Legacy (v1) - The "Inheritance Maze"
In v1, you had to instantiate a complex class hierarchy. The parameters were often mixed with filter configuration, and type hints were unreliable.

```python
# v1: Complex setup, ambiguous types
from bellman_filter_dfsv.core.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.core.models.dfsv import DFSVParamsDataclass

# Setup parameters (verbose dataclass)
params = DFSVParamsDataclass(
    lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, 
    mu=mu, sigma2=sigma2, Q_h=Q_h, N=N, K=K
)

# Initialize filter (Note: 'N' and 'K' passed again, redundant)
bf = DFSVBellmanInformationFilter(N=N, K=K)

# Run filter (returns tuple of numpy arrays, but internal logic is JAX)
# Confusing: is 'filtered_states' a property or return value? Both.
filtered_states, filtered_covs, log_lik = bf.filter(params, observations)

# Accessing results often required getters or property access on 'bf'
# resulting in potential side-effect confusion if filter() was called multiple times.
states = bf.get_filtered_states() 
```

### New (v2) - The "Functional Equinox" Way
In v2, the filter is a clean `eqx.Module`. Parameters are the source of truth. The API is functional and stateless in usage (you get a result object back).

```python
# v2: Clean, functional, strict types
from bellman_filter_dfsv.next import BellmanFilter, DFSVParams

# Setup parameters (NamedTuple, lighter weight)
params = DFSVParams(
    lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, 
    mu=mu, sigma2=sigma2, Q_h=Q_h
    # Note: N and K are inferred from shapes! No redundancy.
)

# Initialize filter with params directly
bf = BellmanFilter(params)

# Run filter (Returns a structured FilterResult object)
result = bf.filter(observations)

# Access results directly with type safety
print(result.means.shape)  # (T, 2K)
print(result.log_likelihood)
# No side effects on 'bf' object. 'bf' is immutable (Equinox pattern).
```

## 2. Particle Filtering

### Legacy (v1)
The particle filter in v1 was awkward because it inherited from the base `DFSVFilter` which was designed for Kalman-like filters. It had unused methods and arguments.

```python
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter

# Had to pass N, K explicitly
pf = DFSVParticleFilter(N=N, K=K, num_particles=1000)

# 'filter' method signature had to match base class, leading to awkwardness
# Internal state management was messy (self.particles, self.weights updated in-place)
states, covs, ll = pf.filter(params, observations)
```

### New (v2)
The v2 Particle Filter is its own distinct Equinox module, sharing the same consistent interface but without forced inheritance baggage.

```python
from bellman_filter_dfsv.next import ParticleFilter

# Parameters define the model structure
pf = ParticleFilter(params, num_particles=1000)

# Run returns a ParticleFilterResult
result = pf.filter(observations)

# Clean access to particle-specific outputs if needed (can be added to result)
# result.means, result.covs are standard
```

## 3. Smoothing (The Biggest Win)

### Legacy (v1)
Smoothing was "bolted on". You had to run the filter first, which stored state internally in the object, and *then* call smooth. If you lost the object or re-ran filter, smoothing could break or smooth the wrong thing.

```python
bf.filter(params, obs)
# Implicit dependency on previous call!
smoothed_states, _, _ = bf.smooth(params) 
```

### New (v2)
Smoothing is a pure function. You pass the filter results into the smoother. Zero ambiguity.

```python
from bellman_filter_dfsv.next import rts_smoother

# Explicit dependency: input -> output
filter_result = bf.filter(observations)

# Smooth using the result from the filter
smooth_result = rts_smoother(params, filter_result.means, filter_result.infos)

# smooth_result is a standalone object
```

## Summary of Improvements

| Feature | v1 (Legacy) | v2 (New) |
| :--- | :--- | :--- |
| **State Management** | Mutable objects, side effects (methods change `self.x`) | Immutable `eqx.Module` & Pure Functions |
| **Parameter Handling** | Redundant (pass N, K separately), messy dataclass | Shape-inferred `NamedTuple`, single source of truth |
| **Type Safety** | 90+ errors, confusing `np.ndarray` vs `jax.Array` | Strict `jaxtyping`, 0 errors, clear contracts |
| **Smoothing** | Stateful, implicit dependency on `filter()` | Pure functional transformation of `FilterResult` |
| **JAX Integration** | Fighting against OOP, `verbose=True` breaks JIT | Native JAX/Equinox, fully JIT-compatible |

The v2 API is significantly "cleaner" because it aligns with how JAX works: **Data in, Data out**. You don't have to worry about the internal state of the filter object.
