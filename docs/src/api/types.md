# Types

Core data structures for problem definition and state management.

## Problem Definition

```@docs
FieldConfig
cache_key
MaterialSpec
Environment
resolve_index
MaxwellProblem
effective_outputs
all_configs
is_elastic
```

## ObjectiveFunction Interface

```@docs
ObjectiveFunction
compute_objective
compute_adjoint_sources
explicit_sensitivity
```

## Main Container

```@docs
OptimizationProblem
init_uniform!
init_random!
init_from_file!
```

## Infrastructure

```@docs
FoundryGrid
getgrid
pf_grid
Control
tanh_projection
Threshold
AbstractSolver
SolverCache
SolverCachePool
UmfpackSolver
get_cache!
clear_maxwell_factors!
Simulation
build_simulation
```
