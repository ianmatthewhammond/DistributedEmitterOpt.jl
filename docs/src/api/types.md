# Types

Core data structures for problem definition and state management.

## Problem Definition

```@docs
FieldConfig
Environment
MaxwellProblem
```

## ObjectiveFunction Interface

```@docs
ObjectiveFunction
compute_objective
compute_adjoint_sources
explicit_sensitivity
```

## Optimization Problem

The main entry point. Use the meshfile-based constructor for most cases:

```julia
prob = OptimizationProblem(pde, objective, meshfile, solver; per_x, per_y, foundry_mode, control)
```

This automatically creates a [`SimulationBundle`](@ref) with simulations for both x and y polarizations.

```@docs
OptimizationProblem
init_uniform!
init_random!
init_from_file!
num_dofs
```

## Simulation Infrastructure

A [`Simulation`](@ref) holds the FEM infrastructure (spaces, measures, mesh). A [`SimulationBundle`](@ref) groups simulations by polarization for mixed-polarization problems.

```@docs
Simulation
build_simulation
SimulationBundle
build_simulation_bundle
default_sim
sim_for
```

## Solver Caches

Caches for LU factorizations, keyed by wavelength. [`SolverPoolBundle`](@ref) mirrors [`SimulationBundle`](@ref) for polarization-aware caching.

```@docs
AbstractSolver
UmfpackSolver
SolverCache
SolverCachePool
SolverPoolBundle
get_cache!
clear_maxwell_factors!
default_pool
pool_for
```

## Foundry Grid (2D DOF Mode)

```@docs
FoundryGrid
getgrid
pf_grid
```

## Control Parameters

```@docs
Control
tanh_projection
Threshold
```
