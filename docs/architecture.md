# Architecture Guide

This document explains the type system and data flow in DistributedEmitterOpt.jl.

## Core Types

### MaxwellProblem

Defines **what PDEs to solve** — the physics configuration.

```julia
struct MaxwellProblem
    env::Environment              # Materials
    inputs::Vector{FieldConfig}   # Pump wavelengths
    outputs::Vector{FieldConfig}  # Emission wavelengths (or same as inputs)
    α_loss::Float64               # Absorption weighting
end
```

**Environment** encapsulates material properties:
```julia
Environment(mat_design="Ag", mat_substrate="SiO2", mat_fluid=1.33)
```

**FieldConfig** defines a single solve:
```julia
FieldConfig(λ=532.0, θ=0.0, pol=:y, weight=1.0)
```

### ObjectiveFunction

Abstract type for optimization targets. Implementations must provide:

```julia
compute_objective(obj, fields, pt, sim)          # g(E)
compute_adjoint_sources(obj, fields, pt, sim)    # ∂g/∂E
explicit_sensitivity(obj, fields, pf, pt, sim, control)  # ∂g/∂pf
```

**SERSObjective** implements SERS enhancement with the trace formula:
```julia
SERSObjective(αₚ=I(3), volume=true, surface=false, use_damage=false)
```

### OptimizationProblem

The main container that ties everything together:

```julia
mutable struct OptimizationProblem{PDE,Obj}
    pde::MaxwellProblem
    objective::ObjectiveFunction
    sim::Simulation
    pool::SolverCachePool
    control::Control
    p::Vector{Float64}      # Design parameters
    g::Float64              # Current objective
    ∇g::Vector{Float64}     # Current gradient
end
```

## Optimization Flow

```
p (design params)
    │
    ▼ filter_grid / filter_helmholtz!
pf (filtered)
    │
    ▼ project_ssp
pt (projected, binary-ish)
    │
    ▼ solve_forward! (Maxwell for each FieldConfig)
E_fields
    │
    ├──▶ compute_objective → g
    │
    ▼ compute_adjoint_sources → b_adj
    │
    ▼ solve_adjoint! (reuse LU)
λ_fields
    │
    ▼ pde_sensitivity + explicit_sensitivity
∂g/∂pf
    │
    ▼ filter_adjoint
∂g/∂p (gradient)
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `solve_forward!(pde, pt, sim, pool)` | Solve Maxwell for all FieldConfigs |
| `solve_adjoint!(pde, sources, sim, pool)` | Reuse LU factors for adjoint |
| `pde_sensitivity(...)` | Material derivative ∂A/∂pf · λ |
| `objective_and_gradient!(∇g, p, prob)` | Main entry point |
| `optimize!(prob)` | NLopt with β-continuation |

## Adding New Objectives

1. Create `Objectives/MyObjective.jl`:
```julia
struct MyObjective <: ObjectiveFunction
    # params
end

compute_objective(obj::MyObjective, fields, pt, sim) = ...
compute_adjoint_sources(obj::MyObjective, fields, pt, sim) = ...
```

2. Include in `DistributedEmitterOpt.jl` and export.

3. Create problem:
```julia
prob = OptimizationProblem(pde, MyObjective(...), sim, solver)
```

## File Organization

| Directory | Contents |
|-----------|----------|
| `Types/` | FieldConfig, Environment, MaxwellProblem, OptimizationProblem, Control, Simulation, SolverCache |
| `Physics/` | Maxwell assembly, SERS utilities, MaxwellSolver |
| `Objectives/` | SERSObjective (future: uLEDObjective) |
| `TopologyOpt/` | Filtering (Helmholtz, grid), SSP projection |
| `Optimization/` | Optimizer (NLopt), GradientCoordinator |
| `Solvers/` | UmfpackSolver (LU wrapper) |
