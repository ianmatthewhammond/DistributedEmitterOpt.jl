# Architecture

How the types and data flow are organized.

## Types

### MaxwellProblem

Specifies the physics: which PDEs to solve.

```julia
struct MaxwellProblem
    env::Environment              # materials
    inputs::Vector{FieldConfig}   # pump wavelengths
    outputs::Vector{FieldConfig}  # emission wavelengths (or same as inputs)
    α_loss::Float64               # absorption weighting
end
```

`Environment` holds material properties:
```julia
Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
```

`FieldConfig` describes a single solve:
```julia
FieldConfig(λ=532.0, θ=0.0, pol=:y, weight=1.0)
```

### ObjectiveFunction

Abstract type. Implementations must provide three methods:

```julia
compute_objective(obj, pde, fields, pt, sim)          # g(E)
compute_adjoint_sources(obj, pde, fields, pt, sim)    # dg/dE
explicit_sensitivity(obj, pde, fields, pf, pt, sim, control)  # dg/dpf
```

`SERSObjective` implements this for the SO(3)-averaged trace formula:
```julia
SERSObjective(αp=I(3), volume=true, surface=false, use_damage_model=false)
```

### OptimizationProblem

Ties everything together:

```julia
mutable struct OptimizationProblem{PDE,Obj}
    pde::MaxwellProblem
    objective::ObjectiveFunction
    sim::Simulation
    foundry_mode::Bool
    pool::SolverCachePool
    control::Control
    p::Vector{Float64}      # design parameters
    g::Float64              # current objective
    grad::Vector{Float64}   # current gradient
end
```

## Optimization flow

```
p (design params)
    |
    v  filter_grid (foundry) / filter_helmholtz! (3D)
pf (filtered)
    |
    v  interpolate grid->mesh (foundry only)
    v  project_ssp
pt (projected, near-binary)
    |
    v  solve_forward! (Maxwell for each FieldConfig)
E_fields
    |
    |-->  compute_objective -> g
    |
    v  compute_adjoint_sources -> b_adj
    |
    v  solve_adjoint! (reuses the LU factors)
lambda_fields
    |
    v  pde_sensitivity + explicit_sensitivity
dg/dpf
    |
    v  filter_adjoint
dg/dp (gradient)
```

## Key functions

| Function | What it does |
|----------|--------------|
| `solve_forward!(pde, pt, sim, pool)` | Solve Maxwell for all FieldConfigs |
| `solve_adjoint!(pde, sources, sim, pool)` | Reuse LU factors for the adjoint |
| `pde_sensitivity(...)` | Material derivative lambda^T dA/dp E |
| `objective_and_gradient!(grad, p, prob)` | Main entry point |
| `optimize!(prob)` | NLopt with beta-continuation |

## Writing a new objective

1. Create `Objectives/MyObjective.jl`:
```julia
struct MyObjective <: ObjectiveFunction
    # params
end

compute_objective(obj::MyObjective, pde, fields, pt, sim) = ...
compute_adjoint_sources(obj::MyObjective, pde, fields, pt, sim) = ...
explicit_sensitivity(obj::MyObjective, pde, fields, pf, pt, sim, control) = ...
```

2. Include it in `DistributedEmitterOpt.jl` and export.

3. Use it:
```julia
prob = OptimizationProblem(pde, MyObjective(...), sim, solver)
```

## File organization

| Directory | What's in it |
|-----------|-------------|
| `Types/` | FieldConfig, Environment, MaxwellProblem, OptimizationProblem, Control, Simulation, SolverCache |
| `Physics/` | Maxwell assembly, SERS utilities, MaxwellSolver |
| `Objectives/` | SERSObjective |
| `TopologyOpt/` | Filtering (Helmholtz, grid), SSP projection |
| `Optimization/` | Optimizer (NLopt), GradientCoordinator |
| `Solvers/` | UmfpackSolver (LU wrapper) |
