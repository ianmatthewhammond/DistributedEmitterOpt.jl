# API Reference

## Types

### Core Types

```julia
FieldConfig(λ; θ=0.0, pol=:y, weight=1.0)
```
Configuration for a single Maxwell solve.

```julia
Environment(; mat_design, mat_substrate="SiO2", mat_fluid=1.33)
```
Material properties. `mat_design` can be a string ("Ag", "Au") or complex refractive index.

```julia
MaxwellProblem(; env, inputs, outputs=nothing, α_loss=0.0)
```
PDE configuration. If `outputs` is `nothing`, uses elastic scattering (outputs = inputs).

```julia
SERSObjective(; αₚ=I(3), volume=true, surface=false, γ_damage=5.0, E_threshold=Inf)
```
SERS enhancement objective with optional damage model.

```julia
OptimizationProblem(sim, solver; λ, mat_design="Ag", kwargs...)
OptimizationProblem(pde, objective, sim, solver; kwargs...)
```
Main container. Two constructors: convenience (single λ) or full control.

### Infrastructure

```julia
Control(; β=8.0, η=0.5, r_filter=0.1, flag_volume=true, flag_surface=false)
```
Optimization hyperparameters.

```julia
SolverCache(; backend=:umfpack)
SolverCachePool()
```
Linear solver caching for efficient LU reuse.

```julia
Simulation(mesh_file)
build_simulation(mesh_file; kwargs...)
```
FEM domain from Gmsh file.

## Functions

### Optimization

```julia
optimize!(prob; max_iter=40, β_schedule=[8,16,...,1024], tol=1e-8)
```
Run topology optimization with β-continuation.

```julia
objective_and_gradient!(∇g, p, prob) → g
```
Compute objective and gradient at design `p`.

```julia
evaluate(prob, p) → (g, ∇g)
```
Single evaluation (no optimization).

```julia
test_gradient(prob, p; δ=1e-6) → (∇g, ∇g_fd, error)
```
Finite difference gradient check.

### Initialization

```julia
init_uniform!(prob, val)
init_random!(prob)
init_from_file!(prob, path)
```

### Maxwell Solver

```julia
solve_forward!(pde, pt, sim, pool) → Dict{Symbol, CellField}
```
Solve for all FieldConfigs, return fields keyed by `cache_key(fc)`.

```julia
solve_adjoint!(pde, sources, sim, pool) → Dict{Symbol, CellField}
```
Solve adjoint using cached LU factors.

```julia
pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control) → Vector
```
Compute ∂A/∂pf contribution to gradient.

### Objective Interface

These are the methods that objective types must implement:

```julia
compute_objective(obj, fields, pt, sim) → Float64
compute_adjoint_sources(obj, fields, pt, sim) → (bp, be)
explicit_sensitivity(obj, fields, pf, pt, sim, control) → Vector
```

### Filtering & Projection

```julia
filter_grid(p, sim, control) → pf
filter_grid_adjoint(∂g_∂pf, sim, control) → ∂g_∂p
filter_helmholtz!(p, sim, cache, control) → pf
project_ssp(pf, control) → pt
```
