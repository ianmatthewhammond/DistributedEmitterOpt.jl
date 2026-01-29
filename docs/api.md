# API Reference

## Types

### Problem setup

```julia
FieldConfig(λ; θ=0.0, pol=:y, weight=1.0)
```
A single Maxwell solve configuration: wavelength, incidence angle, polarization, weight.

```julia
Environment(; mat_design, mat_substrate=mat_design, mat_fluid=1.33)
```
Material properties. Each material can be a string (looked up from `data/materials/*.txt`) or a numeric refractive index.

```julia
MaxwellProblem(; env, inputs, outputs=FieldConfig[], α_loss=0.0)
```
PDE configuration. Leave `outputs` empty for elastic scattering (reuses `inputs`).

Convenience constructors:
```julia
MaxwellProblem(λ; θ=0.0, pol=:y, mat_design="Ag")
MaxwellProblem(λ_pump, λ_emission; θ=0.0, pol=:y, mat_design="Ag")
```

```julia
SERSObjective(; αₚ=I(3), volume=true, surface=false,
               use_damage_model=false, γ_damage=1.0, E_threshold=Inf)
```
SERS enhancement objective using the SO(3)-averaged trace formula.

```julia
OptimizationProblem(sim, solver; λ, λ_emission=nothing, kwargs...)
OptimizationProblem(pde, objective, sim, solver; foundry_mode=true, control=Control(), root="./data/")
```
Main container. Use the short form for quick setup or spell out the PDE and objective yourself.

### Infrastructure

```julia
Control(; use_filter=true, R_filter=(20.0,20.0,20.0), use_dct=true,
         use_projection=true, β=8.0, η=0.5, use_ssp=false,
         use_constraints=false, η_erosion=0.75, η_dilation=0.25, b1=1e-8,
         use_damage=false, γ_damage=1.0, E_threshold=Inf)
```
Hyperparameters for filtering, projection, SSP, and linewidth constraints.

```julia
SolverCache(solver)
SolverCachePool(solver)
UmfpackSolver()
```
Linear solver caching for LU reuse across field configurations.

```julia
Simulation()
build_simulation(meshfile; kwargs...)
```
FEM domain setup from a Gmsh `.msh` file.

## Functions

### Optimization

```julia
optimize!(prob; max_iter=40, β_schedule=[8,16,...,1024], tol=1e-8)
objective_and_gradient!(∇g, p, prob) → g
```
Run beta-continuation optimization, or evaluate objective + gradient once.

```julia
evaluate(prob, p) → (g, ∇g)
test_gradient(prob, p; δ=1e-6) → (∇g, ∇g_fd, error)
```
Single evaluation and finite-difference gradient check.

### Initialization

```julia
init_uniform!(prob, val)
init_random!(prob; seed=nothing)
init_from_file!(prob, path)
```

### Maxwell solver

```julia
solve_forward!(pde, pt, sim, pool) → Dict{CacheKey, CellField}
solve_adjoint!(pde, sources, sim, pool) → Dict{CacheKey, CellField}
pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control) → Vector
build_phys_params(fc, env, sim; α=0.0)
```

### Objective interface

Any objective type must implement these three methods:

```julia
compute_objective(obj, pde, fields, pt, sim) → Float64
compute_adjoint_sources(obj, pde, fields, pt, sim) → Dict{CacheKey, Vector}
explicit_sensitivity(obj, pde, fields, pf, pt, sim, control) → Vector
```

### Filtering and projection

```julia
filter_grid(p, sim, control) → pf
filter_grid_adjoint(∂g_∂pf, sim, control) → ∂g_∂p
filter_helmholtz!(p, cache, sim, control) → pf
filter_helmholtz_adjoint!(∂g_∂pf, cache, sim, control) → ∂g_∂p

project_grid(pf, sim, control) → pt
project_grid_adjoint(∂g_∂pt, pf, control) → ∂g_∂pf
project_ssp(pf, control) → pt
project_fe(pf_vec, sim, control) → pt
```
