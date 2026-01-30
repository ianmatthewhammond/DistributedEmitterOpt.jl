# DistributedEmitterOpt.jl

Topology optimization of nanophotonic structures for SERS enhancement, using adjoint sensitivity analysis with Gridap.jl finite elements.

## What it does

Solves Maxwell's equations on a finite-element mesh and optimizes the material layout to maximize a SERS objective. Gradients come from the adjoint method. Beta-continuation pushes designs toward binary.

Supports 2D foundry-mode DOFs (grid interpolated onto the mesh) and full 3D mesh DOFs.

## Quick start

```julia
using DistributedEmitterOpt

# 1. Define geometry and generate mesh
geo = SymmetricGeometry(532.0; L=200.0, W=200.0)
meshfile = "design.msh"
genmesh(geo, meshfile; per_x=false, per_y=false)

# 2. Configure physics
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
pde = MaxwellProblem(
    env = env,
    inputs = [FieldConfig(532.0; θ=0.0, pol=:y)],
    outputs = [FieldConfig(600.0; θ=0.0, pol=:y)]  # Inelastic
)

# 3. Build optimization problem (meshfile constructor)
prob = OptimizationProblem(pde, SERSObjective(), meshfile, UmfpackSolver();
    per_x = false,
    per_y = false,
    foundry_mode = true
)

# 4. Run optimization
init_uniform!(prob, 0.5)
g_opt, p_opt = optimize!(prob; max_iter=100)
```

## Key concepts

### SimulationBundle

When you use the meshfile-based `OptimizationProblem` constructor, it internally creates a [`SimulationBundle`](@ref) — a collection of `Simulation` objects for different polarizations. This allows mixed-polarization optimization (e.g., pump at y-pol, emission at x-pol).

```julia
# Meshfile constructor automatically creates SimulationBundle
prob = OptimizationProblem(pde, objective, meshfile, solver; per_x, per_y, ...)

# Access underlying simulation
sim = default_sim(prob.sim)  # Gets the default (y-polarized) simulation
```

### DOF Modes

- **Foundry mode** (`foundry_mode=true`): Design DOFs live on a 2D grid, interpolated onto the 3D mesh. Faster, suitable for planar structures.
- **3D FE mode** (`foundry_mode=false`): Design DOFs are mesh elements directly. Full 3D topology optimization.

## Contents

```@contents
Pages = ["architecture.md", "api/types.md", "api/physics.md", "api/objectives.md", "api/topologyopt.md", "api/optimization.md"]
Depth = 2
```

## Index

```@index
```
