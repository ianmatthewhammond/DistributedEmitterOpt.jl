# DistributedEmitterOpt.jl

Topology optimization of nanophotonic structures for SERS enhancement, using adjoint sensitivity analysis with Gridap.jl finite elements.

## What it does

Solves Maxwell's equations on a finite-element mesh and optimizes the material layout to maximize a SERS objective. Gradients come from the adjoint method. The optimizer runs beta-continuation to push designs toward binary.

Supports 2D foundry-mode DOFs (grid interpolated onto the mesh) and full 3D mesh DOFs.

## How it fits together

```
OptimizationProblem
 ├── MaxwellProblem    (materials, wavelengths, angles)
 ├── ObjectiveFunction (e.g. SERSObjective)
 │    └── Environment, FieldConfig
 └── Simulation        (FEM spaces, mesh, measures)
```

## Quick start

```julia
using DistributedEmitterOpt

sim = build_simulation("mesh.msh")

prob = OptimizationProblem(sim, UmfpackSolver();
    λ=532.0,
    mat_design="Ag"
)

init_uniform!(prob, 0.5)
g_opt, p_opt = optimize!(prob; max_iter=100)
```

## Further reading

- [Architecture guide](architecture.md) -- type system and data flow
- [API reference](api.md) -- exported functions and types

## Source layout

```
src/
├── Types/              Core data structures
├── Physics/            Maxwell assembly, materials
├── Objectives/         Objective implementations
├── TopologyOpt/        Filtering, projection
├── Optimization/       NLopt optimizer, gradients
├── Mesh/               Geometry and meshing
└── Solvers/            Linear solver wrappers
```
