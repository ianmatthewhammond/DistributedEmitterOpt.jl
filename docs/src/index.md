# DistributedEmitterOpt.jl

Topology optimization of nanophotonic structures for SERS enhancement, using adjoint sensitivity analysis with Gridap.jl finite elements.

## What it does

Solves Maxwell's equations on a finite-element mesh and optimizes the material layout to maximize a SERS objective. Gradients come from the adjoint method. Beta-continuation pushes designs toward binary.

Supports 2D foundry-mode DOFs (grid interpolated onto the mesh) and full 3D mesh DOFs.

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

## Contents

```@contents
Pages = ["architecture.md", "api/types.md", "api/physics.md", "api/objectives.md", "api/topologyopt.md", "api/optimization.md"]
Depth = 2
```

## Index

```@index
```
