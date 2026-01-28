# DistributedEmitterOpt.jl

A Julia framework for topology optimization of nanophotonic structures using adjoint sensitivity analysis.

## Overview

This package provides a modular architecture for optimizing electromagnetic structures by solving Maxwell's equations with finite elements (Gridap.jl) and computing gradients via the adjoint method.

### Key Features

- **Adjoint-based optimization** — Efficient gradients for large parameter spaces
- **β-continuation** — Smooth transition from gray to binary designs  
- **Extensible objectives** — SERS, uLED enhancement, field localization
- **2D/3D support** — Foundry-mode (2D DOF) and full 3D optimization

## Quick Start

```julia
using DistributedEmitterOpt

# 1. Build simulation from mesh
sim = build_simulation("mesh.msh")

# 2. Create optimization problem
prob = OptimizationProblem(sim, UmfpackSolver(); 
    λ=532.0,           # Wavelength (nm)
    mat_design="Ag"    # Design material
)

# 3. Initialize and optimize
init_uniform!(prob, 0.5)
g_opt, p_opt = optimize!(prob; max_iter=100)
```

## Package Contents

```@contents
Pages = ["architecture.md", "api/types.md", "api/physics.md", "api/objectives.md", "api/topologyopt.md", "api/optimization.md"]
Depth = 2
```

## Index

```@index
```
