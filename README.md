# DistributedEmitterOpt.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://hammy4815.github.io/DistributedEmitterOpt.jl/dev/)

A Julia framework for topology optimization of nanophotonic structures using adjoint sensitivity analysis.

## Overview

This package provides a modular architecture for optimizing electromagnetic structures by solving Maxwell's equations with finite elements ([Gridap.jl](https://github.com/gridap/Gridap.jl)) and computing gradients via the adjoint method.

### Features

- **Adjoint-based optimization** — Efficient gradients for large parameter spaces
- **β-continuation** — Smooth transition from gray to binary designs  
- **Extensible objectives** — SERS enhancement, with easy addition of new objectives
- **2D/3D support** — Foundry-mode (2D DOF grid) and full 3D mesh optimization
- **Caching** — Deduplicated LU factorizations across wavelengths

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hammy4815/DistributedEmitterOpt.jl")
```

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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   OptimizationProblem                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │  MaxwellProblem │ │ObjectiveFunction│ │  Simulation   │ │
│  │  (PDE config)   │ │ (SERSObjective) │ │  (FEM domain) │ │
│  └────────┬────────┘ └────────┬────────┘ └───────────────┘ │
│           │                   │                             │
│           └───────────────────┴─────────────────────────────│
│                          ▼                                  │
│              GradientCoordinator (adjoint pipeline)         │
└─────────────────────────────────────────────────────────────┘
```

### Core Types

| Type | Purpose |
|------|---------|
| `MaxwellProblem` | PDE configuration (materials, wavelengths) |
| `ObjectiveFunction` | Abstract interface for optimization targets |
| `SERSObjective` | SERS enhancement via SO(3)-averaged trace formula |
| `Simulation` | FEM spaces, measures, mesh infrastructure |
| `Control` | Filtering, projection, optimization hyperparameters |

### Gradient Pipeline

```
p (design) → filter → pf → project → pt → Maxwell solve → E
                                                          ↓
∇g ← filter_adj ← explicit + pde_sensitivity ← adjoint ← ∂g/∂E
```

## Adding New Objectives

Implement the `ObjectiveFunction` interface:

```julia
struct MyObjective <: ObjectiveFunction
    # parameters
end

compute_objective(obj::MyObjective, fields, pt, sim) = ...
compute_adjoint_sources(obj::MyObjective, fields, pt, sim) = ...
```

## Directory Structure

```
src/
├── Types/           # Core data structures
├── Physics/         # Maxwell assembly, materials
├── Objectives/      # Objective implementations (SERS, ...)
├── TopologyOpt/     # Filtering, projection, constraints
├── Optimization/    # NLopt optimizer, gradient coordinator
├── Mesh/            # Geometry utilities
└── Solvers/         # Linear solver wrappers
```

## Dependencies

- [Gridap.jl](https://github.com/gridap/Gridap.jl) — Finite element framework
- [GridapGmsh.jl](https://github.com/gridap/GridapGmsh.jl) — Gmsh mesh import
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) — Nonlinear optimization

## License

MIT
