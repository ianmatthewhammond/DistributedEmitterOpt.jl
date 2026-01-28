# DistributedEmitterOpt.jl

A Julia framework for topology optimization of nanophotonic structures using adjoint sensitivity analysis.

## Overview

This package provides a modular architecture for optimizing electromagnetic structures by solving Maxwell's equations with finite elements (Gridap.jl) and computing gradients via the adjoint method.

### Key Features

- **Adjoint-based optimization** — Efficient gradients for large parameter spaces
- **β-continuation** — Smooth transition from gray to binary designs  
- **Extensible objectives** — SERS, uLED enhancement, field localization
- **2D/3D support** — Foundry-mode (2D DOF) and full 3D optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   OptimizationProblem                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │  MaxwellProblem │ │ ObjectiveFunction│ │  Simulation   │ │
│  │  (PDE config)   │ │ (SERSObjective) │ │  (FEM domain) │ │
│  └────────┬────────┘ └────────┬────────┘ └───────────────┘ │
│           │                   │                             │
│  ┌────────▼────────┐ ┌────────▼────────┐                   │
│  │   Environment   │ │  α̂ₚ², damage   │                   │
│  │   FieldConfig   │ │  integrand      │                   │
│  └─────────────────┘ └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
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

## Documentation

- [Architecture Guide](architecture.md) — Type system and data flow
- [API Reference](api.md) — Exported functions and types

## Directory Structure

```
src/
├── Types/              # Core data structures
├── Physics/            # Maxwell assembly, materials
├── Objectives/         # Objective implementations
├── TopologyOpt/        # Filtering, projection
├── Optimization/       # NLopt optimizer, gradients
├── Mesh/              # Geometry and meshing
└── Solvers/           # Linear solver wrappers
```
