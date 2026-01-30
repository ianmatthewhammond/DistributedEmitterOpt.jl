# DistributedEmitterOpt.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl/dev/)

Adjoint-based topology optimization for nanophotonic SERS substrates and other distributed emission problems.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl")
```

## Usage

```julia
using DistributedEmitterOpt

# Define geometry and generate mesh
geo = SymmetricGeometry(532.0; L=200.0, W=200.0)
meshfile = "design.msh"
genmesh(geo, meshfile; per_x=false, per_y=false)

# Configure physics
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
pde = MaxwellProblem(
    env = env,
    inputs = [FieldConfig(532.0; θ=0.0, pol=:y)],
    outputs = [FieldConfig(600.0; θ=0.0, pol=:y)]
)

# Build optimization problem
prob = OptimizationProblem(pde, SERSObjective(), meshfile, UmfpackSolver();
    per_x = false,
    per_y = false,
    foundry_mode = true
)

# Run optimization
init_uniform!(prob, 0.5)
g_opt, p_opt = optimize!(prob; max_iter=100)
```

## Features

- Adjoint sensitivity analysis for efficient gradient computation
- β-continuation for smooth gray-to-binary transition
- 2D foundry mode (grid DOFs) and 3D FE mode (mesh DOFs)
- Multi-wavelength and multi-polarization support
- Anisotropic Raman polarizability tensors
- Linewidth constraints via Zygote AD

## Documentation

See the [documentation](https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl/dev/) for:
- [Architecture overview](https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl/dev/architecture.html)
- [API reference](https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl/dev/api/types.html)
- [Examples](https://ianmatthewhammond.github.io/DistributedEmitterOpt.jl/dev/examples/dielectric_2d_elastic_optimization.html)

## Dependencies

Built on [Gridap.jl](https://github.com/gridap/Gridap.jl) for finite elements and [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) for optimization.

## License

MIT
