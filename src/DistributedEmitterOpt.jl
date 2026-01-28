"""
    DistributedEmitterOpt

Topology optimization for nanophotonic SERS enhancement.

## Main exports
- `SERSProblem` — SERS physics configuration
- `Simulation` — FEM infrastructure
- `Objective` — Optimization container
- `optimize!` — Run β-continuation optimization

## Quick start
```julia
using DistributedEmitterOpt

# Define physics
physics = SERSProblem(λ_pump=532.0, λ_emission=532.0, mat_design="Ag")

# Build simulation from mesh
sim = build_simulation("mesh.msh"; foundry_mode=true)

# Create solver
cache = SolverCache(UmfpackSolver())

# Build objective
obj = Objective(physics, sim, cache; control=Control(β=8.0))

# Initialize and optimize
init_random!(obj)
optimize!(obj; max_iter=40)
```
"""
module DistributedEmitterOpt

# ═══════════════════════════════════════════════════════════════════════════════
# Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra
using SparseArrays
using Random
using Statistics: mean

# Gridap (conditional — for FEM)
# These should be loaded by the user's project
using Gridap
using Gridap: Triangulation, Measure, FESpace, FEFunction
using Gridap: TestFESpace, ReferenceFE, CellField, interpolate
using Gridap: nedelec, lagrangian
using Gridap: VectorValue, ∇, ⋅, outer, num_free_dofs
using Gridap: assemble_matrix, assemble_vector
using Gridap.Geometry: DiscreteModel, get_face_labeling, get_face_mask, BoundaryTriangulation
using GridapGmsh: GmshDiscreteModel

# ═══════════════════════════════════════════════════════════════════════════════
# Type includes (order matters)
# ═══════════════════════════════════════════════════════════════════════════════

# Core types
include("Types/FieldConfig.jl")
include("Types/Environment.jl")
include("Types/ObjectiveFunction.jl")
include("Types/MaxwellProblem.jl")

# Infrastructure
include("Types/FoundryGrid.jl")
include("Types/Control.jl")
include("Types/SolverCache.jl")
include("Types/Simulation.jl")

# Main container
include("Types/OptimizationProblem.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Physics
# ═══════════════════════════════════════════════════════════════════════════════

include("Physics/Materials.jl")
include("Physics/Maxwell.jl")
include("Physics/SERS.jl")
include("Physics/MaxwellSolver.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Objectives (new architecture)
# ═══════════════════════════════════════════════════════════════════════════════

include("Objectives/SERSObjective.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# TopologyOpt
# ═══════════════════════════════════════════════════════════════════════════════

include("TopologyOpt/MaterialInterp.jl")
include("TopologyOpt/Filtering2D.jl")
include("TopologyOpt/Projection2D.jl")
include("TopologyOpt/Constraints2D.jl")
include("TopologyOpt/Filtering3D.jl")
include("TopologyOpt/Projection3D.jl")
include("TopologyOpt/Constraints3D.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh
# ═══════════════════════════════════════════════════════════════════════════════

include("Mesh/Meshes.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Solvers
# ═══════════════════════════════════════════════════════════════════════════════

include("Solvers/UmfpackSolver.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Optimization
# ═══════════════════════════════════════════════════════════════════════════════

include("Optimization/Optimizer.jl")
include("Optimization/GradientCoordinator.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

# Core types
export FieldConfig, cache_key
export MaterialSpec, Environment, resolve_index
export ObjectiveFunction, compute_objective, compute_adjoint_sources, explicit_sensitivity
export MaxwellProblem, effective_outputs, all_configs, is_elastic

# Objectives
export SERSObjective

# Main container
export OptimizationProblem
export init_uniform!, init_random!, init_from_file!

# Infrastructure
export FoundryGrid, getgrid, pf_grid
export Control, tanh_projection, Threshold
export AbstractSolver, SolverCache, SolverCachePool, UmfpackSolver
export get_cache!, clear_maxwell_factors!
export Simulation, build_simulation

# Physics
export refindex, complex_index, plasmon_period
export PhysicalParams, assemble_maxwell, assemble_source

# Maxwell solver
export solve_forward!, solve_adjoint!, pde_sensitivity, build_phys_params

# TopologyOpt
export filter_grid, filter_grid_adjoint
export filter_helmholtz!, filter_helmholtz_adjoint!
export project_grid, project_grid_adjoint
export project_fe, project_ssp
export christiansen_ε, ∂christiansen_ε
export glc_solid, glc_void, glc_solid_fe, glc_void_fe

# Mesh
export AbstractGeometry, SymmetricGeometry
export genperiodic

# Optimization
export optimize!, objective_and_gradient!
export evaluate, test_gradient

end # module
