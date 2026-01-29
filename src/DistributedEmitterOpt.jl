"""
    DistributedEmitterOpt

Topology optimization for nanophotonic SERS enhancement (Gridap FEM + adjoint).

See the README or docs for usage examples.
"""
module DistributedEmitterOpt

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

using LinearAlgebra
using SparseArrays
using Random
using Statistics: mean
using KrylovKit

using Gridap
using Gridap: Triangulation, Measure, FESpace, FEFunction
using Gridap: TestFESpace, ReferenceFE, CellField, interpolate
using Gridap: nedelec, lagrangian
using Gridap: VectorValue, ∇, ⋅, outer, num_free_dofs
using Gridap: assemble_matrix, assemble_vector
using Gridap.Geometry: DiscreteModel, get_face_labeling, get_face_mask, BoundaryTriangulation
using GridapGmsh: GmshDiscreteModel

# ---------------------------------------------------------------------------
# Types (order matters)
# ---------------------------------------------------------------------------

# Core types
include("Types/FieldConfig.jl")
include("Types/Environment.jl")
include("Types/ObjectiveFunction.jl")
include("Types/MaxwellProblem.jl")
include("Types/EigenProblem.jl")

# Infrastructure
include("Types/FoundryGrid.jl")
include("Types/Control.jl")
include("Types/SolverCache.jl")
include("Types/Simulation.jl")
include("Types/SimulationBundle.jl")

# Main container
include("Types/OptimizationProblem.jl")

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

include("Physics/Materials.jl")
include("Physics/Maxwell.jl")
include("Physics/Eigen.jl")
include("Physics/SERS.jl")
include("Physics/MaxwellSolver.jl")
include("Physics/EigenSolver.jl")

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

include("Objectives/SERSObjective.jl")
include("Objectives/EigenObjective.jl")

# ---------------------------------------------------------------------------
# TopologyOpt
# ---------------------------------------------------------------------------

include("TopologyOpt/MaterialInterp.jl")
include("TopologyOpt/Filtering2D.jl")
include("TopologyOpt/Projection2D.jl")
include("TopologyOpt/Constraints2D.jl")
include("TopologyOpt/Filtering3D.jl")
include("TopologyOpt/Projection3D.jl")
include("TopologyOpt/Constraints3D.jl")

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

include("Mesh/Meshes.jl")

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

include("Solvers/UmfpackSolver.jl")
include("Solvers/MUMPSSolver.jl")
include("Solvers/PardisoSolver.jl")

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

include("Optimization/Optimizer.jl")
include("Optimization/GradientCoordinator.jl")
include("Optimization/GradientCoordinatorEigen.jl")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

include("Visualization/Visualization.jl")
using .Visualization


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

# Core types
export FieldConfig, cache_key
export MaterialSpec, Environment, resolve_index
export ObjectiveFunction, compute_objective, compute_adjoint_sources, explicit_sensitivity
export MaxwellProblem, effective_outputs, all_configs, is_elastic
export EigenProblem
export SimulationBundle, SolverPoolBundle
export build_simulation_bundle, sim_for, pool_for, default_sim, default_pool

# Objectives
export SERSObjective
export EigenObjective, AbstractEigenObjective

# Main container
export OptimizationProblem
export EigenOptimizationProblem
export init_uniform!, init_random!, init_from_file!

# Infrastructure
export FoundryGrid, getgrid, pf_grid
export Control, tanh_projection, Threshold
export AbstractSolver, SolverCache, SolverCachePool, UmfpackSolver, MUMPSSolver, PardisoSolver
export get_cache!, clear_maxwell_factors!, clear_eigen_factors!
export Simulation, build_simulation

# Physics
export refindex, complex_index, plasmon_period
export PhysicalParams, assemble_maxwell, assemble_source
export EigenPhysicalParams, assemble_eigen_matrices

# Maxwell solver
export solve_forward!, solve_adjoint!, pde_sensitivity, build_phys_params
export solve_eigen!

# Eigen objective helpers
export compute_eigen_objective

# TopologyOpt
export filter_grid, filter_grid_adjoint
export filter_helmholtz!, filter_helmholtz_adjoint!
export project_grid, project_grid_adjoint
export project_fe, project_ssp
export christiansen_ε, ∂christiansen_ε
export glc_solid, glc_void, glc_solid_fe, glc_void_fe

# Mesh
export AbstractGeometry, SymmetricGeometry
export genmesh

# Optimization
export optimize!, objective_and_gradient!
export evaluate, test_gradient

# Visualization
export vperiodicdesign, visualize, visualizepost, visualize_new
export plot_material, plot_field, plot_substrate, get_figure_data
export combine_figures, add_text!, load_bid_parameters
export plot_directionals, plot_geometrics, plot_geometrics_one_only, plot_tolerance

end # module
