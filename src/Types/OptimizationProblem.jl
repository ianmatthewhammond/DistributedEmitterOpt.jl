"""
    OptimizationProblem

Main container for topology optimization. Holds the PDE config, objective,
FEM infrastructure, solver caches, and current optimization state.
"""
mutable struct OptimizationProblem{PDE<:MaxwellProblem,Obj<:ObjectiveFunction}
    # Problem definition
    pde::PDE
    objective::Obj

    # FEM infrastructure
    sim::Union{Simulation,SimulationBundle}

    # DOF mode
    foundry_mode::Bool

    # Solver caches (deduplicated across field configs)
    pool::Union{SolverCachePool,SolverPoolBundle}

    # Hyperparameters
    control::Control

    # State
    p::Vector{Float64}       # current design
    g::Float64               # current objective value
    ∇g::Vector{Float64}      # current gradient
    g_history::Vector{Float64} # history of objective values

    # Metadata
    root::String             # output directory
    iteration::Int           # iteration counter
end

Base.show(io::IO, prob::OptimizationProblem) =
    print(io, "OptimizationProblem($(typeof(prob.objective)), np=$(length(prob.p)))")

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    OptimizationProblem(pde, objective, sim, solver; kwargs...)

Build an optimization problem from a PDE definition and objective.

Arguments:
- `pde` -- MaxwellProblem
- `objective` -- ObjectiveFunction (e.g. SERSObjective)
- `sim` -- Simulation
- `solver` -- AbstractSolver for cache pool

Keyword arguments:
- `foundry_mode` -- use 2D DOFs (default true)
- `control` -- optimization hyperparameters
- `root` -- output directory
"""
function OptimizationProblem(
    pde::MaxwellProblem,
    objective::ObjectiveFunction,
    sim::Simulation,
    solver::AbstractSolver;
    foundry_mode::Bool=true,
    control::Control=Control(),
    root::String="./data/"
)
    pool = SolverCachePool(solver)

    np = sim.np
    p = zeros(Float64, np)
    g = 0.0
    ∇g = zeros(Float64, np)
    g_history = Float64[]

    OptimizationProblem(
        pde,
        objective,
        sim,
        foundry_mode,
        pool,
        control,
        p,
        g,
        ∇g,
        g_history,
        root,
        0
    )
end

function OptimizationProblem(
    pde::MaxwellProblem,
    objective::ObjectiveFunction,
    sim::SimulationBundle,
    solver::AbstractSolver;
    foundry_mode::Bool=true,
    control::Control=Control(),
    root::String="./data/"
)
    pool = SolverPoolBundle(solver, sim)

    np = default_sim(sim).np
    p = zeros(Float64, np)
    g = 0.0
    ∇g = zeros(Float64, np)
    g_history = Float64[]

    OptimizationProblem(
        pde,
        objective,
        sim,
        foundry_mode,
        pool,
        control,
        p,
        g,
        ∇g,
        g_history,
        root,
        0
    )
end

function OptimizationProblem(
    pde::MaxwellProblem,
    objective::ObjectiveFunction,
    meshfile::String,
    solver::AbstractSolver;
    per_x::Bool=false,
    per_y::Bool=false,
    foundry_mode::Bool=false,
    order::Int=0,
    degree::Int=4,
    control::Control=Control(),
    root::String="./data/"
)
    sim = build_simulation_bundle(meshfile;
        per_x,
        per_y,
        foundry_mode,
        order,
        degree
    )
    OptimizationProblem(pde, objective, sim, solver; foundry_mode, control, root)
end

# ---------------------------------------------------------------------------
# Eigen optimization problem
# ---------------------------------------------------------------------------

"""
    EigenOptimizationProblem

Container for eigenvalue-based optimization. Mirrors OptimizationProblem but
uses EigenProblem + eigen objectives.
"""
mutable struct EigenOptimizationProblem{Obj<:AbstractEigenObjective}
    # Problem definition
    pde::EigenProblem
    objective::Obj

    # FEM infrastructure
    sim::Simulation

    # DOF mode
    foundry_mode::Bool

    # Solver caches
    pool::SolverCachePool

    # Hyperparameters
    control::Control

    # State
    p::Vector{Float64}
    g::Float64
    ∇g::Vector{Float64}

    # Metadata
    root::String
    iteration::Int
end

Base.show(io::IO, prob::EigenOptimizationProblem) =
    print(io, "EigenOptimizationProblem($(typeof(prob.objective)), np=$(length(prob.p)))")

"""
    EigenOptimizationProblem(pde, objective, sim, solver; kwargs...)

Build an eigen optimization problem.
"""
function EigenOptimizationProblem(
    pde::EigenProblem,
    objective::AbstractEigenObjective,
    sim::Simulation,
    solver::AbstractSolver;
    foundry_mode::Bool=true,
    control::Control=Control(),
    root::String="./data/"
)
    pool = SolverCachePool(solver)
    np = sim.np
    p = zeros(Float64, np)
    g = 0.0
    ∇g = zeros(Float64, np)

    EigenOptimizationProblem(
        pde,
        objective,
        sim,
        foundry_mode,
        pool,
        control,
        p,
        g,
        ∇g,
        root,
        0
    )
end

# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

"""
    OptimizationProblem(sim, solver; λ, λ_emission=nothing, ...)

Shorthand for SERS optimization. If `λ_emission` is nothing, uses elastic
scattering; otherwise inelastic (Stokes).
"""
function OptimizationProblem(
    sim::Simulation,
    solver::AbstractSolver;
    λ::Float64,
    λ_emission::Union{Float64,Nothing}=nothing,
    θ::Float64=0.0,
    pol::Symbol=:y,
    mat_design::MaterialSpec="Ag",
    mat_fluid::MaterialSpec=1.33,
    αₚ::Matrix{ComplexF64}=Matrix{ComplexF64}(LinearAlgebra.I, 3, 3),
    kwargs...
)
    env = Environment(mat_design=mat_design, mat_fluid=mat_fluid)

    outputs = if isnothing(λ_emission)
        FieldConfig[]  # elastic: empty outputs means reuse inputs
    else
        [FieldConfig(λ_emission, θ=θ, pol=pol)]
    end

    pde = MaxwellProblem(env=env, inputs=[FieldConfig(λ, θ=θ, pol=pol)], outputs=outputs)
    obj = SERSObjective(αₚ=αₚ)
    OptimizationProblem(pde, obj, sim, solver; kwargs...)
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

"""Number of design DOFs."""
num_dofs(prob::OptimizationProblem) = length(prob.p)

"""Increment and return the iteration counter."""
next_iteration!(prob::OptimizationProblem) = (prob.iteration += 1; prob.iteration)

"""Number of design DOFs (eigen optimization)."""
num_dofs(prob::EigenOptimizationProblem) = length(prob.p)

"""Increment and return the iteration counter (eigen optimization)."""
next_iteration!(prob::EigenOptimizationProblem) = (prob.iteration += 1; prob.iteration)

"""Get environment from the PDE."""
environment(prob::OptimizationProblem) = prob.pde.env

"""Is this elastic scattering?"""
is_elastic(prob::OptimizationProblem) = is_elastic(prob.pde)

# ---------------------------------------------------------------------------
# Main interface (implemented in GradientCoordinator)
# ---------------------------------------------------------------------------

"""
    objective_and_gradient!(grad, p, prob) -> Float64

Forward + adjoint pass. Main entry point for optimization.
"""
function objective_and_gradient! end

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

"""Set all design parameters to `value`."""
function init_uniform!(prob::OptimizationProblem, value::Float64=0.5)
    fill!(prob.p, value)
    return prob
end

"""
Randomize design parameters using legacy initialization behavior.

Legacy `rand` init used a filtered random field with fixed radius 20 nm:
- Foundry mode: conic filter + sharp tanh projection (β=1028, η=0.5)
- 3D mode: Helmholtz filter solve only
"""
function init_random!(prob::OptimizationProblem; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    sim0 = default_sim(prob.sim)
    p_rand = rand(length(prob.p))
    legacy_R = (20.0, 20.0, 20.0)

    if prob.foundry_mode
        # Match legacy foundry initrandom: filtered random + near-binary projection.
        ctrl = deepcopy(prob.control)
        ctrl.use_filter = true
        ctrl.R_filter = legacy_R
        pf_vec = filter_grid(p_rand, sim0, ctrl)
        prob.p .= tanh_projection.(pf_vec, 0.5, 1028.0)
    else
        # Match legacy 3D initrandom: Helmholtz-filtered random field.
        ctrl = deepcopy(prob.control)
        ctrl.use_filter = true
        ctrl.R_filter = legacy_R
        pool0 = default_pool(prob.pool)
        cache = pool0.filter_cache
        pf_vec = filter_helmholtz!(p_rand, cache, sim0, ctrl)
        pf_field = FEFunction(sim0.Pf, pf_vec)
        M = assemble_matrix(sim0.P, sim0.P) do u, v
            ∫(v * u)sim0.dΩ
        end
        b = assemble_vector(sim0.P) do v
            ∫(v * pf_field)sim0.dΩ
        end
        prob.p .= M \ b
        prob.p .= clamp.(prob.p, 0.0, 1.0)

        # Avoid stale filter factor when optimization uses a different filter radius.
        if prob.control.R_filter != legacy_R && !isnothing(cache.F_factor)
            release_factor!(cache.solver, cache.F_factor)
            cache.F_factor = nothing
        end
    end
    return prob
end

"""Load design parameters from a file."""
function init_from_file!(prob::OptimizationProblem, filepath::String)
    prob.p .= vec(load(filepath)["p"])
    return prob
end
