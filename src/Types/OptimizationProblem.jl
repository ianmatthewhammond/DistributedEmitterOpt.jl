"""
    OptimizationProblem

Main container for topology optimization. Holds PDE configuration, objective
function, FEM infrastructure, solver caches, and optimization state.

This replaces the old `Objective` type with cleaner separation of concerns:
- `pde::MaxwellProblem` — PDE configuration (materials, field configs)
- `objective::ObjectiveFunction` — How to compute g from fields (SERS, uLED, etc.)
- `sim::Simulation` — FEM discretization
- `pool::SolverCachePool` — Deduplicated solver caches
- `control::Control` — Optimization hyperparameters
"""
mutable struct OptimizationProblem{PDE<:MaxwellProblem,Obj<:ObjectiveFunction}
    # Problem definition
    pde::PDE
    objective::Obj

    # FEM infrastructure
    sim::Simulation

    # DOF mode
    foundry_mode::Bool

    # Solver caches (deduplicated)
    pool::SolverCachePool

    # Control (optimization hyperparameters only)
    control::Control

    # State
    p::Vector{Float64}       # Current design
    g::Float64               # Current objective value
    ∇g::Vector{Float64}      # Current gradient

    # Metadata
    root::String             # Output directory
    iteration::Int           # Current iteration
end

Base.show(io::IO, prob::OptimizationProblem) =
    print(io, "OptimizationProblem($(typeof(prob.objective)), np=$(length(prob.p)))")

# ═══════════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    OptimizationProblem(pde, objective, sim, solver; kwargs...)

Construct optimization problem with separated PDE and objective.

## Arguments
- `pde` — MaxwellProblem (or will be constructed from kwargs)
- `objective` — ObjectiveFunction (e.g., SERSObjective)
- `sim` — Simulation container
- `solver` — AbstractSolver for cache pool

## Keyword Arguments
- `foundry_mode` — Use 2D DOFs (default: true)
- `control` — Control hyperparameters
- `root` — Output directory
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
    # Create cache pool
    pool = SolverCachePool(solver)

    # Initialize state
    np = sim.np
    p = zeros(Float64, np)
    g = 0.0
    ∇g = zeros(Float64, np)

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
        root,
        0
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    OptimizationProblem(sim, solver; λ, λ_emission=nothing, ...)

Convenience constructor for SERS optimization.
- If `λ_emission` is nothing, uses elastic scattering (emission = pump)
- If `λ_emission` is provided, uses inelastic (Stokes) scattering
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
        FieldConfig[]  # Elastic: empty outputs means reuse inputs
    else
        [FieldConfig(λ_emission, θ=θ, pol=pol)]
    end

    pde = MaxwellProblem(env=env, inputs=[FieldConfig(λ, θ=θ, pol=pol)], outputs=outputs)
    obj = SERSObjective(αₚ=αₚ)
    OptimizationProblem(pde, obj, sim, solver; kwargs...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Accessors
# ═══════════════════════════════════════════════════════════════════════════════

"""Get number of design DOFs."""
num_dofs(prob::OptimizationProblem) = length(prob.p)

"""Update iteration counter."""
next_iteration!(prob::OptimizationProblem) = (prob.iteration += 1; prob.iteration)

"""Get environment from PDE."""
environment(prob::OptimizationProblem) = prob.pde.env

"""Is this elastic scattering?"""
is_elastic(prob::OptimizationProblem) = is_elastic(prob.pde)

# ═══════════════════════════════════════════════════════════════════════════════
# Main interface (stubs — implemented in gradient coordinator)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    objective_and_gradient!(∇g, p, prob) → Float64

Unified forward+adjoint pass. This is the main entry point for optimization.
"""
function objective_and_gradient! end

# ═══════════════════════════════════════════════════════════════════════════════
# Initialization helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""Initialize design with uniform value."""
function init_uniform!(prob::OptimizationProblem, value::Float64=0.5)
    fill!(prob.p, value)
    return prob
end

"""Initialize design with random values."""
function init_random!(prob::OptimizationProblem; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    prob.p .= rand(length(prob.p))
    return prob
end

"""Initialize design from file."""
function init_from_file!(prob::OptimizationProblem, filepath::String)
    prob.p .= vec(load(filepath)["p"])
    return prob
end
