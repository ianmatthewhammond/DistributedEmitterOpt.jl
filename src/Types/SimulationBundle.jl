"""
SimulationBundle and SolverPoolBundle

Helpers to support polarization-dependent simulations (e.g. x vs y),
while sharing the same discrete model.
"""

struct SimulationBundle
    default::Simulation
    by_pol::Dict{Symbol,Simulation}
end

struct SolverPoolBundle
    default::SolverCachePool
    by_pol::Dict{Symbol,SolverCachePool}
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

default_sim(sim::Simulation) = sim
default_sim(bundle::SimulationBundle) = bundle.default

sim_for(sim::Simulation, fc::FieldConfig) = sim
sim_for(bundle::SimulationBundle, fc::FieldConfig) =
    get(bundle.by_pol, fc.polarization, bundle.default)

pool_for(pool::SolverCachePool, fc::FieldConfig) = pool
pool_for(bundle::SolverPoolBundle, fc::FieldConfig) =
    get(bundle.by_pol, fc.polarization, bundle.default)

default_pool(pool::SolverCachePool) = pool
default_pool(bundle::SolverPoolBundle) = bundle.default

"""
    map_field(field, sim, space::Symbol)

Map a CellField/FEFunction onto the target simulation's space.
Assumes both simulations share the same discrete model.
"""
function map_field(field, sim::Simulation, space::Symbol)
    if space == :U
        return interpolate(field, sim.U)
    elseif space == :Pf
        return interpolate(field, sim.Pf)
    else
        error("map_field: unsupported space $space")
    end
end

"""
    map_pt(pt, sim)

Map pt to target sim Pf space when needed.
"""
map_pt(pt, sim::Simulation) = map_field(pt, sim, :Pf)

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    build_simulation_bundle(meshfile; per_x=false, per_y=false, kwargs...)

Build a polarization-aware SimulationBundle with shared discrete model.
"""
function build_simulation_bundle(meshfile::String;
    per_x::Bool=false,
    per_y::Bool=false,
    foundry_mode::Bool=false,
    order::Int=0,
    degree::Int=4,
    sizes=nothing)

    model = GmshDiscreteModel(meshfile)
    model = repair_gmsh_model(meshfile, model)

    sim_y = build_simulation(meshfile;
        order, degree,
        dir_x=false,
        dir_y=!per_y,
        source_y=true,
        foundry_mode,
        sizes,
        model)

    sim_x = build_simulation(meshfile;
        order, degree,
        dir_x=!per_x,
        dir_y=false,
        source_y=false,
        foundry_mode,
        sizes,
        model)

    by_pol = Dict{Symbol,Simulation}(:y => sim_y, :x => sim_x)
    SimulationBundle(sim_y, by_pol)
end

"""
    SolverPoolBundle(solver, sims)

Create a pool bundle keyed by polarization for the given sims.
"""
function SolverPoolBundle(solver::AbstractSolver, sims::SimulationBundle)
    by_pol = Dict{Symbol,SolverCachePool}()
    for (pol, _) in sims.by_pol
        by_pol[pol] = SolverCachePool(solver)
    end
    default = get(by_pol, :y, first(values(by_pol)))
    SolverPoolBundle(default, by_pol)
end

"""
    clear_maxwell_factors!(pool)

Clear LU factors for single or bundled pools.
"""
function clear_maxwell_factors!(bundle::SolverPoolBundle)
    clear_maxwell_factors!(bundle.default)
    for pool in values(bundle.by_pol)
        clear_maxwell_factors!(pool)
    end
end
