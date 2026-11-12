"""
SimulationBundle and SolverPoolBundle

Helpers to support polarization-dependent simulations (e.g. x vs y),
while sharing the same discrete model.
"""

"""
    SimulationBundle

Container for multiple `Simulation` objects (per polarization), sharing the same
underlying discrete model.

# Fields
- `default::Simulation`: The primary simulation (usually y-polarized)
- `by_pol::Dict{Symbol,Simulation}`: Sims keyed by polarization (:x, :y)
"""
struct SimulationBundle
    default::Simulation
    by_pol::Dict{Symbol,Simulation}
end

"""
    SolverPoolBundle

Container for multiple `SolverCachePool` objects (per polarization), corresponding
to the `Simulation` objects in a bundle.

# Fields
- `default::SolverCachePool`: The primary pool
- `by_pol::Dict{Symbol,SolverCachePool}`: Pools keyed by polarization (:x, :y)
"""
struct SolverPoolBundle
    default::SolverCachePool
    by_pol::Dict{Symbol,SolverCachePool}
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
    default_sim(sim_or_bundle)

Get the default `Simulation` from a Simulation or SimulationBundle.
"""
default_sim(sim::Simulation) = sim
default_sim(bundle::SimulationBundle) = bundle.default

"""
    sim_for(sim_or_bundle, fc::FieldConfig)

Get the appropriate `Simulation` for a given `FieldConfig` (based on polarization).
"""
sim_for(sim::Simulation, fc::FieldConfig) = sim
sim_for(bundle::SimulationBundle, fc::FieldConfig) =
    get(bundle.by_pol, fc.polarization, bundle.default)

"""
    pool_for(pool_or_bundle, fc::FieldConfig)

Get the appropriate `SolverCachePool` for a given `FieldConfig`.
"""
pool_for(pool::SolverCachePool, fc::FieldConfig) = pool
pool_for(bundle::SolverPoolBundle, fc::FieldConfig) =
    get(bundle.by_pol, fc.polarization, bundle.default)

"""
    default_pool(pool_or_bundle)

Get the default `SolverCachePool` from a pool or bundle.
"""
default_pool(pool::SolverCachePool) = pool
default_pool(bundle::SolverPoolBundle) = bundle.default

"""
    map_field(field, sim, space::Symbol)

Use a CellField/FEFunction on another simulation of the same model, retaining
the original field without a constrained-space or material projection.
Different-model transfers require a separate forward/adjoint lifting operator.
"""
function map_field(field, sim::Simulation, space::Symbol)
    if space == :U
        # Different mirror parities constrain different Nedelec DOFs. Projecting
        # a physical field into the other polarization's constrained space changes
        # the objective and requires a transpose transfer in its derivative.
        # On the shared model no transfer is needed: integrate the original field
        # and differentiate it against its own source basis.
        Gridap.CellData.get_triangulation(field).model === sim.model ||
            throw(ArgumentError("map_field: U fields require a shared model; multi-mesh lifting is unsupported"))
        return field
    elseif space == :Pf
        # Projected material is generally nonlinear inside each element.
        # Nodal interpolation would alter it in the second polarization's PDE,
        # while the sensitivity would still differentiate the original projection.
        if field isa CellField
            Gridap.CellData.get_triangulation(field).model === sim.model ||
                throw(ArgumentError("map_field: Pf fields require a shared model"))
        elseif !(field isa Function)
            throw(ArgumentError("map_field: unsupported scalar field representation"))
        end
        return field
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
    sizes=nothing,
    bot_PEC::Bool=false)

    model = GmshDiscreteModel(meshfile)
    model = repair_gmsh_model(meshfile, model)

    sim_y = build_simulation(meshfile;
        order, degree,
        dir_x=false,
        dir_y=!per_y,
        source_y=true,
        foundry_mode,
        sizes,
        model,
        bot_PEC=bot_PEC)

    sim_x = build_simulation(meshfile;
        order, degree,
        dir_x=!per_x,
        dir_y=false,
        source_y=false,
        foundry_mode,
        sizes,
        model,
        bot_PEC=bot_PEC)

    # Scalar spaces have no polarization-dependent boundary constraints.
    # Sharing them also lets the tv_void shell use one material geometry.
    sim_x.P = sim_y.P
    sim_x.Pf = sim_y.Pf

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
