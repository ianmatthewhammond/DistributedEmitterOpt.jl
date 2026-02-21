"""
    MaxwellSolver

Forward and adjoint Maxwell solves, plus PDE sensitivity assembly.
Connects the problem definition (FieldConfig, Environment) to the
low-level assembly routines in Maxwell.jl.
"""

# ---------------------------------------------------------------------------
# PhysicalParams from FieldConfig + Environment
# ---------------------------------------------------------------------------

"""
    build_phys_params(fc, env, sim; α=0.0)

Build PhysicalParams for a given FieldConfig and Environment.
"""
function build_phys_params(fc::FieldConfig, env::Environment, sim; α::Float64=0.0)
    λ = fc.λ
    ω = 2π / λ

    nf = resolve_index(env.mat_fluid, λ)
    nm = resolve_index(env.mat_design, λ)
    ns = resolve_index(env.mat_substrate, λ)

    des_low, des_high = getdesignz(sim.labels, sim.Ω)

    PhysicalParams(ω, fc.θ, nf, nm, ns, 1.0, des_low, des_high, α, env.bot_PEC)
end

# ---------------------------------------------------------------------------
# Forward solver
# ---------------------------------------------------------------------------

"""
    solve_forward!(pde, pt, sim, pool) -> Dict{CacheKey, CellField}

Solve Maxwell for every unique field configuration.
Caches LU factorizations in the pool.
"""
function solve_forward!(pde::MaxwellProblem, pt, sim, pool)
    fields = Dict{CacheKey,CellField}()
    sim_base = default_sim(sim)

    for fc in all_configs(pde)
        key = cache_key(fc)
        sim_fc = sim_for(sim, fc)
        pool_fc = pool_for(pool, fc)
        cache = get_cache!(pool_fc, fc)

        phys = build_phys_params(fc, pde.env, sim_fc; α=pde.α_loss)

        if !has_maxwell_factor(cache)
            pt_fc = sim_fc === sim_base ? pt : map_pt(pt, sim_fc)
            A = assemble_maxwell(pt_fc, sim_fc, phys)
            maxwell_lu!(cache, A)
        end

        source_y = (fc.polarization == :y)
        b = assemble_source(sim_fc, phys; source_y)

        E_vec = maxwell_solve!(cache, b)
        E = FEFunction(sim_fc.U, E_vec)
        fields[key] = E
    end

    return fields
end

# ---------------------------------------------------------------------------
# Adjoint solver
# ---------------------------------------------------------------------------

"""
    solve_adjoint!(pde, sources, sim, pool) -> Dict{CacheKey, CellField}

Solve adjoint Maxwell for each adjoint source.
Reuses the LU factors already in the pool.
"""
function solve_adjoint!(pde::MaxwellProblem, sources::Dict{CacheKey,Vector{ComplexF64}},
    sim, pool)
    adjoints = Dict{CacheKey,CellField}()

    for (key, source) in sources
        fc = FieldConfig(key[1]; θ=key[2], pol=key[3])
        sim_fc = sim_for(sim, fc)
        pool_fc = pool_for(pool, fc)
        cache = get_cache!(pool_fc, key)
        λ_vec = maxwell_solve_adjoint!(cache, source)
        λ = FEFunction(sim_fc.U, λ_vec)
        adjoints[key] = λ
    end

    return adjoints
end

# ---------------------------------------------------------------------------
# PDE sensitivity
# ---------------------------------------------------------------------------

"""
    pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control; space=sim.Pf)

Compute dg/dpf from the PDE term (lambda^T dA/dp E), summed over all field configs.
"""
function pde_sensitivity(pde::MaxwellProblem,
    fields::Dict{CacheKey,CellField},
    adjoints::Dict{CacheKey,CellField},
    pf, pt, sim, control;
    space=default_sim(sim).Pf)

    ∂g_∂pf = zeros(Float64, num_free_dofs(space))
    sim_base = default_sim(sim)

    for fc in all_configs(pde)
        key = cache_key(fc)
        if !haskey(fields, key) || !haskey(adjoints, key)
            continue
        end

        sim_fc = sim_for(sim, fc)
        E = fields[key]
        λ = adjoints[key]
        pf_fc = sim_fc === sim_base ? pf : map_field(pf, sim_fc, :Pf)
        pt_fc = sim_fc === sim_base ? pt : map_pt(pt, sim_fc)
        phys = build_phys_params(fc, pde.env, sim_fc; α=pde.α_loss)

        ∂g_∂pf .+= assemble_material_sensitivity_pf(E, λ, pf_fc, pt_fc, sim_fc, phys, control; space)
    end

    return ∂g_∂pf
end

# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

"""Clear all cached Maxwell factorizations."""
function invalidate_maxwell_cache!(pool::SolverCachePool)
    clear_maxwell_factors!(pool)
end
