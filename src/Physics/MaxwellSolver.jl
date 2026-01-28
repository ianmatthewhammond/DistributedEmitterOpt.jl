"""
    MaxwellSolver

High-level solve functions for MaxwellProblem that work with the new
architecture (FieldConfig, SolverCachePool, Environment).

These functions bridge the gap between the abstract problem definition
and the existing assembly/solve routines in Maxwell.jl.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PhysicalParams construction from new types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_phys_params(fc::FieldConfig, env::Environment, sim; α::Float64=0.0)

Construct PhysicalParams from a FieldConfig and Environment.
"""
function build_phys_params(fc::FieldConfig, env::Environment, sim; α::Float64=0.0)
    λ = fc.λ
    ω = 2π / λ

    # Resolve material indices at this wavelength
    nf = resolve_index(env.mat_fluid, λ)
    nm = resolve_index(env.mat_design, λ)
    ns = resolve_index(env.mat_substrate, λ)

    # Get design region bounds from simulation
    des_low, des_high = getdesignz(sim.labels, sim.Ω)

    PhysicalParams(ω, fc.θ, nf, nm, ns, 1.0, des_low, des_high, α)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Forward solver
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_forward!(pde::MaxwellProblem, pt, sim, pool) → Dict{CacheKey, CellField}

Solve Maxwell equation for all unique field configurations.
Returns a Dict mapping cache keys to electric field CellFields.

Caches LU factorizations in the pool for reuse.
"""
function solve_forward!(pde::MaxwellProblem, pt, sim, pool::SolverCachePool)
    fields = Dict{CacheKey,CellField}()

    for fc in all_configs(pde)
        key = cache_key(fc)
        cache = get_cache!(pool, fc)

        # Build physical params for this config
        phys = build_phys_params(fc, pde.env, sim; α=pde.α_loss)

        # Assemble and factorize if needed
        if !has_maxwell_factor(cache)
            A = assemble_maxwell(pt, sim, phys)
            maxwell_lu!(cache, A)
        end

        # Assemble source (polarization determines y vs x)
        source_y = (fc.polarization == :y)
        b = assemble_source(sim, phys; source_y)

        # Solve
        E_vec = maxwell_solve!(cache, b)

        # Create CellField from solution vector
        E = FEFunction(sim.U, E_vec)
        fields[key] = E
    end

    return fields
end

# ═══════════════════════════════════════════════════════════════════════════════
# Adjoint solver
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_adjoint!(pde::MaxwellProblem, sources::Dict, sim, pool) → Dict{CacheKey, CellField}

Solve adjoint Maxwell equations given adjoint sources (∂g/∂E).
Returns a Dict mapping cache keys to adjoint field CellFields.

Assumes Maxwell matrices are already factorized in the pool.
"""
function solve_adjoint!(pde::MaxwellProblem, sources::Dict{CacheKey,Vector{ComplexF64}},
    sim, pool::SolverCachePool)
    adjoints = Dict{CacheKey,CellField}()

    for (key, source) in sources
        cache = get_cache!(pool, key)

        # Solve adjoint: A' λ = source
        λ_vec = maxwell_solve_adjoint!(cache, source)

        # Create CellField
        λ = FEFunction(sim.U, λ_vec)
        adjoints[key] = λ
    end

    return adjoints
end

# ═══════════════════════════════════════════════════════════════════════════════
# PDE sensitivity (material sensitivity from adjoint)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pde_sensitivity(pde::MaxwellProblem, fields, adjoints, pf, pt, sim, control; space=sim.Pf)

Compute ∂g/∂pf from PDE (λᵀ ∂A/∂p E) for all field/adjoint pairs.
Sums contributions from all configurations weighted appropriately.
"""
function pde_sensitivity(pde::MaxwellProblem,
    fields::Dict{CacheKey,CellField},
    adjoints::Dict{CacheKey,CellField},
    pf, pt, sim, control;
    space=sim.Pf)

    ∂g_∂pf = zeros(Float64, num_free_dofs(space))

    # Get all input configs
    for fc_in in pde.inputs
        key_in = cache_key(fc_in)
        if !haskey(fields, key_in) || !haskey(adjoints, key_in)
            continue
        end

        E = fields[key_in]
        λ = adjoints[key_in]
        phys = build_phys_params(fc_in, pde.env, sim; α=pde.α_loss)

        # Material sensitivity for this input
        ∂g_∂pf .+= fc_in.weight * assemble_material_sensitivity_pf(E, λ, pf, pt, sim, phys, control; space)
    end

    # Get output configs (if different from inputs)
    for fc_out in effective_outputs(pde)
        key_out = cache_key(fc_out)
        key_in_match = any(cache_key(fc) == key_out for fc in pde.inputs)

        if key_in_match
            continue  # Already handled above
        end

        if !haskey(fields, key_out) || !haskey(adjoints, key_out)
            continue
        end

        E = fields[key_out]
        λ = adjoints[key_out]
        phys = build_phys_params(fc_out, pde.env, sim; α=pde.α_loss)

        ∂g_∂pf .+= fc_out.weight * assemble_material_sensitivity_pf(E, λ, pf, pt, sim, phys, control; space)
    end

    return ∂g_∂pf
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: clear cached factors when design changes
# ═══════════════════════════════════════════════════════════════════════════════

"""Clear Maxwell factorizations when pt changes significantly."""
function invalidate_maxwell_cache!(pool::SolverCachePool)
    clear_maxwell_factors!(pool)
end
