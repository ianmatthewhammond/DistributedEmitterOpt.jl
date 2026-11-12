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

    validate_field_boundaries(sim, fc)
    if fc.formulation == :bloch && (abs(imag(nf)) > 1e-12 || !isapprox(real(nf), fc.n_incident; atol=1e-12))
        throw(ArgumentError("FieldConfig: n_incident must equal the lossless fluid index"))
    end

    if fc.formulation == :bloch
        check_diffraction_orders(lattice_periods(sim), fc, ns;
            buffers=(sim.bounds[6] - des_high, des_low - sim.bounds[5]), bottom_pec=env.bot_PEC)
    end

    pml = nothing
    if !isnothing(env.pml)
        z_bot, d_bot, z_top, d_top = getpmlz(sim.labels, sim.Ω)
        pml = ZPMLParams(env.pml, ω, ns, nf, z_bot, d_bot, z_top, d_top)
    end

    PhysicalParams(ω, fc.θ, nf, nm, ns, 1.0, des_low, des_high, α, env.bot_PEC, pml,
        fc.tilt_x, fc.n_incident, fc.polarization, fc.formulation)
end

# ---------------------------------------------------------------------------
# Diffraction orders (Bloch formulation)
# ---------------------------------------------------------------------------

"An order with |q| ≤ CUTOFF_RTOL k |n| is at cutoff (it decays over more than about 160 wavelengths) and counts as propagating."
const CUTOFF_RTOL = 1e-3

"""
    check_diffraction_orders(periods, fc, ns; buffers=nothing, bottom_pec=false)

The Bloch boundary terms are the specular-order admittance, exact for the specular wave
only. Every nonspecular order (m, n) ≠ (0, 0) of the lattice, with in-plane wavevector
κ + 2π (m/Px, n/Py), must be evanescent in the incidence medium (index `fc.n_incident`) and
in the substrate (index `ns`); otherwise it reaches a boundary as a wave the admittance does
not absorb, and this throws, naming the orders and the lattice periods (same aspect ratio)
below which every order is evanescent. An order counts as propagating when its normal
wavenumber oscillates faster than it decays (|Re q| > |Im q|), or when it is at or near
cutoff: |q| ≤ `CUTOFF_RTOL` k |n|, a decay length above about 160 wavelengths in the medium.
Lossless and low-loss substrates carry orders, a metal absorbs them. A PEC bottom
(`bottom_pec=true`) has no admittance term, so its substrate orders are not checked.

`buffers = (d_top, d_bottom)` are the distances from the design region to the top and bottom
faces; a warning is issued when the slowest-decaying evanescent order still has amplitude
above 1e-2 at either checked face.
"""
function check_diffraction_orders(periods, fc::FieldConfig, ns::Number; buffers=nothing, bottom_pec::Bool=false)
    k = 2π / fc.λ
    κ = inplane_wavevector(fc)
    κx, κy = real(κ[1]), real(κ[2])
    Gx, Gy = 2π / periods[1], 2π / periods[2]
    media = (("incidence medium", complex(fc.n_incident)), ("substrate", complex(ns)))
    bottom_pec && (media = media[1:1])
    kmax = k * maximum(abs(last(m)) for m in media) + hypot(κx, κy)
    M, N = ceil(Int, kmax / Gx) + 1, ceil(Int, kmax / Gy) + 1
    propagating = Tuple{Int,Int,String}[]
    slowest = [Inf, Inf]
    for m in -M:M, n in -N:N
        (m, n) == (0, 0) && continue
        kt2 = (κx + m * Gx)^2 + (κy + n * Gy)^2
        for (i, (name, nmed)) in enumerate(media)
            q = outgoing_normal_wavenumber(k, nmed^2, kt2)
            if abs(real(q)) > abs(imag(q)) || abs(q) <= CUTOFF_RTOL * k * abs(nmed)
                push!(propagating, (m, n, name))
            else
                slowest[i] = min(slowest[i], abs(imag(q)))
            end
        end
    end
    if !isempty(propagating)
        sort!(propagating; by=o -> (abs(o[1]) + abs(o[2]), o[3]))
        indices = [real(fc.n_incident)]
        !bottom_pec && abs(imag(ns)) < abs(real(ns)) && push!(indices, real(ns))
        s = largest_evanescent_scale(periods, (κx, κy), k, indices)
        orders = join(["($(o[1]), $(o[2])) in the $(o[3])" for o in propagating[1:min(end, 6)]], ", ")
        r1(x) = round(x; digits=1)
        throw(ArgumentError("$(repr(fc)): nonspecular diffraction order(s) $orders propagate or sit at cutoff at lattice periods $(r1(periods[1])) × $(r1(periods[2])) nm; the specular-order boundary does not absorb them. Every order is evanescent below lattice periods $(r1(s * periods[1])) × $(r1(s * periods[2])) nm (same aspect ratio). In build_angular_simulation, geo.L and geo.W are half the lattice period."))
    end
    if buffers !== nothing
        for (i, (name, _)) in enumerate(media)
            amp = exp(-slowest[i] * buffers[i])
            amp > 1e-2 && @warn "$(repr(fc)): the slowest evanescent diffraction order still has amplitude $(round(amp; sigdigits=2)) at the $(i == 1 ? "top" : "bottom") face ($(round(buffers[i]; digits=1)) nm from the design region); the specular-order boundary is not exact for it" maxlog = 1
        end
    end
    return nothing
end

"Largest scale factor s such that, at periods s·P, every nonzero order is evanescent in every lossless medium of the given indices."
function largest_evanescent_scale(periods, κ, k, indices)
    Gx, Gy = 2π / periods[1], 2π / periods[2]
    s = Inf
    for nmed in indices, m in -4:4, n in -4:4
        (m, n) == (0, 0) && continue
        gx, gy = m * Gx, n * Gy
        g2 = gx^2 + gy^2
        kg = κ[1] * gx + κ[2] * gy
        disc = kg^2 + g2 * ((k * nmed)^2 - (κ[1]^2 + κ[2]^2))
        disc > 0 || continue
        t = (-kg + sqrt(disc)) / g2     # the order propagates for 1/s < t
        t > 0 && (s = min(s, 1 / t))
    end
    return s
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

        # Always update the Maxwell factorization for current pt.
        # `maxwell_lu!` performs in-place refactorization when a cached factor exists.
        pt_fc = sim_fc === sim_base ? pt : map_pt(pt, sim_fc)
        A = assemble_maxwell(pt_fc, sim_fc, phys)
        maxwell_lu!(cache, A)

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
function solve_adjoint!(pde::MaxwellProblem, sources::AbstractDict{<:CacheKey,Vector{ComplexF64}},
    sim, pool)
    adjoints = Dict{CacheKey,CellField}()

    for (key, source) in sources
        fc = field_config(key)
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
    fields::AbstractDict{<:CacheKey,<:CellField},
    adjoints::AbstractDict{<:CacheKey,<:CellField},
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

"""
    pde_sensitivity_pt(pde, fields, adjoints, pt, sim; space=sim.Pf)

Compute dg/dpt from the PDE term (lambda^T dA/dpt E), summed over all configs.
Used by legacy foundry smoothing mode where SSP lives on the 2D grid.
"""
function pde_sensitivity_pt(pde::MaxwellProblem,
    fields::AbstractDict{<:CacheKey,<:CellField},
    adjoints::AbstractDict{<:CacheKey,<:CellField},
    pt, sim;
    space=default_sim(sim).Pf)

    ∂g_∂pt = zeros(Float64, num_free_dofs(space))
    sim_base = default_sim(sim)

    for fc in all_configs(pde)
        key = cache_key(fc)
        if !haskey(fields, key) || !haskey(adjoints, key)
            continue
        end

        sim_fc = sim_for(sim, fc)
        E = fields[key]
        λ = adjoints[key]
        pt_fc = sim_fc === sim_base ? pt : map_pt(pt, sim_fc)
        phys = build_phys_params(fc, pde.env, sim_fc; α=pde.α_loss)

        ∂g_∂pt .+= assemble_material_sensitivity_pt(E, λ, pt_fc, sim_fc, phys; space)
    end

    return ∂g_∂pt
end

"""
    pde_sensitivity_pt_grid!(out, pde, fields, adjoints, pt, sim)

Legacy foundry accumulation of dg/dpt directly onto 2D grid DOFs via quadrature.
"""
function pde_sensitivity_pt_grid!(out::Vector{Float64},
    pde::MaxwellProblem,
    fields::AbstractDict{<:CacheKey,<:CellField},
    adjoints::AbstractDict{<:CacheKey,<:CellField},
    pt, sim)

    fill!(out, 0.0)
    sim_base = default_sim(sim)

    for fc in all_configs(pde)
        key = cache_key(fc)
        if !haskey(fields, key) || !haskey(adjoints, key)
            continue
        end

        sim_fc = sim_for(sim, fc)
        E = fields[key]
        λ = adjoints[key]
        pt_fc = sim_fc === sim_base ? pt : map_pt(pt, sim_fc)
        phys = build_phys_params(fc, pde.env, sim_fc; α=pde.α_loss)
        ∂εε = p -> ∂ε_∂p(p, phys)
        density = real((phys.ω^2) * ((∂εε ∘ pt_fc) * (conj(λ) ⋅ E)))
        accumulate_density_to_grid!(out, sim_base.grid, sim_fc.dΩ_design, density)
    end

    return out
end

# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

"""Clear all cached Maxwell factorizations."""
function invalidate_maxwell_cache!(pool::SolverCachePool)
    clear_maxwell_factors!(pool)
end
