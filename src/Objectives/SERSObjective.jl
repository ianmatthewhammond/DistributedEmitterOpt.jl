"""
    SERSObjective <: ObjectiveFunction

SERS (Surface-Enhanced Raman Scattering) objective: maximize enhancement
over molecular region using SO(3)-averaged trace formulation.

## Fields
- `αₚ` — 3×3 Raman polarizability tensor
- `use_damage_model` — Enable molecular quenching
- `γ_damage` — Damage model steepness
- `E_threshold` — Damage threshold field magnitude
- `volume` — Integrate over fluid volume
- `surface` — Integrate over metal-fluid interface
"""
Base.@kwdef struct SERSObjective <: ObjectiveFunction
    # Raman tensor (3×3 complex)
    αₚ::Matrix{ComplexF64} = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)

    # Damage model
    use_damage_model::Bool = false
    γ_damage::Float64 = 1.0
    E_threshold::Float64 = Inf

    # Integration mode
    volume::Bool = true
    surface::Bool = false

    # Surface measure (only used when surface=true):
    #   :band    — legacy pt(1−pt) on Ω_design (a 1/β-thin sidewall band; see SurfaceShell.jl docs)
    #   :tv_void — 2(1−χ)|∇χ_s| over Ω_raman ∪ Ω_design, χ_s = Helmholtz(t_shell)-smoothed material
    surface_mode::Symbol = :band
    t_shell::Float64 = 10.0
    ε_grad::Float64 = 1e-6
    # :tv_void fails closed when the mesh has no "Substrate" region (the indicator would silently
    # lose its substrate=1 part). Set false only for a deliberately substrate-free geometry.
    require_substrate::Bool = true
    shell_cache::Base.RefValue{Any} = Ref{Any}(nothing)

    function SERSObjective(αₚ, use_damage_model, γ_damage, E_threshold, volume, surface,
                           surface_mode, t_shell, ε_grad, require_substrate, shell_cache)
        surface_mode in (:band, :tv_void) ||
            throw(ArgumentError("SERSObjective: surface_mode must be :band or :tv_void, got $(repr(surface_mode))"))
        (isfinite(t_shell) && t_shell > 0) ||
            throw(ArgumentError("SERSObjective: t_shell must be finite and > 0, got $t_shell"))
        # :tv_void divides by |∇χ_s|_ε to form û; ε_grad = 0 gives 0/0 wherever χ_s is flat (bulk
        # void, bulk metal), so the regularisation is mandatory there. :band never uses it.
        if surface_mode == :tv_void
            (isfinite(ε_grad) && ε_grad > 0) ||
                throw(ArgumentError("SERSObjective: surface_mode=:tv_void needs ε_grad finite and > 0 " *
                                    "(û = ∇χ_s/|∇χ_s|_ε is 0/0 in flat regions otherwise), got $ε_grad"))
        else
            (isfinite(ε_grad) && ε_grad >= 0) ||
                throw(ArgumentError("SERSObjective: ε_grad must be finite and >= 0, got $ε_grad"))
        end
        new(αₚ, use_damage_model, γ_damage, E_threshold, volume, surface,
            surface_mode, t_shell, ε_grad, require_substrate, shell_cache)
    end
end

# NOTE: The following functions are defined in Physics/SERS.jl and reused here:
#   α_invariants, α_cellfields, α̂ₚ², sumabs2, damage_factor, ∂damage_∂E

# ═══════════════════════════════════════════════════════════════════════════════
# ObjectiveFunction interface implementation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_objective(obj::SERSObjective, pde, fields, pt, sim) → Float64

Compute SERS objective using trace formula.
Sums over all input/output combinations with their weights.
"""
function compute_objective(obj::SERSObjective, pde::MaxwellProblem, fields::Dict, pt, sim)
    inputs = pde.inputs
    outputs = effective_outputs(pde)

    g = 0.0
    sim_base = default_sim(sim)

    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            # Map pump to output sim if needed
            Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
            pt_out = sim_out === sim_base ? pt : map_pt(pt, sim_out)
            αc1, αc2 = α_cellfields(obj.αₚ, sim_out.Ω)
            dmg = if obj.use_damage_model
                (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
            else
                1.0
            end

            Ee_conj = Ee'
            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep_out, Ep_out, αc1, αc2)

            if obj.volume
                g += weight * real(sum(∫(dmg * integrand)sim_out.dΩ_raman))
                g += weight * real(sum(∫(dmg * (integrand - integrand * pt_out))sim_out.dΩ_design))
            end

            if obj.surface && obj.surface_mode == :band
                g += weight * real(sum(∫(dmg * (pt_out * (integrand - integrand * pt_out)))sim_out.dΩ_design))
            elseif obj.surface && obj.surface_mode == :tv_void
                g += weight * shell_objective_pair(obj, dmg, integrand, pt_out, sim, sim_out)
            end
        end
    end

    return g
end

"""
    compute_adjoint_sources(obj::SERSObjective, pde, fields, pt, sim) → Dict{CacheKey, Vector}

Compute adjoint RHS vectors ∂g/∂E for each field.
Sums contributions over all input/output combinations with weights.
"""
function compute_adjoint_sources(obj::SERSObjective, pde::MaxwellProblem, fields::Dict, pt, sim)
    inputs = pde.inputs
    outputs = effective_outputs(pde)

    sources = Dict{CacheKey,Vector{ComplexF64}}()
    sim_base = default_sim(sim)

    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]
        nV_in = num_free_dofs(sim_in.V)

        dmg = if obj.use_damage_model
            (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep)
        else
            1.0
        end

        ∂dmg = if obj.use_damage_model
            (u -> ∂damage_∂E(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ Ep
        else
            0.0 * Ep
        end

        bp = get!(sources, key_in) do
            zeros(ComplexF64, nV_in)
        end

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            is_elastic = key_out == key_in
            coef_p = is_elastic ? 4.0 : 2.0
            Ee_conj = Ee'

            # Map to input sim for pump adjoint
            Ee_in = sim_out === sim_in ? Ee : map_field(Ee, sim_in, :U)
            Ee_conj_in = Ee_in'
            pt_in = sim_in === sim_base ? pt : map_pt(pt, sim_in)
            αc1_in, αc2_in = α_cellfields(obj.αₚ, sim_in.Ω)

            if obj.volume
                bp .+= weight * assemble_vector(sim_in.V) do v
                    ∫(
                        coef_p * (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) * dmg +
                        (obj.use_damage_model ?
                         (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) * (v ⋅ ∂dmg) : 0.0)
                    )sim_in.dΩ_raman
                end

                bp .+= weight * assemble_vector(sim_in.V) do v
                    ∫(
                        coef_p * ((α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) -
                                  (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) * pt_in) * dmg +
                        (obj.use_damage_model ?
                         ((α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) -
                          (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) * pt_in) * (v ⋅ ∂dmg) : 0.0)
                    )sim_in.dΩ_design
                end
            end

            if obj.surface && obj.surface_mode == :band
                bp .+= weight * assemble_vector(sim_in.V) do v
                    ∫(
                        coef_p * (pt_in * ((α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) -
                                        (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) * pt_in)) * dmg +
                        (obj.use_damage_model ?
                         pt_in * ((α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) -
                               (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) * pt_in) * (v ⋅ ∂dmg) : 0.0)
                    )sim_in.dΩ_design
                end
            end
            if obj.surface && obj.surface_mode == :tv_void
                _shell_check_single_sim(sim, sim_in)
                _, _, S_r, S_d = shell_weights(obj, pt_in, sim_in)
                bp .+= weight * assemble_vector(sim_in.V) do v
                    ∫(
                        coef_p * (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) * S_r * dmg +
                        (obj.use_damage_model ?
                         (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) * S_r * (v ⋅ ∂dmg) : 0.0)
                    )sim_in.dΩ_raman +
                    ∫(
                        coef_p * (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, v, αc1_in, αc2_in)) * S_d * dmg +
                        (obj.use_damage_model ?
                         (α̂ₚ² ∘ (Ee_in, Ee_conj_in, Ep, Ep, αc1_in, αc2_in)) * S_d * (v ⋅ ∂dmg) : 0.0)
                    )sim_in.dΩ_design
                end
            end

            if !is_elastic
                nV_out = num_free_dofs(sim_out.V)
                be = get!(sources, key_out) do
                    zeros(ComplexF64, nV_out)
                end

                # Map pump to output sim for emission adjoint
                Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
                pt_out = sim_out === sim_base ? pt : map_pt(pt, sim_out)
                αc1_out, αc2_out = α_cellfields(obj.αₚ, sim_out.Ω)
                dmg_out = if obj.use_damage_model
                    (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
                else
                    1.0
                end

                if obj.volume
                    be .+= weight * assemble_vector(sim_out.V) do v
                        ∫(2.0 * (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * dmg_out)sim_out.dΩ_raman
                    end
                    be .+= weight * assemble_vector(sim_out.V) do v
                        ∫(2.0 * ((α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) -
                                 (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * pt_out) * dmg_out)sim_out.dΩ_design
                    end
                end

                if obj.surface && obj.surface_mode == :band
                    be .+= weight * assemble_vector(sim_out.V) do v
                        ∫(2.0 * (pt_out * ((α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) -
                                       (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * pt_out)) * dmg_out)sim_out.dΩ_design
                    end
                end
                if obj.surface && obj.surface_mode == :tv_void
                    _shell_check_single_sim(sim, sim_out)
                    _, _, S_r, S_d = shell_weights(obj, pt_out, sim_out)
                    be .+= weight * assemble_vector(sim_out.V) do v
                        ∫(2.0 * (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * S_r * dmg_out)sim_out.dΩ_raman +
                        ∫(2.0 * (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * S_d * dmg_out)sim_out.dΩ_design
                    end
                end
            end
        end
    end

    return sources
end

"""
    explicit_sensitivity(obj::SERSObjective, pde, fields, pf, pt, sim, control; space=sim.Pf) → Vector

Explicit ∂g/∂pf term from objective's direct dependence on pt.
Sums contributions over all input/output combinations with weights.
"""
function explicit_sensitivity(obj::SERSObjective, pde::MaxwellProblem, fields::Dict, pf, pt, sim, control; space=default_sim(sim).Pf)
    if !obj.volume && !obj.surface
        return zeros(Float64, num_free_dofs(space))
    end

    inputs = pde.inputs
    outputs = effective_outputs(pde)

    ∂g_∂pf = zeros(Float64, num_free_dofs(space))
    sim_base = default_sim(sim)
    pair_terms = obj.volume || (obj.surface && obj.surface_mode == :band)

    for fc_in in (pair_terms ? inputs : FieldConfig[])
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            Ee_conj = Ee'

            Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
            dmg = if obj.use_damage_model
                (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
            else
                1.0
            end
            pf_out = sim_out === sim_base ? pf : map_field(pf, sim_out, :Pf)
            pt_out = sim_out === sim_base ? pt : map_pt(pt, sim_out)
            ∇pf_out = ∇(pf_out)
            αc1_out, αc2_out = α_cellfields(obj.αₚ, sim_out.Ω)

            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep_out, Ep_out, αc1_out, αc2_out)

            contrib = assemble_vector(sim_out.Pf) do v
                term = 0.0
                if obj.volume
                    term += dmg * (-real(integrand))
                end
                if obj.surface && obj.surface_mode == :band
                    term += dmg * real(integrand - 2 * integrand * pt_out)
                end
                ∫(dpt_dpf_field(v, pf_out, control, sim_out.Ω_design) *
                  term)sim_out.dΩ_design
            end

            ∂g_∂pf .+= weight .* contrib
        end
    end

    if obj.surface && obj.surface_mode == :tv_void
        dens = shell_explicit_density(obj, pde, fields, pt, sim)
        if dens !== nothing
            ∇pf0 = ∇(pf)
            ∂g_∂pf .+= assemble_vector(sim_base.Pf) do v
                ∫(dpt_dpf_field(v, pf, control, sim_base.Ω_design) *
                  dens)sim_base.dΩ_design
            end
        end
    end

    return ∂g_∂pf
end

"""
    explicit_sensitivity_pt(obj, pde, fields, pt, sim; space=sim.Pf) -> Vector

Explicit objective sensitivity with respect to projected design `pt`.
Used by legacy foundry smoothing mode where SSP is applied on the 2D grid.
"""
function explicit_sensitivity_pt(obj::SERSObjective, pde::MaxwellProblem, fields::Dict, pt, sim; space=default_sim(sim).Pf)
    if !obj.volume && !obj.surface
        return zeros(Float64, num_free_dofs(space))
    end

    inputs = pde.inputs
    outputs = effective_outputs(pde)

    ∂g_∂pt = zeros(Float64, num_free_dofs(space))
    sim_base = default_sim(sim)
    pair_terms = obj.volume || (obj.surface && obj.surface_mode == :band)

    for fc_in in (pair_terms ? inputs : FieldConfig[])
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            Ee_conj = Ee'

            Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
            dmg = if obj.use_damage_model
                (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
            else
                1.0
            end
            pt_out = sim_out === sim_base ? pt : map_pt(pt, sim_out)
            αc1_out, αc2_out = α_cellfields(obj.αₚ, sim_out.Ω)

            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep_out, Ep_out, αc1_out, αc2_out)

            contrib = assemble_vector(sim_out.Pf) do v
                term = 0.0
                if obj.volume
                    term += dmg * (-real(integrand))
                end
                if obj.surface && obj.surface_mode == :band
                    term += dmg * real(integrand - 2 * integrand * pt_out)
                end
                ∫(v * term)sim_out.dΩ_design
            end

            ∂g_∂pt .+= weight .* contrib
        end
    end

    if obj.surface && obj.surface_mode == :tv_void
        dens = shell_explicit_density(obj, pde, fields, pt, sim)
        if dens !== nothing
            ∂g_∂pt .+= assemble_vector(sim_base.Pf) do v
                ∫(v * dens)sim_base.dΩ_design
            end
        end
    end

    return ∂g_∂pt
end

"""
    explicit_sensitivity_pt_grid!(out, obj, pde, fields, pt, sim)

Legacy foundry accumulation of explicit dg/dpt directly onto 2D grid DOFs.
"""
function explicit_sensitivity_pt_grid!(out::Vector{Float64},
    obj::SERSObjective,
    pde::MaxwellProblem,
    fields::Dict,
    pt, sim)

    fill!(out, 0.0)
    if !obj.volume && !obj.surface
        return out
    end

    inputs = pde.inputs
    outputs = effective_outputs(pde)
    sim_base = default_sim(sim)
    pair_terms = obj.volume || (obj.surface && obj.surface_mode == :band)

    for fc_in in (pair_terms ? inputs : FieldConfig[])
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        sim_in = sim_for(sim, fc_in)
        Ep = fields[key_in]

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            sim_out = sim_for(sim, fc_out)
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            Ee_conj = Ee'
            Ep_out = sim_out === sim_in ? Ep : map_field(Ep, sim_out, :U)
            dmg = if obj.use_damage_model
                (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep_out)
            else
                1.0
            end
            pt_out = sim_out === sim_base ? pt : map_pt(pt, sim_out)
            αc1_out, αc2_out = α_cellfields(obj.αₚ, sim_out.Ω)
            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep_out, Ep_out, αc1_out, αc2_out)

            term = 0.0
            if obj.volume
                term += dmg * (-real(integrand))
            end
            if obj.surface && obj.surface_mode == :band
                term += dmg * real(integrand - 2 * integrand * pt_out)
            end

            density = weight * term
            accumulate_density_to_grid!(out, sim_base.grid, sim_out.dΩ_design, density)
        end
    end

    if obj.surface && obj.surface_mode == :tv_void
        dens = shell_explicit_density(obj, pde, fields, pt, sim)
        dens === nothing || accumulate_density_to_grid!(out, sim_base.grid, sim_base.dΩ_design, dens)
    end

    return out
end
