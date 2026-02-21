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

            if obj.surface
                g += weight * real(sum(∫(dmg * (pt_out * (integrand - integrand * pt_out)))sim_out.dΩ_design))
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

            if obj.surface
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

                if obj.surface
                    be .+= weight * assemble_vector(sim_out.V) do v
                        ∫(2.0 * (pt_out * ((α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) -
                                       (α̂ₚ² ∘ (Ee, v, Ep_out, Ep_out, αc1_out, αc2_out)) * pt_out)) * dmg_out)sim_out.dΩ_design
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
                if obj.surface
                    term += dmg * real(integrand - 2 * integrand * pt_out)
                end
                ∫((((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf_out, ∇pf_out)) *
                  term)sim_out.dΩ_design
            end

            ∂g_∂pf .+= weight .* contrib
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
                if obj.surface
                    term += dmg * real(integrand - 2 * integrand * pt_out)
                end
                ∫(v * term)sim_out.dΩ_design
            end

            ∂g_∂pt .+= weight .* contrib
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
            if obj.surface
                term += dmg * real(integrand - 2 * integrand * pt_out)
            end

            density = weight * term
            accumulate_density_to_grid!(out, sim_base.grid, sim_out.dΩ_design, density)
        end
    end

    return out
end
