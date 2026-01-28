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

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)

    g = 0.0

    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        Ep = fields[key_in]

        dmg = if obj.use_damage_model
            (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep)
        else
            1.0
        end

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            Ee_conj = Ee'
            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)

            if obj.volume
                g += weight * real(sum(∫(dmg * integrand)sim.dΩ_raman))
                g += weight * real(sum(∫(dmg * (integrand - integrand * pt))sim.dΩ_design))
            end

            if obj.surface
                g += weight * real(sum(∫(dmg * (pt * (integrand - integrand * pt)))sim.dΩ_design))
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

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)
    nV = num_free_dofs(sim.V)

    sources = Dict{CacheKey,Vector{ComplexF64}}()

    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        Ep = fields[key_in]

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
            zeros(ComplexF64, nV)
        end

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            is_elastic = key_out == key_in
            coef_p = is_elastic ? 4.0 : 2.0
            Ee_conj = Ee'

            if obj.volume
                bp .+= weight * assemble_vector(sim.V) do v
                    ∫(
                        coef_p * (α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) * dmg +
                        (obj.use_damage_model ?
                         (α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) * (v ⋅ ∂dmg) : 0.0)
                    )sim.dΩ_raman
                end

                bp .+= weight * assemble_vector(sim.V) do v
                    ∫(
                        coef_p * ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) -
                                  (α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) * pt) * dmg +
                        (obj.use_damage_model ?
                         ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) -
                          (α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) * pt) * (v ⋅ ∂dmg) : 0.0)
                    )sim.dΩ_design
                end
            end

            if obj.surface
                bp .+= weight * assemble_vector(sim.V) do v
                    ∫(
                        coef_p * (pt * ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) -
                                        (α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) * pt)) * dmg +
                        (obj.use_damage_model ?
                         pt * ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) -
                               (α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) * pt) * (v ⋅ ∂dmg) : 0.0)
                    )sim.dΩ_design
                end
            end

            if !is_elastic
                be = get!(sources, key_out) do
                    zeros(ComplexF64, nV)
                end

                if obj.volume
                    be .+= weight * assemble_vector(sim.V) do v
                        ∫(2.0 * (α̂ₚ² ∘ (Ee, v, Ep, Ep, αc1, αc2)) * dmg)sim.dΩ_raman
                    end
                    be .+= weight * assemble_vector(sim.V) do v
                        ∫(2.0 * ((α̂ₚ² ∘ (Ee, v, Ep, Ep, αc1, αc2)) -
                                 (α̂ₚ² ∘ (Ee, v, Ep, Ep, αc1, αc2)) * pt) * dmg)sim.dΩ_design
                    end
                end

                if obj.surface
                    be .+= weight * assemble_vector(sim.V) do v
                        ∫(2.0 * (pt * ((α̂ₚ² ∘ (Ee, v, Ep, Ep, αc1, αc2)) -
                                       (α̂ₚ² ∘ (Ee, v, Ep, Ep, αc1, αc2)) * pt)) * dmg)sim.dΩ_design
                    end
                end
            end
        end
    end

    return sources
end

"""
    explicit_sensitivity(obj::SERSObjective, pde, fields, pf, pt, sim, control) → Vector

Explicit ∂g/∂pf term from objective's direct dependence on pt.
Sums contributions over all input/output combinations with weights.
"""
function explicit_sensitivity(obj::SERSObjective, pde::MaxwellProblem, fields::Dict, pf, pt, sim, control)
    if !obj.volume && !obj.surface
        return zeros(Float64, num_free_dofs(sim.Pf))
    end

    inputs = pde.inputs
    outputs = effective_outputs(pde)

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)
    ∇pf = ∇(pf)

    ∂g_∂pf = zeros(Float64, num_free_dofs(sim.Pf))

    for fc_in in inputs
        key_in = cache_key(fc_in)
        haskey(fields, key_in) || continue
        Ep = fields[key_in]

        dmg = if obj.use_damage_model
            (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep)
        else
            1.0
        end

        for fc_out in outputs
            key_out = cache_key(fc_out)
            haskey(fields, key_out) || continue
            Ee = fields[key_out]

            weight = fc_in.weight * fc_out.weight
            Ee_conj = Ee'
            integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)

            contrib = assemble_vector(sim.Pf) do v
                term = 0.0
                if obj.volume
                    term += dmg * (-real(integrand))
                end
                if obj.surface
                    term += dmg * real(integrand - 2 * integrand * pt)
                end
                ∫((((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf, ∇pf)) *
                  term)sim.dΩ_design
            end

            ∂g_∂pf .+= weight .* contrib
        end
    end

    return ∂g_∂pf
end
