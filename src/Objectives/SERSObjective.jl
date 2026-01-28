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
    compute_objective(obj::SERSObjective, fields, pt, sim) → Float64

Compute SERS objective using trace formula.
fields is Dict{CacheKey, CellField} but we expect exactly one pump and one emission.
"""
function compute_objective(obj::SERSObjective, fields::Dict, pt, sim)
    # Get first input (pump) and first output (emission)
    # For now, assume simple case of one pump, one emission
    keys_sorted = sort(collect(keys(fields)))
    Ep = fields[keys_sorted[1]]
    Ee = length(keys_sorted) > 1 ? fields[keys_sorted[end]] : Ep

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)

    dmg = if obj.use_damage_model
        (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep)
    else
        1.0
    end

    Ee_conj = Ee'
    integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)

    g = 0.0

    if obj.volume
        g += real(sum(∫(dmg * integrand)sim.dΩ_raman))
        g += real(sum(∫(dmg * (integrand - integrand * pt))sim.dΩ_design))
    end

    if obj.surface
        g += real(sum(∫(dmg * (pt * (integrand - integrand * pt)))sim.dΩ_design))
    end

    return g
end

"""
    compute_adjoint_sources(obj::SERSObjective, fields, pt, sim) → Dict{CacheKey, Vector}

Compute adjoint RHS vectors ∂g/∂E for each field.
"""
function compute_adjoint_sources(obj::SERSObjective, fields::Dict, pt, sim)
    keys_sorted = sort(collect(keys(fields)))
    pump_key = keys_sorted[1]
    emission_key = length(keys_sorted) > 1 ? keys_sorted[end] : pump_key
    is_elastic = pump_key == emission_key

    Ep = fields[pump_key]
    Ee = fields[emission_key]

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)

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

    coef_p = is_elastic ? 4.0 : 2.0
    Ee_conj = Ee'

    # Adjoint source for pump
    bp = zeros(ComplexF64, num_free_dofs(sim.V))

    if obj.volume
        bp .+= assemble_vector(sim.V) do v
            ∫(
                coef_p * (α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) * dmg +
                (obj.use_damage_model ?
                 (α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) * (v ⋅ ∂dmg) : 0.0)
            )sim.dΩ_raman
        end

        bp .+= assemble_vector(sim.V) do v
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
        bp .+= assemble_vector(sim.V) do v
            ∫(
                coef_p * (pt * ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) -
                                (α̂ₚ² ∘ (Ee, Ee_conj, Ep, v, αc1, αc2)) * pt)) * dmg +
                (obj.use_damage_model ?
                 pt * ((α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) -
                       (α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)) * pt) * (v ⋅ ∂dmg) : 0.0)
            )sim.dΩ_design
        end
    end

    sources = Dict{Tuple{Float64,Float64,Symbol},Vector{ComplexF64}}()
    sources[pump_key] = bp

    # Emission adjoint
    if !is_elastic
        be = zeros(ComplexF64, num_free_dofs(sim.V))

        if obj.volume
            be .+= assemble_vector(sim.V) do v
                ∫(2.0 * (α̂ₚ² ∘ (v, Ee_conj, Ep, Ep, αc1, αc2)) * dmg)sim.dΩ_raman
            end
            be .+= assemble_vector(sim.V) do v
                ∫(2.0 * ((α̂ₚ² ∘ (v, Ee_conj, Ep, Ep, αc1, αc2)) -
                         (α̂ₚ² ∘ (v, Ee_conj, Ep, Ep, αc1, αc2)) * pt) * dmg)sim.dΩ_design
            end
        end

        if obj.surface
            be .+= assemble_vector(sim.V) do v
                ∫(2.0 * (pt * ((α̂ₚ² ∘ (v, Ee_conj, Ep, Ep, αc1, αc2)) -
                               (α̂ₚ² ∘ (v, Ee_conj, Ep, Ep, αc1, αc2)) * pt)) * dmg)sim.dΩ_design
            end
        end

        sources[emission_key] = be
    end

    return sources
end

"""
    explicit_sensitivity(obj::SERSObjective, fields, pf, pt, sim, control) → Vector

Explicit ∂g/∂pf term from objective's direct dependence on pt.
"""
function explicit_sensitivity(obj::SERSObjective, fields::Dict, pf, pt, sim, control)
    keys_sorted = sort(collect(keys(fields)))
    Ep = fields[keys_sorted[1]]
    Ee = length(keys_sorted) > 1 ? fields[keys_sorted[end]] : Ep

    αc1, αc2 = α_cellfields(obj.αₚ, sim.Ω)

    dmg = if obj.use_damage_model
        (u -> damage_factor(u; γ=obj.γ_damage, E_th=obj.E_threshold)) ∘ (sumabs2 ∘ Ep)
    else
        1.0
    end

    Ee_conj = Ee'
    integrand = α̂ₚ² ∘ (Ee, Ee_conj, Ep, Ep, αc1, αc2)

    if !obj.volume && !obj.surface
        return zeros(Float64, num_free_dofs(sim.Pf))
    end

    ∇pf = ∇(pf)

    return assemble_vector(sim.Pf) do v
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
end
