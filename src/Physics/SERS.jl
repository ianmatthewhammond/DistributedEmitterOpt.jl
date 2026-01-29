"""
    SERS

SO(3)-averaged trace formula for SERS enhancement, plus an optional
molecular damage/quenching model. Used by SERSObjective.
"""

# ---------------------------------------------------------------------------
# Polarizability tensor invariants
# ---------------------------------------------------------------------------

"""
    α_invariants(αp) -> (α_par², α_perp²)

SO(3) invariants for orientational averaging of the Raman tensor.
"""
function α_invariants(αₚ::Matrix{ComplexF64})
    β = 1.0
    m = 3

    α_par² = (abs(tr(αₚ))^2 + tr(αₚ' * αₚ + conj.(αₚ) * αₚ)) * β / (m * (m * β + 2))
    α_perp² = (tr(αₚ' * αₚ) / 3 - α_par²) / 2

    return α_par², α_perp²
end

"""Build CellField constants for the alpha invariants over Ω."""
function α_cellfields(αₚ::Matrix{ComplexF64}, Ω)
    α_par², α_perp² = α_invariants(αₚ)
    CellField(α_par², Ω), CellField(α_perp², Ω)
end

# ---------------------------------------------------------------------------
# Trace formula integrand
# ---------------------------------------------------------------------------

"""
    α̂ₚ²(Ee, E′e, Ep, E′p, αc1, αc2) -> scalar

SO(3)-averaged SERS intensity integrand.
For elastic scattering with identity αp, reduces to |E|^4.
"""
function α̂ₚ²(Ee, E′e, Ep, E′p, αc1, αc2)
    return (
        αc1 * (E′e ⋅ (outer(Ep, E′p') ⋅ Ee)) +
        αc2 * (E′p' ⋅ Ep) * (E′e ⋅ Ee) -
        αc2 * (E′e ⋅ ((2 * outer(Ep, E′p') - conj(outer(E′p, Ep'))) ⋅ Ee))
    )
end

Gridap.zero(::typeof(α̂ₚ²)) = 0.0

# ---------------------------------------------------------------------------
# |E|² helper
# ---------------------------------------------------------------------------

"""Squared magnitude of a VectorValue."""
sumabs2(E::VectorValue) = sum(abs.(Tuple(E)) .^ 2)
sumabs2(E::CellField) = Operation(sumabs2)(E)

# ---------------------------------------------------------------------------
# Damage / quenching model
# ---------------------------------------------------------------------------

"""
    damage_factor(E²; γ, E_th) -> Real

Molecular quenching factor: attenuates SERS signal at high field intensity.
Returns a value in (0, 1] (1 = no damage).
"""
function damage_factor(E²::Real; γ::Float64, E_th::Float64)
    if E_th == Inf || !isfinite(γ)
        return 1.0
    end
    val = 1.0 / (1.0 + exp(γ * (E² - E_th^2)))
    return (isnan(val) || isinf(val)) ? 1.0 : val
end

"""Derivative of damage_factor with respect to E."""
function ∂damage_∂E(E::VectorValue; γ::Float64, E_th::Float64)
    if E_th == Inf || !isfinite(γ)
        return E * 0.0
    end
    E² = sumabs2(E)
    dmg = damage_factor(E²; γ, E_th)
    val = exp(γ * (E² - E_th^2)) * dmg * dmg
    total = -γ * 2 * E * val
    return (isnan(val) || isinf(val)) ? E * 0.0 : total
end
