"""
    MaterialInterp

Material interpolation schemes mapping design variable p ∈ [0,1] to
permittivity ε. Default is Christiansen interpolation.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Christiansen interpolation (default)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    christiansen_ε(p, nf, nm) -> Complex

Christiansen refractive index interpolation:
  n(p) = nf + (nm - nf) * p
  ε(p) = n(p)²

This gives:
  p=0 → ε = nf² (fluid)
  p=1 → ε = nm² (metal)
"""
function christiansen_ε(p::Real, nf::ComplexF64, nm::ComplexF64)
    n = nf + (nm - nf) * p
    return n^2
end

"""Shifted version: returns ε(p) - ε_base for additive formulation."""
function christiansen_ε_shift(p::Real, nf::ComplexF64, nm::ComplexF64, ε_base::ComplexF64)
    return christiansen_ε(p, nf, nm) - ε_base
end

"""Derivative: ∂ε/∂p = 2n(p)(nm - nf)"""
function ∂christiansen_ε(p::Real, nf::ComplexF64, nm::ComplexF64)
    n = nf + (nm - nf) * p
    return 2 * n * (nm - nf)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SIMP interpolation (alternative)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    simp_ε(p, ε_min, ε_max; q=3) -> Complex

SIMP (Solid Isotropic Material with Penalization):
  ε(p) = ε_min + p^q * (ε_max - ε_min)
"""
function simp_ε(p::Real, ε_min::ComplexF64, ε_max::ComplexF64; q::Float64=3.0)
    return ε_min + p^q * (ε_max - ε_min)
end

"""Derivative: ∂ε/∂p = q * p^(q-1) * (ε_max - ε_min)"""
function ∂simp_ε(p::Real, ε_min::ComplexF64, ε_max::ComplexF64; q::Float64=3.0)
    return q * p^(q - 1) * (ε_max - ε_min)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Lorentz-Lorenz (for metamaterials)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lorentz_lorenz_ε(p, ε1, ε2) -> Complex

Lorentz-Lorenz effective medium:
  (ε_eff - 1)/(ε_eff + 2) = p(ε1-1)/(ε1+2) + (1-p)(ε2-1)/(ε2+2)
"""
function lorentz_lorenz_ε(p::Real, ε1::ComplexF64, ε2::ComplexF64)
    f1 = (ε1 - 1) / (ε1 + 2)
    f2 = (ε2 - 1) / (ε2 + 2)
    f_eff = p * f1 + (1 - p) * f2
    ε_eff = (1 + 2 * f_eff) / (1 - f_eff)
    return ε_eff
end
