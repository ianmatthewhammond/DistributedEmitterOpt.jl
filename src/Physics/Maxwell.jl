"""
    Maxwell

Weak form assembly for the 3D Maxwell curl-curl equation with
Sommerfeld ABC (first-order absorbing boundary conditions).

## Main functions
- `assemble_maxwell(pt, sim, phys)` — System matrix
- `assemble_source(sim, phys)` — Source vector
- `ε_background(x, phys)` — Background permittivity
- `ε_design(p, phys)` — Design variable permittivity
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Permittivity functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ε_background(x, phys) -> ComplexF64

Background permittivity: ns² in substrate (z < des_low), nf² in fluid (z > des_low).
"""
function ε_background(x, phys)
    if x[3] < phys.des_low
        return phys.ns^2 * (1 - im * phys.α)
    else
        return phys.nf^2 * (1 - im * phys.α)
    end
end

"""Background for WF (wave-function) formulation."""
function ε_background_wf(x, phys)
    if x[3] < phys.des_low
        return phys.ns^2 * (1 - im * phys.α)
    end
    return phys.nf^2 * (1 - im * phys.α)
end

"""
    ε_design(p, phys) -> ComplexF64

Nonlinear Material interpolation: (nf + (nm - nf)p)² - base²
See [A non-linear material interpolation for metallic-nanoparticles](https://www.sciencedirect.com/science/article/pii/S0045782518304328) for more information.
This gives ε = ε_base when p=0 and ε = εm when p=1.
"""
function ε_design_wf(p, phys)
    neff = phys.nf + (phys.nm - phys.nf) * p
    (neff^2 - phys.nf^2) * (1 - im * phys.α)
end

"""
Permittivity sensitivity w.r.t. design variable.

Reference: Old codebase Gradients.jl:∂ϵd_∂pf
  = 2 * (nm - nf) * (1 - iα) * (nf + (nm - nf) * pt)

Note: The absorption factor (1 - iα) must be included to match the
permittivity function ε_design_wf.
"""
function ∂ε_∂p(p, phys)
    2 * (phys.nm - phys.nf) * (1 - im * phys.α) * (phys.nf + (phys.nm - phys.nf) * p)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Curl helper
# ═══════════════════════════════════════════════════════════════════════════════

"""Curl of a 3D vector field from its gradient tensor."""
function curl_op(grad_u)
    c1 = grad_u[2, 3] - grad_u[3, 2]
    c2 = grad_u[3, 1] - grad_u[1, 3]
    c3 = grad_u[1, 2] - grad_u[2, 1]
    VectorValue(c1, c2, c3)
end

"""
    shifted_curl_op(u, k, θ)

Bloch-shifted curl used for oblique incidence:
  ∇ₛ = ∇ + i k sin(θ) êy
  ∇ₛ × u = ∇ × u + i k sin(θ) (êy × u)
"""
function shifted_curl_op(u, k::Real, θ::Real)
    ∇ₛ = ∇ + im * k * sind(θ) * VectorValue(0.0, 1.0, 0.0)
    return curl_op ∘ (∇ₛ(u))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PhysicalParams

Physical parameters for a single frequency solve.
"""
struct PhysicalParams
    ω::Float64               # Angular frequency (= 2π/λ)
    θ::Float64               # Incidence angle (degrees)
    nf::ComplexF64           # Fluid index
    nm::ComplexF64           # Metal index
    ns::ComplexF64           # Substrate index
    μ::Float64               # Permeability (= 1.0)
    des_low::Float64         # z-coord of design region bottom
    des_high::Float64        # z-coord of design region top
    α::Float64               # Artificial absorption
    bot_PEC::Bool        # bottom PEC or not
end

"""
    assemble_maxwell(pt, sim, phys) -> SparseMatrix

Assemble Maxwell curl-curl matrix with Sommerfeld ABC.

A = ∫ (∇×v)·(∇×u) dΩ - k²∫ ε(p)v·u dΩ + ik√ε ∫ v·u dS (ABC)
"""
function assemble_maxwell(pt, sim, phys::PhysicalParams)
    k = phys.ω

    # Permittivity as composed functions
    ε₀ = x -> ε_background_wf(x, phys)
    sqrt_ε₀ = sqrt ∘ ε₀
    εₘ = (p -> ε_design_wf(p, phys)) ∘ pt

    bot_PEC = phys.bot_PEC ? 0.0 : 1.0

    A_mat = assemble_matrix(sim.U, sim.V) do u, v
        (
            # Shifted curl-curl term for oblique incidence (legacy ∇ₛ formulation)
            ∫(shifted_curl_op(v, k, phys.θ) ⋅ shifted_curl_op(u, k, phys.θ))sim.dΩ -
            # Volume terms with background permittivity
            (k^2) * ∫(v ⋅ (ε₀ * u))sim.dΩ -
            # Design region permittivity
            (k^2) * ∫(v ⋅ (εₘ * u))sim.dΩ_design +
            # Sommerfeld ABC (top and bottom)
            +im * k * cosd(phys.θ) * ∫(v ⋅ (u * sqrt_ε₀))sim.dS_top +
            +bot_PEC * 1im * k * cosd(phys.θ) * ∫(v ⋅ (u * sqrt_ε₀))sim.dS_bottom
        )
    end

    return A_mat
end

"""
    assemble_source(sim, phys; source_y=true) -> Vector

Assemble source vector for plane wave incidence.
"""
function assemble_source(sim, phys::PhysicalParams; source_y::Bool=true)
    Jxy = im * phys.ω * cosd(phys.θ)

    pol = source_y ? VectorValue(0, Jxy, 0) : VectorValue(Jxy, 0, 0)

    b_vec = assemble_vector(sim.V) do v
        ∫(v ⋅ pol)sim.dΓ_source
    end

    return b_vec
end

# ═══════════════════════════════════════════════════════════════════════════════
# Material sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

"""
    assemble_material_sensitivity_pf(E, λ, pf, pt, sim, phys, control; space=sim.Pf) -> Vector

Compute ∂g/∂pf with SSP chain included:
  ∂A/∂pf = -k² (∂ε/∂pt) * (dpt/dpf)

The directional derivative dpt/dpf is provided by `dpt_dpf` (SSP only).
"""
function assemble_material_sensitivity_pf(E, λ, pf, pt, sim, phys::PhysicalParams, control; space=sim.Pf)
    k = phys.ω
    ∂εε = p -> ∂ε_∂p(p, phys)
    ∇pf = ∇(pf)

    ∂g_∂pf = assemble_vector(space) do v
        (k^2) * ∫(
            real(
                (((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf, ∇pf)) *
                (∂εε ∘ pt) *
                (conj(λ) ⋅ E)
            )
        )sim.dΩ_design
    end

    return ∂g_∂pf
end

"""
    assemble_material_sensitivity_pt(E, λ, pt, sim, phys; space=sim.Pf) -> Vector

Legacy-style material sensitivity with respect to projected design `pt`.
This excludes any SSP chain term (dpt/dpf), matching old foundry smoothing flow.
"""
function assemble_material_sensitivity_pt(E, λ, pt, sim, phys::PhysicalParams; space=sim.Pf)
    k = phys.ω
    ∂εε = p -> ∂ε_∂p(p, phys)

    ∂g_∂pt = assemble_vector(space) do v
        (k^2) * ∫(
            real(
                v *
                (∂εε ∘ pt) *
                (conj(λ) ⋅ E)
            )
        )sim.dΩ_design
    end

    return ∂g_∂pt
end
