"""
    Projection3D

Subpixel-smoothed projection (SSP) for 3D DOF mode.
Uses gradient magnitude to smooth interface region.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SSP projection
# ═══════════════════════════════════════════════════════════════════════════════

"""
    smooth_project(ρ̃, ∇ρ̃; R, β, η) -> Real

Subpixel-smoothed projection at a point.
Interpolates between tanh projection and linear based on gradient magnitude.
"""
function smooth_project(ρ̃::Real, ∇ρ̃; R::Float64, β::Float64, η::Float64)
    T = typeof(ρ̃)

    if ρ̃ == 0.0 && ∇ρ̃ == zero(∇ρ̃)
        return T(0.0)
    end

    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        return T(tanh_projection(ρ̃, η, β))
    end

    d = ((η - ρ̃) / norm∇ρ̃) / R

    fill_factor = if d < -1
        T(1.0)
    elseif d > 1
        T(0.0)
    else
        0.5 - 15 / 16 * d + 5 / 8 * d^3 - 3 / 16 * d^5
    end

    n̂ = ∇ρ̃ / norm∇ρ̃
    x_plus = ρ̃ + (∇ρ̃ ⋅ n̂) * R
    x_minus = ρ̃ - (∇ρ̃ ⋅ n̂) * R

    x_plus_eff_projected = tanh_projection(x_plus, η, β)
    x_minus_eff_projected = tanh_projection(x_minus, η, β)

    (1 - fill_factor) * x_minus_eff_projected + fill_factor * x_plus_eff_projected
end

# ═══════════════════════════════════════════════════════════════════════════════
# SSP derivatives for adjoint
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DSP_dpf(p, ρ̃, ∇ρ̃; R, β, η) -> Real

Derivative ∂pt/∂pf for SSP adjoint.
"""
function DSP_dpf(p::Real, ρ̃::Real, ∇ρ̃; R::Float64, β::Float64, η::Float64)
    if ρ̃ == 0.0 && ∇ρ̃ == zero(∇ρ̃)
        return 0.0
    end

    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        return ∂projection_∂pf(ρ̃, β, η)
    end

    d = ((η - ρ̃) / norm∇ρ̃) / R

    fill_factor, dfill_factor_dd = if d < -1
        1.0, 0.0
    elseif d > 1
        0.0, 0.0
    else
        fill_factor = 0.5 - 15 / 16 * d + 5 / 8 * d^3 - 3 / 16 * d^5
        dfill_factor_dd = -15 / 16 + 5 / 8 * 3 * d^2 - 3 / 16 * 5 * d^4
        fill_factor, dfill_factor_dd
    end

    dd_dpf = -(p / R) / norm∇ρ̃
    dfill_factor_dpf = dfill_factor_dd * dd_dpf

    n̂ = ∇ρ̃ / norm∇ρ̃
    x_plus = ρ̃ + (∇ρ̃ ⋅ n̂) * R
    x_minus = ρ̃ - (∇ρ̃ ⋅ n̂) * R

    x_plus_eff_projected = tanh_projection(x_plus, η, β)
    x_minus_eff_projected = tanh_projection(x_minus, η, β)

    dx_plus_eff_projected_dpf = ∂projection_∂pf(x_plus, β, η) * p
    dx_minus_eff_projected_dpf = ∂projection_∂pf(x_minus, β, η) * p

    return (
        (1 - fill_factor) * dx_minus_eff_projected_dpf +
        fill_factor * dx_plus_eff_projected_dpf +
        dfill_factor_dpf * (x_plus_eff_projected - x_minus_eff_projected)
    )
end

"""
    DSP_d∇pf(∇p, ρ̃, ∇ρ̃; R, β, η) -> VectorValue

Derivative ∂pt/∂(∇pf) for SSP adjoint (contributes to gradient assembly).
"""
function DSP_d∇pf(∇p, ρ̃::Real, ∇ρ̃; R::Float64, β::Float64, η::Float64)
    if ρ̃ == 0.0 && ∇ρ̃ == zero(∇ρ̃)
        return 0.0
    end

    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    d = ((η - ρ̃) / norm∇ρ̃) / R

    fill_factor, dfill_factor_dd = if d < -1
        1.0, 0.0
    elseif d > 1
        0.0, 0.0
    else
        fill_factor = 0.5 - 15 / 16 * d + 5 / 8 * d^3 - 3 / 16 * d^5
        dfill_factor_dd = -15 / 16 + 5 / 8 * 3 * d^2 - 3 / 16 * 5 * d^4
        fill_factor, dfill_factor_dd
    end

    dd_dpf = -((η - ρ̃) / norm∇ρ̃^2) / R * (∇p ⋅ ∇ρ̃) / norm∇ρ̃
    dfill_factor_dpf = dfill_factor_dd * dd_dpf

    n̂ = ∇ρ̃ / norm∇ρ̃
    x_plus = ρ̃ + (∇ρ̃ ⋅ n̂) * R
    x_minus = ρ̃ - (∇ρ̃ ⋅ n̂) * R

    dx_plus_dpf = (∇p ⋅ ∇ρ̃) * R / norm∇ρ̃
    dx_minus_dpf = -(∇p ⋅ ∇ρ̃) * R / norm∇ρ̃

    x_plus_eff_projected = tanh_projection(x_plus, η, β)
    x_minus_eff_projected = tanh_projection(x_minus, η, β)

    dx_plus_eff_projected_dpf = ∂projection_∂pf(x_plus, β, η) * dx_plus_dpf
    dx_minus_eff_projected_dpf = ∂projection_∂pf(x_minus, β, η) * dx_minus_dpf

    return (
        (1 - fill_factor) * dx_minus_eff_projected_dpf +
        fill_factor * dx_plus_eff_projected_dpf +
        dfill_factor_dpf * (x_plus_eff_projected - x_minus_eff_projected)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main interface
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: Threshold function is defined in Types/Control.jl for use in lazy composition

"""
    project_ssp(pf, control) -> ComposedFunction

Subpixel-smoothed projection on the FEM mesh (SSP is ALWAYS used).
Returns a ComposedFunction for lazy evaluation in Gridap.
"""
function project_ssp(pf, control)
    ∇pf = ∇(pf)
    R, β, η = control.R_ssp, control.β, control.η
    return ((pf, ∇pf) -> smooth_project(pf, ∇pf; R, β, η)) ∘ (pf, ∇pf)
end

"""
    project_fe(pf_vec, sim, control) -> ComposedFunction

Convenience wrapper: build FEFunction from vector and apply SSP projection.
"""
function project_fe(pf_vec::Vector{Float64}, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)
    return project_ssp(pf, control)
end

"""
    dpt_dpf(p, ∇p, pf, ∇pf; control) -> Real

Directional derivative of SSP projection w.r.t. pf in direction (p, ∇p).
This is the analytic chain-rule factor used inside ∂g/∂pf assembly.
"""
function dpt_dpf(p, ∇p, pf, ∇pf; control)
    R, β, η = control.R_ssp, control.β, control.η
    return DSP_dpf(p, pf, ∇pf; R, β, η) + DSP_d∇pf(∇p, pf, ∇pf; R, β, η)
end

"""
    project_fe_adjoint!(∂g_∂pt, pf_vec, sim, control) -> Vector

Adjoint of FE projection: chain rule through pt projection.

Reference: Old codebase Gradients.jl:∂pt_∂pf_ and ∂A_∂pf patterns
"""
function project_fe_adjoint!(∂g_∂pt::Vector{Float64}, pf_vec::Vector{Float64}, sim, control)
    # SSP-only adjoint (kept for compatibility, not used in current pipeline)
    pf = FEFunction(sim.Pf, pf_vec)
    ∇pf = ∇(pf)
    ∂g_∂pt_h = FEFunction(sim.Pf, ∂g_∂pt)

    ∂g_∂pf = assemble_vector(sim.Pf) do v
        ∫((((p, ∇p, pf, ∇pf) -> dpt_dpf(p, ∇p, pf, ∇pf; control)) ∘ (v, ∇(v), pf, ∇pf)) *
          ∂g_∂pt_h)sim.dΩ_design
    end

    return ∂g_∂pf
end
