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
        # Flat-field branch. The `* p` test-function factor is REQUIRED here, exactly
        # as in the non-flat branch below (dd_dpf = -(p / R) / norm∇ρ̃). Without it each
        # P1 nodal entry receives the full cell integral instead of its 1/4 share, so
        # the gradient is exactly 4x too large wherever the filtered field is flat --
        # i.e. everywhere at uniform-0.5 init, and dormant at developed designs.
        return ∂projection_∂pf(ρ̃, β, η) * p
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
    if norm∇ρ̃ < 1e-8
        return 0.0
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
    project_ssp(pf, control, trian=nothing) -> ComposedFunction

Subpixel-smoothed projection on the FEM mesh (SSP is ALWAYS used).
Returns a ComposedFunction for lazy evaluation in Gridap.
"""
function project_ssp(pf, control, trian=nothing)
    ∇pf = ∇(pf)
    R, β, η = control.R_ssp, control.β, control.η
    if getfield_default(control, :use_sidewall, false)
        trian === nothing && error(
            "project_ssp: control.use_sidewall=true needs a triangulation to read z from; " *
            "pass sim.Ω (the sidewall lean is a function of the physical height).")
        slope, z0 = control.sidewall_slope, control.sidewall_z0
        x = Gridap.CellData.get_physical_coordinate(trian)
        return ((pf, ∇pf, xx) ->
                smooth_project_sidewall(pf, ∇pf, xx[3]; R, β, η, slope, z0)) ∘ (pf, ∇pf, x)
    end
    return ((pf, ∇pf) -> smooth_project(pf, ∇pf; R, β, η)) ∘ (pf, ∇pf)
end

"""
    getfield_default(x, name, default)

Read an optional Control field, tolerating a Control built before it existed
(checkpoints and saved problems predate `use_sidewall`).
"""
@inline getfield_default(x, name::Symbol, default) =
    hasproperty(x, name) ? getproperty(x, name) : default

"""
    dpt_dpf_field(v, pf, control, trian) -> CellField

The dpt/dpf factor used inside every ∂g/∂pf assembly, with or without sidewalls.
Call sites pass their own triangulation; it is only read when sidewalls are on.
"""
function dpt_dpf_field(v, pf, control, trian=nothing)
    ∇pf = ∇(pf)
    if getfield_default(control, :use_sidewall, false)
        trian === nothing && error("dpt_dpf_field: use_sidewall=true needs a triangulation")
        x = Gridap.CellData.get_physical_coordinate(trian)
        return ((p, ∇p, q, ∇q, xx) ->
                dpt_dpf_sidewall(p, ∇p, q, ∇q, xx[3]; control)) ∘ (v, ∇(v), pf, ∇pf, x)
    end
    return ((p, ∇p, q, ∇q) -> dpt_dpf(p, ∇p, q, ∇q; control)) ∘ (v, ∇(v), pf, ∇pf)
end

"""
    project_fe(pf_vec, sim, control) -> ComposedFunction

Convenience wrapper: build FEFunction from vector and apply SSP projection.
"""
function project_fe(pf_vec::Vector{Float64}, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)
    return project_ssp(pf, control, sim.Ω)
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
        ∫(dpt_dpf_field(v, pf, control, sim.Ω_design) * ∂g_∂pt_h)sim.dΩ_design
    end

    return ∂g_∂pf
end

# ═══════════════════════════════════════════════════════════════════════════════
# Linear sidewalls (optional; foundry 2D-DOF extrusion)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A 2D-DOF design is extruded in z. With `control.use_sidewall` the extrusion is
# no longer z-invariant: every wall leans by a fixed angle, modelling the taper a
# real process leaves behind instead of a perfectly vertical wall.
#
# The SSP already computes the first-order (eikonal) signed distance from a point
# to the η level-set, d = (η − ρ̃)/|∇ρ̃|, positive in void. A linear sidewall is
#
#     d_sw(x,y,z) = d(x,y) + s·(z − z₀)                                       (1)
#
# and (1) is *identically* the unmodified SSP evaluated at a z-shifted density
#
#     ρ̃_sw = ρ̃ − |∇ρ̃|·s·(z − z₀)                                             (2)
#
# since (η − ρ̃_sw)/|∇ρ̃| = d + s(z−z₀). Implementing it as (2) means the fill
# factor, both sub-cell densities, the β→∞ limit and the adjoint are the code
# below, unchanged; only the chain rule through (2) is new. It also makes the
# flag-off path bit-identical by construction: s = 0 ⇒ ρ̃_sw = ρ̃.
#
# Two properties fall out of (2) for free:
#   * where the design is flat |∇ρ̃| → 0, so the shift vanishes — no wall there,
#     no tilt, and no material invented in bulk metal or bulk void;
#   * at a hole the sign of ∇ρ̃ flips but |∇ρ̃| > 0 always, so the hole grows with
#     z while the outer edge shrinks: material recedes at every wall.
#
# `s = cot(θ)` for a sidewall θ degrees from horizontal; s > 0 tapers inward going
# up (the ordinary etch profile), s < 0 is re-entrant (lift-off-like).
#
# NOTE ON THE HESSIAN TERM (SSP1 here, SSP2 in CutSSP — the difference is FE order).
# CutSSP's SSP2 uses g_τ = √(|∇ρ̃|² + R²‖Hρ̃‖²_F + τ²) to stay conditioned at saddle
# points, and it can, because its filtered field is P2 (CutSSP raman.jl:276 picks
# order 2 whenever SSP is on) and it recovers ∇ρ̃ and Hρ̃ by L2 projection onto P1
# vector/tensor spaces (raman.jl:305-318, filter.jl:98-99), with hess_mass' for the
# adjoint and DSP_dHpf closing the chain.
#
# DEO's `sim.Pf` is Lagrangian order 1 (Simulation.jl:122), so ∇∇ρ̃ ≡ 0 elementwise
# and that projection would return zero. The Hessian term is therefore unavailable
# AT THIS FE ORDER, not unavailable in principle: making DEO P2 like CutSSP would
# supply it, at the cost of moving every production number.
#
# Sidewalls are indifferent to this. The construction is written in terms of g_τ;
# today g_τ = |∇ρ̃| on the mesh, and a P2 DEO would pick up the Hessian term by
# changing one expression. (The standalone prototype in scripts/experiments does
# both, since on a Cartesian grid the Hessian is a finite-difference stencil.)
#
# Inherited caveat: DEO uses ∇(pf) of a P1 field directly, which is elementwise
# constant and discontinuous across faces, where CutSSP L2-projects it to a
# continuous space. The wall position is proportional to |∇ρ̃|, so it inherits that
# discontinuity — small element-to-element jumps in the tilted wall.

"""
    sidewall_shift(norm∇ρ̃, z; slope, z0) -> Real

The lateral lean written as a density offset: equation (2)'s `|∇ρ̃|·s·(z−z₀)`.
"""
@inline sidewall_shift(norm∇ρ̃, z; slope, z0) = norm∇ρ̃ * slope * (z - z0)

"""
    smooth_project_sidewall(ρ̃, ∇ρ̃, z; R, β, η, slope, z0) -> Real

SSP with a linear sidewall. `slope == 0` reproduces `smooth_project` exactly.
"""
function smooth_project_sidewall(ρ̃::Real, ∇ρ̃, z::Real; R::Float64, β::Float64,
                                 η::Float64, slope::Float64, z0::Float64)
    # A flat wall is the existing projection. Dispatch rather than rely on the
    # algebra collapsing: `smooth_project` writes the sub-cell offset as
    # (∇ρ̃⋅n̂)R and this writes it as |∇ρ̃|R, which agree mathematically but differ
    # in the last ulp for ~4% of inputs. Taking the original path makes
    # "flag on at slope 0" bit-identical to "flag off" by construction.
    slope == 0.0 && return smooth_project(ρ̃, ∇ρ̃; R, β, η)

    T = typeof(ρ̃)

    if ρ̃ == 0.0 && ∇ρ̃ == zero(∇ρ̃)
        return T(0.0)
    end

    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        return T(tanh_projection(ρ̃, η, β))   # flat: no wall, so nothing to lean
    end

    # band measured perpendicular to the tilted wall, not horizontally
    g  = norm∇ρ̃ * sqrt(1 + slope^2)
    ρ̃s = ρ̃ - sidewall_shift(norm∇ρ̃, z; slope, z0)

    d = ((η - ρ̃s) / g) / R
    fill_factor = if d < -1
        T(1.0)
    elseif d > 1
        T(0.0)
    else
        0.5 - 15 / 16 * d + 5 / 8 * d^3 - 3 / 16 * d^5
    end

    x_plus  = ρ̃s + g * R
    x_minus = ρ̃s - g * R

    (1 - fill_factor) * tanh_projection(x_minus, η, β) +
    fill_factor * tanh_projection(x_plus, η, β)
end

"""
    sidewall_partials(ρ̃, ∇ρ̃, z; R, β, η, slope, z0) -> (∂pt/∂ρ̃, ∂pt/∂|∇ρ̃|)

Closed-form partials of `smooth_project_sidewall`.

`|∇ρ̃|` is a *differentiated input* here, not a constant: besides setting the SSP
band it also sets how far the wall leans, so its sensitivity carries real signal.
Measured on the band, |∂pt/∂|∇ρ̃|| is a median 2.8× |∂pt/∂ρ̃| — dropping it would
lose most of the gradient, not a small correction.
"""
function sidewall_partials(ρ̃::Real, ∇ρ̃, z::Real; R::Float64, β::Float64,
                           η::Float64, slope::Float64, z0::Float64)
    if ρ̃ == 0.0 && ∇ρ̃ == zero(∇ρ̃)
        return 0.0, 0.0
    end
    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        return ∂projection_∂pf(ρ̃, β, η), 0.0
    end

    k  = sqrt(1 + slope^2)
    g  = norm∇ρ̃ * k
    ρ̃s = ρ̃ - sidewall_shift(norm∇ρ̃, z; slope, z0)

    d = ((η - ρ̃s) / g) / R
    ff, f′ = if d < -1
        1.0, 0.0
    elseif d > 1
        0.0, 0.0
    else
        (0.5 - 15/16*d + 5/8*d^3 - 3/16*d^5, -15/16 + 15/8*d^2 - 15/16*d^4)
    end

    x_plus, x_minus = ρ̃s + g*R, ρ̃s - g*R
    Tp,  Tm  = tanh_projection(x_plus, η, β),   tanh_projection(x_minus, η, β)
    dTp, dTm = ∂projection_∂pf(x_plus, β, η),   ∂projection_∂pf(x_minus, β, η)

    # w.r.t. the shifted density ρ̃s, then w.r.t. the SSP's own band g
    a     = f′ * (-1/(g*R)) * (Tp - Tm) + (1 - ff)*dTm + ff*dTp
    b_ssp = f′ * (-d/g)     * (Tp - Tm) + (1 - ff)*dTm*(-R) + ff*dTp*R

    # chain (2): ρ̃s and g both depend on |∇ρ̃|
    ∂pt_∂ρ̃  = a
    ∂pt_∂g₀ = a * (-slope * (z - z0)) + b_ssp * k
    return ∂pt_∂ρ̃, ∂pt_∂g₀
end

"""
    dpt_dpf_sidewall(p, ∇p, pf, ∇pf, z; control) -> Real

Directional derivative of the sidewall SSP w.r.t. pf in direction (p, ∇p) —
the drop-in replacement for `dpt_dpf` when `control.use_sidewall` is set.
"""
function dpt_dpf_sidewall(p, ∇p, pf, ∇pf, z; control)
    R, β, η = control.R_ssp, control.β, control.η
    slope, z0 = control.sidewall_slope, control.sidewall_z0

    slope == 0.0 && return dpt_dpf(p, ∇p, pf, ∇pf; control)   # see smooth_project_sidewall

    if pf == 0.0 && ∇pf == zero(∇pf)
        return 0.0
    end
    norm∇pf = sqrt(∇pf ⋅ ∇pf)
    a, c = sidewall_partials(pf, ∇pf, z; R, β, η, slope, z0)
    norm∇pf < 1e-8 && return a * p
    return a * p + c * ((∇pf ⋅ ∇p) / norm∇pf)
end
