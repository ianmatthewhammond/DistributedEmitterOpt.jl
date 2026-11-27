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

_tanhc(u) = iszero(u) ? one(u) : tanh(u) / u   # tanh(u)/u, with its limit 1 at u = 0

# sech²(t)/D′ from a = 2|t| and D′, in two forms (see `∂projection_∂pf_stable`).
# Direct: sech²(t) = 4e^{−a}/(1 + e^{−a})², relative error a few eps once a is given.
_endpoint_slope_direct(a, den) = (q = exp(-a); 4 * q / (1 + q)^2 / den)
# Log space: e^{−a} and 1/D′ meet in one exponent, so neither underflows or overflows alone.
_endpoint_slope_log(a, den) = exp((log(oftype(a, 4)) - 2 * log1p(exp(-a))) - log(den) - a)

# Largest a = 2|t| that takes the direct form. e^{−600} ≈ 2.6e-261 is a normal Float64 with
# about 108 natural-log units of margin above the smallest normal number (e^{−708.4}), so the
# direct form loses no digits to underflow; and since D′ ≤ 1 the slope, which is at least
# e^{−a}, is normal too. The log form is used only above it, where e^{−a} approaches or falls
# below the normal range. The value assumes Float64 or a wider type, which every caller in the
# package passes (`Control` stores β and η as Float64).
const _ENDPOINT_SLOPE_DIRECT_MAX = 600

"""
    ∂projection_∂pf_stable(x, β, η) -> Real

The derivative of `tanh_projection(x, η, β)` with respect to `x`, the same quantity as
`∂projection_∂pf(x, β, η)`, evaluated as sech²(t)/D′ with

    t = β(x − η),   D′ = η·tanhc(βη) + (1 − η)·tanhc(β(1 − η)) = (tanh(βη) + tanh(β(1 − η)))/β,
    tanhc(u) = tanh(u)/u,   sech²(t) = 4e^{−2|t|}/(1 + e^{−2|t|})².

The form in `∂projection_∂pf`, β(1 − tanh²(β(x − η)))/(tanh(βη) + tanh(β(1 − η))), loses digits
to cancellation as tanh approaches ±1 and all of them once it rounds to ±1 (|β(x − η)| above
about 19 in Float64, where it returns 0 although the slope is still representable), and its
denominator rounds to 0 when βη and β(1 − η) underflow at the smallest positive β.

For a = 2|t| ≤ 600 the quotient is formed directly from e^{−a}, which is then a normal number.
Its relative error in Float64 is a few eps plus the rounding of β(x − η) amplified about a times
by the exponential (none when β(x − η) is exact, as for η = 0.5 and β a power of 2). Above that,
where e^{−a} alone would lose digits or round to 0 while the slope may still be representable
(subnormal, or normal when 1/D′ is large), it is evaluated in log space as
exp(log 4 − a − 2·log1p(e^{−a}) − log D′). There the relative error is about (a + |log D′| + 4)·eps,
because rounding occurs in forming the exponent before it is exponentiated; a subnormal result has
the same relative error plus its final rounding. In units in the last place this can be hundreds
for a subnormal result (51 was the largest subnormal miss for integer β from 700 to 760 at
η = 0.5; other accepted inputs miss by hundreds). The result is 0 only where the slope is below
about the smallest positive number. At β = ∞ it returns 0, as `∂projection_∂pf` does; as β → 0⁺
it tends to 1. These accuracy statements are for Float64 or wider arguments (the switch at
a = 600 assumes the Float64 range); it computes in the promoted floating type of its arguments.

This is the unit slope. With Float64 operands, products with the rounded unit slope can underflow or lose accuracy when that slope falls below the normal range (near β = 716 for η = 0.5).
"""
function ∂projection_∂pf_stable(x::Real, β::Real, η::Real)
    T = float(promote_type(typeof(x), typeof(β), typeof(η)))
    isinf(β) && return zero(T)
    xT, βT, ηT = T(x), T(β), T(η)
    a = 2 * abs(βT * (xT - ηT))                                     # 2|t|
    den = ηT * _tanhc(βT * ηT) + (1 - ηT) * _tanhc(βT * (1 - ηT))   # D′
    return a <= _ENDPOINT_SLOPE_DIRECT_MAX ? _endpoint_slope_direct(a, den) :
                                             _endpoint_slope_log(a, den)
end

"""
    flat_ssp_slope(ρ̃, ∇ρ̃, β, η) -> Real

Density slope of the SSP where the filtered field is flat (‖∇ρ̃‖ < 1e-8). There the forward
model is the formula `tanh_projection(ρ̃, η, β)`; at ρ̃ = 0 with ∇ρ̃ = 0 exactly, `smooth_project`
returns 0, which is that formula's value at 0 in exact arithmetic. That point, the feasible lower
bound of an all-void region, gets the formula's analytic one-sided slope into ρ̃ > 0 from
`∂projection_∂pf_stable`. The slope is positive at every finite β, and its Float64 value is 0
only where the slope is below about the smallest positive Float64. Returning 0 there made an
all-void design look stationary. At the smallest positive Float64 β with η = 0.5, a nearby flat Float64 density gives 0/0 in the forward formula. A higher-precision density can also expose rounding in the forward formula.
Every other flat point keeps `∂projection_∂pf`, unchanged.
"""
flat_ssp_slope(ρ̃, ∇ρ̃, β, η) =
    (ρ̃ == 0 && ∇ρ̃ == zero(∇ρ̃)) ? ∂projection_∂pf_stable(ρ̃, β, η) : ∂projection_∂pf(ρ̃, β, η)

"""
    DSP_dpf(p, ρ̃, ∇ρ̃; R, β, η) -> Real

Derivative ∂pt/∂pf for SSP adjoint.
"""
function DSP_dpf(p::Real, ρ̃::Real, ∇ρ̃; R::Float64, β::Float64, η::Float64)
    # ρ̃ = 0 with ∇ρ̃ = 0 takes the flat-field branch: see `flat_ssp_slope`.
    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        # Flat-field branch. The `* p` test-function factor is REQUIRED here, exactly
        # as in the non-flat branch below (dd_dpf = -(p / R) / norm∇ρ̃). Without it each
        # P1 nodal entry receives the full cell integral instead of its 1/4 share, so
        # the gradient is exactly 4x too large wherever the filtered field is flat --
        # i.e. everywhere at uniform-0.5 init, and dormant at developed designs.
        return flat_ssp_slope(ρ̃, ∇ρ̃, β, η) * p
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
    # A vertical wall (slope 0) takes the same path as no sidewall, so the two stay bit-identical.
    if getfield_default(control, :use_sidewall, false) && control.sidewall_slope != 0.0
        trian === nothing && error("dpt_dpf_field: use_sidewall=true needs a triangulation")
        x = Gridap.CellData.get_physical_coordinate(trian)
        return ((p, ∇p, q, ∇q, xx) ->
                dpt_dpf_sidewall(p, ∇p, q, ∇q, xx[3]; control)) ∘ (v, ∇(v), pf, ∇pf, x)
    end
    # A plain function of position (legacy foundry mode) keeps the per-test-function composition:
    # only there does Gridap turn the function into a field alongside the basis.
    pf isa Function && return ((p, ∇p, q, ∇q) -> dpt_dpf(p, ∇p, q, ∇q; control)) ∘ (v, ∇(v), pf, ∇pf)
    # dpt/dpf is linear in the test function: a·v + c·(∇v⋅∇pf), with a and c functions of pf alone.
    # Composing them separately evaluates the projection derivative once per quadrature point
    # instead of once per test function and point.
    R, β, η = control.R_ssp, control.β, control.η
    a = ((q, ∇q) -> DSP_dpf(1.0, q, ∇q; R, β, η)) ∘ (pf, ∇pf)
    c = ((q, ∇q) -> DSP_d∇pf_coefficient(q, ∇q; R, β, η)) ∘ (pf, ∇pf)
    return v * a + (∇(v) ⋅ ∇pf) * c
end

"""
    DSP_d∇pf_coefficient(ρ̃, ∇ρ̃; R, β, η) -> Float64

`DSP_d∇pf(∇p, ρ̃, ∇ρ̃)` depends on `∇p` only through `∇p ⋅ ∇ρ̃`; this is its value per unit of
that product, so that `DSP_d∇pf(∇p, ρ̃, ∇ρ̃) == DSP_d∇pf_coefficient(ρ̃, ∇ρ̃) * (∇p ⋅ ∇ρ̃)`.
"""
function DSP_d∇pf_coefficient(ρ̃::Real, ∇ρ̃; R::Float64, β::Float64, η::Float64)
    n2 = ∇ρ̃ ⋅ ∇ρ̃
    n2 < 1e-16 && return 0.0          # DSP_d∇pf is zero below a gradient norm of 1e-8
    return float(DSP_d∇pf(∇ρ̃ / n2, ρ̃, ∇ρ̃; R, β, η))
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
# The filtered density uses a first-order finite-element space; this projection
# uses its gradient and omits a Hessian term. A higher-order filter space with
# a matching derivative could support a Hessian-regularized transition band.

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
    # ρ̃ = 0 with ∇ρ̃ = 0 takes the flat branch, as in `DSP_dpf`: see `flat_ssp_slope`.
    norm∇ρ̃ = sqrt(∇ρ̃ ⋅ ∇ρ̃)
    if norm∇ρ̃ < 1e-8
        return flat_ssp_slope(ρ̃, ∇ρ̃, β, η), 0.0
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

    norm∇pf = sqrt(∇pf ⋅ ∇pf)
    a, c = sidewall_partials(pf, ∇pf, z; R, β, η, slope, z0)
    norm∇pf < 1e-8 && return a * p
    return a * p + c * ((∇pf ⋅ ∇p) / norm∇pf)
end
