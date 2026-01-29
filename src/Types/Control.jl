"""
    Control

Hyperparameters for the topology optimization pipeline: filtering, projection,
and linewidth constraints.

## Filtering
- `use_filter` — Enable filtering step
- `R_filter` — Filter radius (rx, ry, rz) in nm
- `use_dct` — Use DCT convolution (true) or Helmholtz PDE (false)

## Projection / SSP
- SSP on the FEM mesh is always used (pf → pt).
- `β` — Projection steepness (higher = sharper, use continuation)
- `η` — Threshold point (typically 0.5)
- `R_ssp` — SSP smoothing radius

## Linewidth Constraints
- `use_constraints` — Enable erosion/dilation constraints
- `η_erosion` — Erosion threshold (typically 0.75)
- `η_dilation` — Dilation threshold (typically 0.25)
- `b1` — Constraint tolerance
- `c0` — Indicator function steepness
"""
Base.@kwdef mutable struct Control
    # ═══ Filtering ═══
    use_filter::Bool = true
    R_filter::NTuple{3,Float64} = (20.0, 20.0, 20.0)
    use_dct::Bool = true

    # ═══ Projection (legacy flags; SSP is always used) ═══
    use_projection::Bool = true
    β::Float64 = 8.0
    η::Float64 = 0.5

    # ═══ SSP ═══
    use_ssp::Bool = false
    R_ssp::Float64 = 11.0

    # ═══ Linewidth Constraints ═══
    use_constraints::Bool = false
    η_erosion::Float64 = 0.75
    η_dilation::Float64 = 0.25
    b1::Float64 = 1e-8
    c0::Float64 = -1.0

    # ═══ E-field regularization ═══
    use_damage::Bool = false
    γ_damage::Float64 = 1.0
    E_threshold::Float64 = Inf

    # ═══ Objective Integration Domains ═══
    flag_volume::Bool = true   # Volume integral over Raman region
    flag_surface::Bool = false  # Surface integral (target plane)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Projection functions (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tanh_projection(x, η, β) -> Real

Heaviside-ish projection: smooth step from 0 to 1 centered at η.
As β → ∞, approaches true Heaviside function.
"""
function tanh_projection(x::Real, η::Real, β::Real)
    if isinf(β)
        return x > η ? 1.0 : 0.0
    end
    return (tanh(β * η) + tanh(β * (x - η))) / (tanh(β * η) + tanh(β * (1.0 - η)))
end

"""Vectorized tanh projection for arrays."""
tanh_projection(x::AbstractArray, η::Real, β::Real) = tanh_projection.(x, η, β)

"""Convenience with Control struct."""
tanh_projection(x, control::Control) = tanh_projection(x, control.η, control.β)

"""
    Threshold(pf; control) -> Real

Tanh projection for lazy composition with Gridap's `∘` operator.
This is the function form used in ComposedFunctions.

Reference: Old codebase Controls.jl:Threshold
"""
function Threshold(pf::Real; control)
    tanh_projection(pf, control.η, control.β)
end

"""
    ∂projection_∂pf(pf, β, η) -> Real

Derivative of tanh projection w.r.t. filtered density.
"""
function ∂projection_∂pf(pf::Real, β::Real, η::Real)
    if isinf(β)
        return 0.0
    end
    return β * (1.0 - tanh(β * (pf - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

"""Check that control parameters are valid."""
function validate(c::Control)
    @assert all(r > 0 for r in c.R_filter) "Filter radius must be positive"
    @assert c.β > 0 "Projection steepness β must be positive"
    @assert 0 < c.η < 1 "Threshold η must be in (0, 1)"
    @assert 0 < c.η_dilation < c.η < c.η_erosion < 1 "Must have η_dilation < η < η_erosion"
    return true
end
