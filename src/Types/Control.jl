"""
    Control

Hyperparameters for the topology optimization pipeline: filtering, projection,
and linewidth constraints.

## Filtering
- `use_filter` — Enable filtering step
- `R_filter` — Filter radius (rx, ry, rz) in nm
- `use_dct` — Select the 2D convolution filter; must be `true` for 2D designs (see below)

The filter depends on where the design lives, not on `use_dct` alone:
- 2D designs (`foundry_mode=true`, design on the `(x, y)` grid): a conic kernel of radius
  `R_filter[1]` applied by DCT-I convolution, whose walls are mirrors. The design filter
  (`filter_grid`), its adjoint (`filter_grid_adjoint`) and the linewidth constraints
  (`glc_solid`, `glc_void`) all use it. `use_dct=false` is not supported here: those
  functions throw an `ArgumentError` asking for `use_dct=true` (`filter_grid` and its
  adjoint only when `use_filter` is on). So do the entry points that reach them, before
  they change the problem, its caches, the control or any output:
  - `optimize!` and `run_epoch!` for an `OptimizationProblem`, when `use_filter` is on or
    linewidth constraints are requested;
  - `init_random!` always (its random field is filtered whatever `use_filter` says);
  - `optimize!` and `objective_and_gradient!` for an `EigenOptimizationProblem`,
    `spectral_sweep`, `fabrication_sweep` and `visualize_results`, when `use_filter` is on;
  - `robust_sidewall_samples!` and `robust_sidewall_objective!` for these two problem types,
    when `use_filter` is on (a custom problem with its own `objective_and_gradient!` is
    not checked).
  Three calls that evaluate nothing are not checked and return as before: the eigen
  `optimize!` with an empty `β_schedule`, and either sweep with `points=0`. The
  `OptimizationProblem` `optimize!` checks the requested configuration before it loads a
  checkpoint, so `resume_from` a checkpoint whose epochs are all complete is refused as
  well, even when no evaluation would be needed. `evaluate`, `test_gradient`
  and `objective_and_gradient!` for an `OptimizationProblem` reach `filter_grid` first and
  throw there when `use_filter` is on. There is no periodic FFT filter.
- 3D designs (`foundry_mode=false`, design on the FE mesh): a Helmholtz PDE filter with the
  anisotropic radii `R_filter = (rx, ry, rz)` (`filter_helmholtz!`). `use_dct` is ignored;
  by convention 3D examples set `use_dct=false`.

## Projection / SSP
- SSP on the FEM mesh is always used (pf → pt).
- `β` — Projection steepness (higher = sharper, use continuation)
- `η` — Threshold point (typically 0.5)
- `R_ssp` — SSP smoothing radius
- `use_sidewall` / `sidewall_slope` / `sidewall_z0` — optional linear sidewall on
  the 2D-DOF extrusion (see Projection3D.jl). Requires SSP and 2D DOFs.

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
    R_ssp::Float64 = 0.55 * R_filter[1]
    foundry_projection_mode::Symbol = :current

    # ═══ Linear sidewalls (2D-DOF extrusion; SSP only) ═══
    # Off by default and bit-identical to the z-invariant extrusion when off.
    # `sidewall_slope = cot(θ)` for a wall θ degrees from horizontal: > 0 tapers
    # inward going up (etch-like), < 0 is re-entrant (lift-off-like).
    # `sidewall_z0` is the height at which the design's own linewidth is met.
    use_sidewall::Bool = false
    sidewall_slope::Float64 = 0.0
    sidewall_z0::Float64 = 0.0

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
    @assert c.foundry_projection_mode in (:current, :legacy) "foundry_projection_mode must be :current or :legacy"
    return true
end
