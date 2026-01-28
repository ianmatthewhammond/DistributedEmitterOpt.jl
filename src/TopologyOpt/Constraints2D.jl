"""
    Constraints2D

Grid-based linewidth constraints for 2D DOF mode.
Uses Zygote AD for gradient computation through the filter-project pipeline.
"""

import Zygote

# ═══════════════════════════════════════════════════════════════════════════════
# Indicator functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    indicator_solid(x, η, c0) -> Real

Solid linewidth indicator: penalizes small features in the solid phase.
Uses gradient magnitude to detect thin regions.
"""
function indicator_solid(x::AbstractMatrix, η::Float64, c0::Float64)
    # Smooth indicator
    h = @. 1.0 / (1.0 + exp(c0 * (x - η)))
    return mean(h)
end

"""
    indicator_void(x, η, c0) -> Real

Void linewidth indicator: penalizes small features in the void phase.
"""
function indicator_void(x::AbstractMatrix, η::Float64, c0::Float64)
    # Smooth indicator (reversed)
    h = @. 1.0 / (1.0 + exp(c0 * (η - x)))
    return mean(h)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Control functions (objective for constraint)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    control_solid(p; c0, ηe, filter_f, threshold_f, resolution) -> Real

Solid-phase constraint objective (should be ≤ 0 for feasibility).
"""
function control_solid(p::AbstractMatrix; c0::Float64, ηe::Float64,
    filter_f, threshold_f, resolution::Float64)
    # Filter and project
    pf = filter_f(p)
    pt = threshold_f(pf)

    # Compute gradient magnitude
    ∂x, ∂y = compute_gradient_2d(pt)
    ∇mag = @. sqrt(∂x^2 + ∂y^2) * resolution

    # Indicator
    h = indicator_solid(pt, ηe, c0)

    return h * mean(∇mag)
end

"""
    control_void(p; c0, ηd, filter_f, threshold_f, resolution) -> Real

Void-phase constraint objective.
"""
function control_void(p::AbstractMatrix; c0::Float64, ηd::Float64,
    filter_f, threshold_f, resolution::Float64)
    # Filter and project
    pf = filter_f(p)
    pt = threshold_f(pf)

    # Compute gradient magnitude
    ∂x, ∂y = compute_gradient_2d(pt)
    ∇mag = @. sqrt(∂x^2 + ∂y^2) * resolution

    # Indicator (1-pt for void)
    h = indicator_void(pt, ηd, c0)

    return h * mean(∇mag)
end

"""Finite difference gradient for 2D array."""
function compute_gradient_2d(x::AbstractMatrix)
    ni, nj = size(x)

    ∂x = similar(x)
    ∂x[2:end-1, :] = (x[3:end, :] .- x[1:end-2, :]) ./ 2
    ∂x[1, :] = x[2, :] .- x[1, :]
    ∂x[end, :] = x[end, :] .- x[end-1, :]

    ∂y = similar(x)
    ∂y[:, 2:end-1] = (x[:, 3:end] .- x[:, 1:end-2]) ./ 2
    ∂y[:, 1] = x[:, 2] .- x[:, 1]
    ∂y[:, end] = x[:, end] .- x[:, end-1]

    return ∂x, ∂y
end

flatten(x) = vec(x)

# ═══════════════════════════════════════════════════════════════════════════════
# NLopt constraint interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    glc_solid(x, grad; sim, control) -> Float64

Solid linewidth constraint for NLopt. Constraint satisfied when ≤ 0.
"""
function glc_solid(x::Vector{Float64}, grad::Vector{Float64}; sim, control)
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    resolution = (nx - 1) / Lx

    # Build filter/threshold functions
    R = control.R_filter[1]
    h = conic_filter(R, Lx, Ly, resolution)
    filter_f = p -> control.use_dct ? convolvedcti(p; h) : convolvefft(p; h)
    threshold_f = p -> tanh_projection(p, control.η, control.β)

    # Hyperparameters
    ηe = control.η_erosion
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M1 = p -> control_solid(p; c0, ηe, filter_f, threshold_f, resolution)

    # Gradient via Zygote
    if length(grad) > 0
        g1 = Zygote.gradient(M1, p)
        grad[:] = flatten(g1[1])
    end

    return M1(p) - b1
end

"""
    glc_void(x, grad; sim, control) -> Float64

Void linewidth constraint for NLopt.
"""
function glc_void(x::Vector{Float64}, grad::Vector{Float64}; sim, control)
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    Lx = sim.grid.x[end] - sim.grid.x[1]
    Ly = sim.grid.y[end] - sim.grid.y[1]
    resolution = (nx - 1) / Lx

    # Build filter/threshold functions
    R = control.R_filter[1]
    h = conic_filter(R, Lx, Ly, resolution)
    filter_f = p -> control.use_dct ? convolvedcti(p; h) : convolvefft(p; h)
    threshold_f = p -> tanh_projection(p, control.η, control.β)

    # Hyperparameters
    ηd = control.η_dilation
    b1 = control.b1
    c0 = 64 * R^2

    # Reshape
    p = reshape(x, (nx, ny))

    # Objective
    M2 = p -> control_void(p; c0, ηd, filter_f, threshold_f, resolution)

    # Gradient via Zygote
    if length(grad) > 0
        g2 = Zygote.gradient(M2, p)
        grad[:] = flatten(g2[1])
    end

    return M2(p) - b1
end
