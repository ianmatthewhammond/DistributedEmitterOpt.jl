"""
    Projection2D

Array-based projection functions for 2D DOF mode.
Includes smoothed projection for linewidth constraints.
"""

# NOTE: tanh_projection is defined in Types/Control.jl to avoid duplication

# ═══════════════════════════════════════════════════════════════════════════════
# Gradient computation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_gradient(x::Matrix) -> (∂x_∂i, ∂x_∂j)

Finite difference gradient of 2D array.
"""
function compute_gradient(x::AbstractMatrix{Float64})
    ni, nj = size(x)

    # ∂/∂i (central differences, forward/backward at edges)
    ∂x_∂i = similar(x)
    ∂x_∂i[2:end-1, :] = (x[3:end, :] .- x[1:end-2, :]) ./ 2.0
    ∂x_∂i[1, :] = x[2, :] .- x[1, :]
    ∂x_∂i[end, :] = x[end, :] .- x[end-1, :]

    # ∂/∂j
    ∂x_∂j = similar(x)
    ∂x_∂j[:, 2:end-1] = (x[:, 3:end] .- x[:, 1:end-2]) ./ 2.0
    ∂x_∂j[:, 1] = x[:, 2] .- x[:, 1]
    ∂x_∂j[:, end] = x[:, end] .- x[:, end-1]

    return ∂x_∂i, ∂x_∂j
end

"""Gradient magnitude."""
function gradient_magnitude(x::AbstractMatrix{Float64})
    ∂x_∂i, ∂x_∂j = compute_gradient(x)
    return sqrt.(∂x_∂i .^ 2 .+ ∂x_∂j .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Smoothed projection for constraints
# ═══════════════════════════════════════════════════════════════════════════════

"""
    smoothed_projection(pf, control, sim) -> Array

Gradient-aware smoothed projection for linewidth constraints.
This version works on arrays (2D DOF mode).
"""
function smoothed_projection(pf::AbstractMatrix{Float64}, η::Float64, β::Float64,
    gridx::Vector{Float64}, gridy::Vector{Float64})
    Lx = gridx[end] - gridx[1]
    Ly = gridy[end] - gridy[1]
    nx, ny = length(gridx), length(gridy)

    # Compute gradient magnitude
    ∂pf_∂i, ∂pf_∂j = compute_gradient(pf)

    # Scale to physical units
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    ∇pf_mag = sqrt.((∂pf_∂i ./ dx) .^ 2 .+ (∂pf_∂j ./ dy) .^ 2)

    # Standard projection
    pt = tanh_projection(pf, η, β)

    # Smooth based on gradient magnitude (high gradient = intermediate values)
    # This creates the "smoothing" effect at interfaces
    smooth_factor = @. 1.0 / (1.0 + ∇pf_mag)

    return pt .* smooth_factor .+ pf .* (1.0 .- smooth_factor)
end

"""Convenience wrapper with Control struct."""
function smoothed_projection(pf::AbstractMatrix{Float64}, control, sim)
    smoothed_projection(pf, control.η, control.β, sim.grid.x, sim.grid.y)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    project_grid(pf_vec, sim, control) -> Vector

Project filtered design to binary-ish values (2D DOF mode).
"""
function project_grid(pf_vec::Vector{Float64}, sim, control)
    if !control.use_projection
        return copy(pf_vec)
    end

    nx, ny = length(sim.grid.x), length(sim.grid.y)
    pf = reshape(pf_vec, (nx, ny))

    pt = tanh_projection(pf, control.η, control.β)

    return vec(pt)
end

"""
    project_grid_adjoint(∂g_∂pt, pf_vec, control) -> Vector

Adjoint of projection: chain rule through tanh.
"""
function project_grid_adjoint(∂g_∂pt::Vector{Float64}, pf_vec::Vector{Float64}, control)
    if !control.use_projection
        return copy(∂g_∂pt)
    end

    β, η = control.β, control.η

    if β == Inf
        return zeros(length(∂g_∂pt))
    end

    # ∂pt/∂pf = β(1 - tanh²(β(pf-η))) / (tanh(βη) + tanh(β(1-η)))
    denom = tanh(β * η) + tanh(β * (1.0 - η))
    ∂pt_∂pf = @. β * (1.0 - tanh(β * (pf_vec - η))^2) / denom

    return ∂g_∂pt .* ∂pt_∂pf
end
