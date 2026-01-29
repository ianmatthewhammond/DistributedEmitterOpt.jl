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

Finite difference gradient of 2D array (non-mutating, Zygote-safe).
"""
function compute_gradient(x::AbstractMatrix{Float64})
    dx_first = reshape(x[2, :] .- x[1, :], 1, :)
    dx_mid = (x[3:end, :] .- x[1:end-2, :]) ./ 2.0
    dx_last = reshape(x[end, :] .- x[end-1, :], 1, :)
    ∂x_∂i = vcat(dx_first, dx_mid, dx_last)

    dy_first = reshape(x[:, 2] .- x[:, 1], :, 1)
    dy_mid = (x[:, 3:end] .- x[:, 1:end-2]) ./ 2.0
    dy_last = reshape(x[:, end] .- x[:, end-1], :, 1)
    ∂x_∂j = hcat(dy_first, dy_mid, dy_last)

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

Legacy smoothed projection for linewidth constraints (ported from old code).
"""
function smoothed_projection(pf::AbstractMatrix{Float64}, η::Float64, β::Float64,
    gridx::Vector{Float64}, gridy::Vector{Float64})
    Lx = gridx[end] - gridx[1]
    Ly = gridy[end] - gridy[1]
    resolution_x = (length(gridx) - 1) / Lx
    resolution_y = (length(gridy) - 1) / Ly

    dx = 1 / resolution_x
    dy = 1 / resolution_y
    R_smoothing_x = 0.55 * dx
    R_smoothing_y = 0.55 * dy
    R_smoothing = max(R_smoothing_x, R_smoothing_y)

    ρ_projected = tanh_projection(pf, η, β)

    # Gradient of filtered field
    ∂pf_∂i, ∂pf_∂j = compute_gradient(pf)
    grad_helper = (∂pf_∂i ./ dx) .^ 2 .+ (∂pf_∂j ./ dy) .^ 2

    nonzero_norm = abs.(grad_helper) .> 0
    grad_norm = sqrt.(ifelse.(nonzero_norm, grad_helper, 1))
    grad_norm_eff = ifelse.(nonzero_norm, grad_norm, 1)

    d = (η .- pf) ./ grad_norm_eff
    needs_smoothing = nonzero_norm .& (abs.(d) .<= R_smoothing)

    d_R = d ./ R_smoothing
    F = ifelse.(
        needs_smoothing,
        0.5 .- 15/16 .* d_R .+ 5/8 .* d_R .^ 3 .- 3 / 16 .* d_R .^ 5,
        1,
    )
    F_minus = ifelse.(
        needs_smoothing,
        0.5 .+ 15/16 .* d_R .- 5/8 .* d_R .^ 3 .+ 3 / 16 .* d_R .^ 5,
        1,
    )

    ρ_minus = pf .- R_smoothing .* grad_norm_eff .* F
    ρ_plus = pf .+ R_smoothing .* grad_norm_eff .* F_minus

    ρ_plus_proj = tanh_projection(ρ_plus, η, β)
    ρ_minus_proj = tanh_projection(ρ_minus, η, β)

    ρ_smoothed = (1 .- F) .* ρ_minus_proj .+ F .* ρ_plus_proj
    return ifelse.(needs_smoothing, ρ_smoothed, ρ_projected)
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
