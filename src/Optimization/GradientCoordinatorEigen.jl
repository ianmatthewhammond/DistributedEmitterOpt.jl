"""
    GradientCoordinatorEigen

Coordinates eigenvalue-based optimization:
  p -> filter -> project -> eigen solve -> objective
                                        |
                   gradient <- filter adjoint <- eigen sensitivity
"""

"""
    objective_and_gradient!(grad, p, prob::EigenOptimizationProblem) -> Float64

Forward + eigen sensitivity pass. Dispatches on foundry_mode.
"""
function objective_and_gradient!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::EigenOptimizationProblem)
    clear_eigen_factors!(prob.pool)
    if prob.foundry_mode
        return compute_gradient_eigen_2d!(∇g, p, prob)
    else
        return compute_gradient_eigen_3d!(∇g, p, prob)
    end
end

# ---------------------------------------------------------------------------
# 2D DOF (foundry) gradient
# ---------------------------------------------------------------------------

function compute_gradient_eigen_2d!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::EigenOptimizationProblem)
    (; pde, objective, sim, pool, control) = prob
    sim0 = default_sim(sim)

    pf_vec = filter_grid(p, sim0, control)

    nx, ny = length(sim0.grid.x), length(sim0.grid.y)
    sim0.grid.params[:, :] = reshape(pf_vec, (nx, ny))
    pf_vals = [pf_grid(node, sim0.grid) for node in sim0.grid.nodes]
    pf = FEFunction(sim0.Pf, pf_vals)

    pt = project_ssp(pf, control)

    result = solve_eigen!(pde, pt, sim, pool)

    g = compute_eigen_objective(objective, result)

    # TODO: eigen_sensitivity not implemented
    ∂g_∂pf_vec = eigen_sensitivity(objective, result, pf, pt, sim, control; space=sim0.Pf)

    ∂g_∂pf_grid = similar(p)
    apply_grid_adjoint!(∂g_∂pf_grid, sim0.grid, ∂g_∂pf_vec)
    ∂g_∂p = filter_grid_adjoint(∂g_∂pf_grid, sim0, control)

    if !isempty(∇g)
        ∇g .= ∂g_∂p
    end

    prob.g = g
    if !isempty(prob.∇g)
        prob.∇g .= ∂g_∂p
    end

    return g
end

# ---------------------------------------------------------------------------
# 3D DOF gradient
# ---------------------------------------------------------------------------

function compute_gradient_eigen_3d!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::EigenOptimizationProblem)
    (; pde, objective, sim, pool, control) = prob
    sim0 = default_sim(sim)

    pf_vec = filter_helmholtz!(p, default_pool(pool).filter_cache, sim0, control)
    pf = FEFunction(sim0.Pf, pf_vec)

    pt = project_ssp(pf, control)

    result = solve_eigen!(pde, pt, sim, pool)

    g = compute_eigen_objective(objective, result)

    # TODO: eigen_sensitivity not implemented
    ∂g_∂pf = eigen_sensitivity(objective, result, pf, pt, sim, control; space=sim0.Pf)

    ∂g_∂p = filter_helmholtz_adjoint!(∂g_∂pf, default_pool(pool).filter_cache, sim0, control)

    if !isempty(∇g)
        ∇g .= ∂g_∂p
    end

    prob.g = g
    if !isempty(prob.∇g)
        prob.∇g .= ∂g_∂p
    end

    return g
end
