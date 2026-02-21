"""
    GradientCoordinator

Coordinates the full adjoint gradient pipeline:
  p -> filter -> project -> forward solve -> objective
                                              |
  gradient <- filter adjoint <- sensitivity <- adjoint solve

Works with both 2D (foundry) and 3D DOF modes.
"""

# ---------------------------------------------------------------------------
# Main gradient interface
# ---------------------------------------------------------------------------

"""
    objective_and_gradient!(grad, p, prob) -> Float64

Forward + adjoint pass. Dispatches on foundry_mode.
"""
function objective_and_gradient!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    # TODO: use design-aware cache invalidation (hash/version) instead of always clearing.
    clear_maxwell_factors!(prob.pool)
    if prob.foundry_mode
        return compute_gradient_2d_opt!(∇g, p, prob)
    else
        return compute_gradient_3d_opt!(∇g, p, prob)
    end
end

# ---------------------------------------------------------------------------
# 2D DOF (foundry) gradient
# ---------------------------------------------------------------------------

"""
    compute_gradient_2d_opt!(grad, p, prob) -> Float64

Full gradient computation for 2D foundry DOF.
"""
function compute_gradient_2d_opt!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    (; pde, objective, sim, pool, control) = prob
    sim0 = default_sim(sim)

    # Filter on grid
    pf_vec = filter_grid(p, sim0, control)

    local pt
    local ∂g_∂pf_vec
    if control.foundry_projection_mode == :legacy
        # Legacy foundry flow: apply SSP on 2D grid first, then interpolate pt to FEM.
        pt_vec = smoothed_projection_vec(pf_vec, control, sim0)
        pt = build_foundry_pf(pt_vec, sim0, control)
    else
        # Current flow: interpolate pf to FEM first, then apply FEM SSP.
        pf = build_foundry_pf(pf_vec, sim0, control)
        pt = project_ssp(pf, control)
    end

    # Forward Maxwell solves
    fields = solve_forward!(pde, pt, sim, pool)

    # Objective value
    g = compute_objective(objective, pde, fields, pt, sim)

    # Adjoint sources and solves
    sources = compute_adjoint_sources(objective, pde, fields, pt, sim)
    adjoints = solve_adjoint!(pde, sources, sim, pool)

    if control.foundry_projection_mode == :legacy
        # Legacy foundry sensitivity is accumulated directly to 2D grid pt DOFs.
        ∂g_∂pt_grid = zeros(Float64, length(p))
        pde_sensitivity_pt_grid!(∂g_∂pt_grid, pde, fields, adjoints, pt, sim)
        ∂g_∂pt_explicit = zeros(Float64, length(p))
        explicit_sensitivity_pt_grid!(∂g_∂pt_explicit, objective, pde, fields, pt, sim)
        ∂g_∂pt_grid .+= ∂g_∂pt_explicit

        # Chain through legacy grid SSP: pt_vec = SSP_grid(pf_vec).
        ∂g_∂pf_vec = smoothed_projection_adjoint(∂g_∂pt_grid, pf_vec, control, sim0)
    else
        # PDE sensitivity (dA/dpf from adjoint fields)
        ∂g_∂pf_pde = pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control; space=sim0.Pf)
        if any(isnan, ∂g_∂pf_pde)
            println("WARNING: NaNs in PDE sensitivity")
        end

        # Explicit sensitivity (dg/dpf, not through the PDE)
        ∂g_∂pf_explicit = explicit_sensitivity(objective, pde, fields, pf, pt, sim, control; space=sim0.Pf)
        if any(isnan, ∂g_∂pf_explicit)
            println("WARNING: NaNs in explicit sensitivity")
        end

        # Total sensitivity on mesh
        ∂g_∂pf_vec = ∂g_∂pf_pde .+ ∂g_∂pf_explicit

        # Map mesh sensitivities back to grid
        ∂g_∂pf_grid = similar(p)
        map_foundry_adjoint!(∂g_∂pf_grid, sim0, control, ∂g_∂pf_vec)
        ∂g_∂pf_vec = ∂g_∂pf_grid
    end

    # Chain through filter adjoint
    ∂g_∂p = filter_grid_adjoint(∂g_∂pf_vec, sim0, control)

    if !isempty(∇g)
        ∇g .= ∂g_∂p
    end

    # Store in problem state
    prob.g = g
    if !isempty(prob.∇g)
        prob.∇g .= ∂g_∂p
    end

    return g
end

# ---------------------------------------------------------------------------
# 3D DOF gradient
# ---------------------------------------------------------------------------

"""
    compute_gradient_3d_opt!(grad, p, prob) -> Float64

Full gradient computation for 3D mesh DOF.
"""
function compute_gradient_3d_opt!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    (; pde, objective, sim, pool, control) = prob
    sim0 = default_sim(sim)

    # Helmholtz PDE filter
    pf_vec = filter_helmholtz!(p, default_pool(pool).filter_cache, sim0, control)
    pf = FEFunction(sim0.Pf, pf_vec)

    # SSP projection
    pt = project_ssp(pf, control)

    # Forward Maxwell solves
    fields = solve_forward!(pde, pt, sim, pool)

    # Objective
    g = compute_objective(objective, pde, fields, pt, sim)

    # Adjoint sources and solves
    sources = compute_adjoint_sources(objective, pde, fields, pt, sim)
    adjoints = solve_adjoint!(pde, sources, sim, pool)

    # PDE + explicit sensitivities
    ∂g_∂pf = pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control; space=sim0.Pf)
    ∂g_∂pf .+= explicit_sensitivity(objective, pde, fields, pf, pt, sim, control; space=sim0.Pf)

    # Chain through filter adjoint
    ∂g_∂p = filter_helmholtz_adjoint!(∂g_∂pf, default_pool(pool).filter_cache, sim0, control)

    if !isempty(∇g)
        ∇g .= ∂g_∂p
    end

    # Store in problem state
    prob.g = g
    if !isempty(prob.∇g)
        prob.∇g .= ∂g_∂p
    end

    return g
end
