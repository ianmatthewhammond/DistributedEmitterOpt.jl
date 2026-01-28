"""
    GradientCoordinator

Gradient coordinator for the new architecture (OptimizationProblem).
Uses solve_forward!, solve_adjoint!, pde_sensitivity from MaxwellSolver.jl
and compute_objective, compute_adjoint_sources, explicit_sensitivity from objectives.

This coordinates the adjoint method:
1. p → pf (filter) → pt (SSP project)
2. Solve forward Maxwell for all field configs
3. Compute objective from fields
4. Compute adjoint sources ∂g/∂E
5. Solve adjoint Maxwell
6. Assemble gradient: ∂g/∂pf = PDE sensitivity + explicit sensitivity
7. Chain back through filter/projection

Works with both 2D (foundry) and 3D DOF modes.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Main gradient interface for OptimizationProblem
# ═══════════════════════════════════════════════════════════════════════════════

"""
    objective_and_gradient!(∇g, p, prob::OptimizationProblem) → Float64

Unified forward+adjoint pass for the new architecture.
Dispatches based on foundry_mode.
"""
function objective_and_gradient!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    # TODO: upgrade to design-aware cache invalidation (hash/version) instead of always clearing.
    clear_maxwell_factors!(prob.pool)
    if prob.foundry_mode
        return compute_gradient_2d_opt!(∇g, p, prob)
    else
        return compute_gradient_3d_opt!(∇g, p, prob)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2D DOF (foundry) gradient
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_gradient_2d_opt!(∇g, p, prob::OptimizationProblem) → Float64

Compute gradient for 2D DOF mode using new architecture.
"""
function compute_gradient_2d_opt!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    (; pde, objective, sim, pool, control) = prob

    # ═══ Step 1: Filter on grid ═══
    pf_vec = filter_grid(p, sim, control)

    # ═══ Step 2: Interpolate onto FEM mesh ═══
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    sim.grid.params[:, :] = reshape(pf_vec, (nx, ny))
    pf = interpolate(r -> pf_grid(r, sim.grid), sim.Pf)

    # ═══ Step 3: SSP projection on mesh ═══
    pt = project_ssp(pf, control)

    # ═══ Step 4: Solve forward Maxwell for all configs ═══
    fields = solve_forward!(pde, pt, sim, pool)

    # ═══ Step 5: Compute objective ═══
    g = compute_objective(objective, pde, fields, pt, sim)

    # ═══ Step 6: Compute adjoint sources ═══
    sources = compute_adjoint_sources(objective, pde, fields, pt, sim)

    # ═══ Step 7: Solve adjoint Maxwell ═══
    adjoints = solve_adjoint!(pde, sources, sim, pool)

    # ═══ Step 8: PDE sensitivity (∂A/∂pf from adjoint) ═══
    ∂g_∂pf_pde = pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control; space=sim.P)

    # ═══ Step 9: Explicit sensitivity (∂g/∂pf direct) ═══
    ∂g_∂pf_explicit = explicit_sensitivity(objective, pde, fields, pf, pt, sim, control)

    # ═══ Step 10: Total sensitivity ═══
    ∂g_∂pf_vec = ∂g_∂pf_pde .+ ∂g_∂pf_explicit

    # ═══ Step 11: Grid Jacobian — mesh quadrature → grid params ═══
    getjacobian!(sim.grid)
    ∂g_∂pf_grid = sim.grid.jacobian * ∂g_∂pf_vec

    # ═══ Step 12: Chain through filter ═══
    ∂g_∂p = filter_grid_adjoint(∂g_∂pf_grid, sim, control)

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

# ═══════════════════════════════════════════════════════════════════════════════
# 3D DOF gradient (placeholder — implement if needed)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_gradient_3d_opt!(∇g, p, prob::OptimizationProblem) → Float64

Compute gradient for 3D DOF mode using new architecture.
"""
function compute_gradient_3d_opt!(∇g::Vector{Float64}, p::Vector{Float64},
    prob::OptimizationProblem)
    (; pde, objective, sim, pool, control) = prob

    # ═══ Step 1: Filter (Helmholtz PDE) ═══
    pf_vec = filter_helmholtz!(p, pool.filter_cache, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)

    # ═══ Step 2: SSP projection ═══
    pt = project_ssp(pf, control)

    # ═══ Step 3: Solve forward Maxwell ═══
    fields = solve_forward!(pde, pt, sim, pool)

    # ═══ Step 4: Compute objective ═══
    g = compute_objective(objective, pde, fields, pt, sim)

    # ═══ Step 5: Compute adjoint sources ═══
    sources = compute_adjoint_sources(objective, pde, fields, pt, sim)

    # ═══ Step 6: Solve adjoint Maxwell ═══
    adjoints = solve_adjoint!(pde, sources, sim, pool)

    # ═══ Step 7: PDE sensitivity ═══
    ∂g_∂pf = pde_sensitivity(pde, fields, adjoints, pf, pt, sim, control; space=sim.Pf)

    # ═══ Step 8: Explicit sensitivity ═══
    ∂g_∂pf .+= explicit_sensitivity(objective, pde, fields, pf, pt, sim, control)

    # ═══ Step 9: Chain through filter ═══
    ∂g_∂p = filter_helmholtz_adjoint!(∂g_∂pf, pool.filter_cache, sim, control)

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
