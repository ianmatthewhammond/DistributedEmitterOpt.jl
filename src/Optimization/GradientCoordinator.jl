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
    # Keep cached factors and let solver backends refactor in-place each call.
    # This avoids repeated destroy/recreate cycles (notably costly for Pardiso)
    # while still updating matrix values for the current design.
    g = if prob.foundry_mode
        compute_gradient_2d_opt!(∇g, p, prob)
    else
        compute_gradient_3d_opt!(∇g, p, prob)
    end
    scale = default_sim(prob.sim).objective_normalization
    if scale != 1.0
        ∇g .*= scale
        ∇g === prob.∇g || (prob.∇g .*= scale)   # scale an aliased gradient once
        prob.g = scale*g
    end
    scale == 1.0 ? g : scale*g
end

# ---------------------------------------------------------------------------
# Physical density
# ---------------------------------------------------------------------------

"""
    physical_density(p, prob; filter_cache) -> (; pf_vec, pf, pt)

Map the design vector `p` to the physical density used by the forward solve,
exactly as the objective and gradient do:

- foundry mode, `:current`: grid filter, interpolation onto `Pf`, FEM SSP;
- foundry mode, `:legacy`: grid filter, grid SSP, interpolation of the projected grid;
- 3D mode: Helmholtz filter into `Pf`, FEM SSP.

`pf_vec` is the filtered vector (2D grid values in foundry mode, element values in unfiltered 3D, `Pf` DOFs in filtered 3D),
`pf` the filtered field on the mesh (`nothing` in the legacy foundry mode, which has
no filtered mesh field) and `pt` the projected physical density on the mesh.

`filter_cache` holds the 3D Helmholtz filter factor (unused in foundry mode). The
objective and gradient use the problem's own cache, the default; an export passes a
cache of its own so that it leaves the problem's cache as it found it.

`write_grid` (foundry mode): with `true`, the default used by the objective and gradient,
the values to interpolate are written (projected values in legacy mode) into the shared `sim.grid.params` before they are
interpolated (`build_foundry_pf`); with `false`, as in an export, they are interpolated
from a copy and `sim.grid` is not written (`foundry_pf_local`). Both give the same field.

`copy_filtered` (3D mode): with `true`, as in an export, the filter solve's result is copied
before `pf` and `pt` are built on it, so they stay valid after the filter factor is released,
whatever the backend does with its solution storage then. The default `false` keeps the
solver's vector, as the objective and gradient always have.
"""
function physical_density(p::Vector{Float64}, prob::OptimizationProblem;
    filter_cache::SolverCache=default_pool(prob.pool).filter_cache, write_grid::Bool=true,
    copy_filtered::Bool=false)
    (; sim, control) = prob
    sim0 = default_sim(sim)
    if prob.foundry_mode
        # Filter on grid
        pf_vec = filter_grid(p, sim0, control)
        to_mesh = write_grid ? build_foundry_pf : foundry_pf_local
        if control.foundry_projection_mode == :legacy
            # Legacy foundry flow: apply SSP on 2D grid first, then interpolate pt to FEM.
            pt_vec = smoothed_projection_vec(pf_vec, control, sim0)
            pt = to_mesh(pt_vec, sim0, control)
            return (; pf_vec, pf=nothing, pt)
        end
        # Current flow: interpolate pf to FEM first, then apply FEM SSP.
        pf = to_mesh(pf_vec, sim0, control)
        pt = project_ssp(pf, control, sim0.Ω)
        return (; pf_vec, pf, pt)
    end
    # Helmholtz PDE filter
    pf_vec = filter_helmholtz!(p, filter_cache, sim0, control)
    copy_filtered && (pf_vec = copy(pf_vec))
    pf = FEFunction(sim0.Pf, pf_vec)
    # SSP projection
    pt = project_ssp(pf, control, sim0.Ω)
    return (; pf_vec, pf, pt)
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

    # Filter on grid, then project (see `physical_density`)
    (; pf_vec, pf, pt) = physical_density(p, prob)

    local ∂g_∂pf_vec

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
            log_warn(:objective, "NaNs detected in PDE sensitivity")
        end

        # Explicit sensitivity (dg/dpf, not through the PDE)
        ∂g_∂pf_explicit = explicit_sensitivity(objective, pde, fields, pf, pt, sim, control; space=sim0.Pf)
        if any(isnan, ∂g_∂pf_explicit)
            log_warn(:objective, "NaNs detected in explicit sensitivity")
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

    # Helmholtz PDE filter, then SSP projection (see `physical_density`)
    (; pf, pt) = physical_density(p, prob)

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
