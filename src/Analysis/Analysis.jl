module Analysis

using JLD2
using Printf
using CairoMakie
import ..DistributedEmitterOpt: OptimizationProblem, FieldConfig, MaxwellProblem, evaluate, default_sim, sim_for, pool_for, get_cache!, build_phys_params, map_pt, assemble_maxwell, maxwell_lu!, assemble_source, maxwell_solve!, has_maxwell_factor, glc_solid, filter_grid, project_grid, pf_grid, project_ssp, build_foundry_pf, smoothed_projection_vec, physical_density, solve_forward_config!, default_pool, SolverCache, release_factor!, clear_filter_factor!, returns_owned_solutions, require_dct_filter_problem, validate_field_boundaries
using Gridap
using LinearAlgebra
using Statistics

export spectral_sweep, fabrication_sweep, visualize_results, design_and_fields, plot_iteration_history, save_design_2d_image

sumabs4_uh2(uh²) = uh² * uh²
sumabs4(uh::VectorValue) = sumabs2(uh)^2
sumabs3(uh::VectorValue) = sqrt(sumabs2(uh))^3
sumabs2(uh::VectorValue) = sum(abs.(Tuple(uh)) .^ 2)
sumabs(uh::VectorValue) = sqrt(sumabs2(uh))
sumabs4(a::CellField) = Operation(sumabs4)(a)
sumabs3(a::CellField) = Operation(sumabs3)(a)
sumabs2(a::CellField) = Operation(sumabs2)(a)
sumabs(a::CellField) = Operation(sumabs)(a)

"""
    spectral_sweep(prob, p_opt; center_λ, range_λ, points, root)

The sweep evaluates `p_opt` at each wavelength on `prob`. On return, normal or by an error, `prob.g` and
`prob.∇g` hold the values they had before the call.
At zero points, only single-pool sweeps previously returned empty results.

Sweep wavelength around a center value, re-evaluating the objective.
Does not perform optimization, only analysis.

A foundry problem whose control has `use_dct=false` and `use_filter` on is refused with an `ArgumentError` before
`prob.pde` is replaced or any file is written. With `points=0` nothing is evaluated and the call returns empty
results as before, without the check.
"""
function spectral_sweep(prob::OptimizationProblem, p_opt::Vector{Float64};
    center_λ::Float64=532.0,
    range_λ::Float64=50.0,
    points::Int=40,
    root::String=prob.root,
    save_plot::Bool=true
)
    # Each evaluation of the sweep overwrites the caller's score and gradient: save them and restore them on
    # return, normal or by an error.
    original_g = prob.g
    original_∇g = copy(prob.∇g)
    try
        return _spectral_sweep_evaluations(prob, p_opt; center_λ, range_λ, points, root, save_plot)
    finally
        prob.g = original_g
        prob.∇g .= original_∇g
    end
end

# The body of `spectral_sweep`, which saves and restores `prob.g` and `prob.∇g` around it.
function _spectral_sweep_evaluations(prob::OptimizationProblem, p_opt::Vector{Float64};
    center_λ::Float64,
    range_λ::Float64,
    points::Int,
    root::String,
    save_plot::Bool
)
    # Generate wavelength list (log-spaced away from center to capture resonance)
    # Similar to Setup.jl logic
    e = range(-1, 1; length=points)
    offsets = sign.(e) .* range_λ .* ((10 .^ abs.(e) .- 1) ./ (10 - 1))
    λ_s = sort(unique(center_λ .+ offsets))
    # Before prob.pde is replaced below: a 2D problem without the DCT-I filter cannot be evaluated. With no
    # wavelength nothing is evaluated.
    isempty(λ_s) || require_dct_filter_problem(prob, "spectral_sweep"; use_constraints=false)

    # Store results
    gys = zeros(length(λ_s))

    # Original config to restore later
    original_inputs = prob.pde.inputs
    original_outputs = prob.pde.outputs

    println("Starting spectral sweep over $(length(λ_s)) points...")

    for (i, λ) in enumerate(λ_s)
        # Update PDE configuration for new wavelength
        # We assume elastic scattering for simplicity unless inputs/outputs differ
        # TODO: Handle inelastic case more generally if needed

        # Create new inputs with updated wavelength
        new_inputs = [FieldConfig(λ; tilt_x=inp.tilt_x, tilt_y=inp.θ, pol=inp.polarization,
            weight=inp.weight, n_incident=inp.n_incident, formulation=inp.formulation) for inp in original_inputs]

        # For outputs, if they are empty (elastic reuse), keep empty. 
        # If they exist (inelastic), we might need to shift them too? 
        # For now, let's assume we are sweeping the global scale or just the input.
        # If elastic, outputs are empty.
        new_outputs = FieldConfig[]
        if !isempty(original_outputs)
            # If inelastic, we typically sweep pump, keeping emission fixed? 
            # Or sweep both? Let's assume pump sweep for now.
            new_outputs = original_outputs
        end

        # Update environment logic if explicit material check is needed?
        # The MaxwellSolver typically calls complex_index(mat_design, λ) internally
        # so we just need to pass the new wavelength in the FieldConfig.

        # Replace PDE (lightweight copy)
        prob.pde = MaxwellProblem(
            env=prob.pde.env,
            inputs=new_inputs,
            outputs=new_outputs,
            α_loss=prob.pde.α_loss
        )

        # Clear solver cache because wavelength changed
        empty!(prob.pool)

        # Evaluate
        # We use evaluate() which computes gradients too, but we only need g.
        # To save time we could have a `compute_objective` wrapper, but `evaluate` is fine.
        val, _ = evaluate(prob, p_opt)

        gys[i] = val
        @printf "  λ = %.2f nm, g = %.4e\n" λ val
    end

    # Restore problem state
    prob.pde = MaxwellProblem(
        env=prob.pde.env,
        inputs=original_inputs,
        outputs=original_outputs,
        α_loss=prob.pde.α_loss
    )
    empty!(prob.pool)

    # Save data
    JLD2.save(joinpath(root, "spectral.jld2"), Dict("lambda" => λ_s, "g" => gys))

    # Plot
    if save_plot
        fig = Figure(resolution=(600, 500))
        ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="Objective", yscale=log10, title="Spectral Response")
        lines!(ax, λ_s, gys, label="Objective", linewidth=2)
        vlines!(ax, [center_λ], color=:red, linestyle=:dash, label="Center")
        axislegend(ax)
        save(joinpath(root, "spectral_response.png"), fig)
    end

    return λ_s, gys
end

"""
    fabrication_sweep(prob, p_opt; center_R, range_R, points, root)

Sweep filter radius (or threshold) to simulate fabrication errors.

The sweep evaluates `p_opt` at each radius on `prob`. On return, normal or by an error,
`prob.control.R_filter`, `prob.g` and `prob.∇g` hold the values they had before the call.

A foundry problem whose control has `use_dct=false` and `use_filter` on is refused with an `ArgumentError` before
`prob.control.R_filter` is changed or any file is written. With `points=0` nothing is evaluated and the call returns
empty results as before, without the check.
"""
function fabrication_sweep(prob::OptimizationProblem, p_opt::Vector{Float64};
    center_R::Float64=20.0,
    range_R::Float64=10.0, # +/- range
    points::Int=15,
    root::String=prob.root,
    save_plot::Bool=true
)
    # Sweep filter radius R_filter (assuming isotropic x/y/z)
    Rs = range(center_R - range_R, center_R + range_R, length=points)
    # Before prob.control.R_filter is changed below. With no radius nothing is evaluated.
    isempty(Rs) || require_dct_filter_problem(prob, "fabrication_sweep"; use_constraints=false)

    gys = zeros(length(Rs))

    # Save the caller's radius, score and gradient (evaluate overwrites the last two)
    original_R = prob.control.R_filter
    original_g = prob.g
    original_∇g = copy(prob.∇g)

    println("Starting fabrication sweep (R_filter) over $(length(Rs)) points...")

    try
        for (i, R) in enumerate(Rs)
            # Update control
            # We assume R_filter stored as tuple
            prob.control.R_filter = (R, R, R)

            # Evaluate
            val, _ = evaluate(prob, p_opt)

            gys[i] = val
            @printf "  R = %.2f nm, g = %.4e\n" R val
        end
    finally
        # Restore on both exits
        prob.control.R_filter = original_R
        prob.g = original_g
        prob.∇g .= original_∇g
    end

    # Save
    JLD2.save(joinpath(root, "fabrication.jld2"), Dict("R" => Rs, "g" => gys))

    # Plot
    if save_plot
        fig = Figure(resolution=(600, 500))
        ax = Axis(fig[1, 1], xlabel="Filter Radius (nm)", ylabel="Objective", yscale=log10, title="Fabrication Tolerance")
        lines!(ax, Rs, gys, label="Objective", linewidth=2)
        vlines!(ax, [center_R], color=:red, linestyle=:dash, label="Nominal")
        axislegend(ax)
        save(joinpath(root, "fabrication_tolerance.png"), fig)
    end

    return Rs, gys
end

"""
    design_and_fields(prob, p_opt; config=prob.pde.inputs[1]) -> (; pf_vec, pf, pt, config, E)

Physical density and forward field for the design `p_opt`: `physical_density` (filter, then
projection), then a forward solve that assembles and factorizes Maxwell for this design
(`solve_forward_config!`). `pf_vec` is the filtered vector, `pf` the filtered field on the
mesh (`nothing` in the legacy foundry projection mode), `pt` the physical density on the mesh
and `E` the field of `config`.

The call reads the Maxwell solver of `config`'s pool and the default pool's solver once, at
entry, and uses only those objects. It refuses:
- with `ArgumentError`, a Maxwell solver, or (3D with `use_filter` on) a filter solver, that
  declares `returns_owned_solutions` `false` (see `AbstractSolver`);
- with `ArgumentError`, a foundry problem whose control has `use_filter` on and `use_dct=false`
  (`require_dct_filter_problem`);
- with `DimensionMismatch`, a design of the wrong length;
- the control and `config` flag combinations of `check_export_config` (projection mode,
  legacy sidewalls, mirror walls, Bloch with PML).
Filter solver errors can occur while building the density. Maxwell assembly and solver errors
occur afterwards.

The filter and Maxwell factors are made in solver caches owned by the call and handed to
`release_factor!`; each solver result is copied before its factor is released. The call does
not use or fill the caches of `prob`'s pools and, in foundry mode, does not write
`sim.grid.params`. For solution vectors a caller holds from earlier solves, see
`AbstractSolver` and issue #41. Concurrent use of one `prob` is not supported (#58).
"""
function design_and_fields(prob::OptimizationProblem, p_opt::Vector{Float64};
    config::FieldConfig=prob.pde.inputs[1])
    # The solver objects this call uses are read once, here; the checks and the call's own
    # caches use these objects, whatever the pools hold later.
    solvers = export_solvers(prob, config)
    # Checks that read only their arguments come first: the backend declaration, the foundry
    # filter setting, the design length, then the control and configuration flags.
    check_export_backend(prob, solvers, "design_and_fields")
    require_dct_filter_problem(prob, "design_and_fields"; use_constraints=false)
    check_export_design_length(prob, p_opt, "design_and_fields")
    check_export_config(prob, config)
    return export_design_and_fields(prob, p_opt, config, solvers)
end

"""
The work of `design_and_fields` after its checks, on the solver objects `solvers` (from
`export_solvers`) read by the caller at entry.
"""
function export_design_and_fields(prob::OptimizationProblem, p_opt::Vector{Float64}, config::FieldConfig, solvers)
    # Can load a material table into the material cache and log; refused before any density.
    phys = build_phys_params(config, prob.pde.env, sim_for(prob.sim, config); α=prob.pde.α_loss)

    (; pf_vec, pf, pt) = export_density_with(p_opt, prob, solvers.filter)
    maxwell_cache = analysis_cache(solvers.maxwell)
    try
        # The field is copied out of the solver's storage before its factor is released.
        E = solve_forward_config!(prob.pde, config, pt, prob.sim, maxwell_cache; phys, copy_field=true)
        return (; pf_vec, pf, pt, config, E)
    finally
        release_analysis_cache!(maxwell_cache)
    end
end

"""
The solver objects (and MPI head flags) an export of `config` uses, read once from the
problem's pools: `maxwell` from `config`'s pool, `filter` from the default pool. An export
builds its own caches on these objects, so a pool or solver replaced during
the call (for example by a logger reacting to a warning) is not used by it.
"""
export_solvers(prob::OptimizationProblem, config::FieldConfig) =
    (; maxwell=solver_of(pool_for(prob.pool, config)), filter=solver_of(default_pool(prob.pool)))

solver_of(pool) = (; solver=pool.solver, am_head=pool.am_head)

"""
Check the backend declarations required by the settings at entry.
"""
function check_export_backend(prob::OptimizationProblem, solvers, caller::String)
    needs_filter_solve(prob) && check_owned_solutions(solvers.filter.solver, caller)
    check_owned_solutions(solvers.maxwell.solver, caller)
    return nothing
end

"""
Check foundry projection mode and legacy sidewalls, mirror compatibility, and Bloch compatibility with PML. They are
the refusals the density and the operator would raise later, with the same messages: in
foundry mode an unsupported `foundry_projection_mode` (`ErrorException`), and
`foundry_projection_mode=:legacy` with `use_sidewall` (`ArgumentError`); for every mode the
mirror and polarization checks of `validate_field_boundaries` and, for the Bloch
formulation, an environment with the ±z PML (`ArgumentError`).
"""
function check_export_config(prob::OptimizationProblem, config::FieldConfig)
    control = prob.control
    if prob.foundry_mode
        mode = control.foundry_projection_mode
        mode in (:current, :legacy) ||
            error("Unsupported foundry_projection_mode=$(mode). Use :current or :legacy.")
        mode == :legacy && control.use_sidewall && throw(ArgumentError(
            "sidewalls need foundry_projection_mode=:current; :legacy projects on the 2D grid, " *
            "which has no sidewall model"))
    end
    validate_field_boundaries(sim_for(prob.sim, config), config)
    config.formulation == :bloch && !isnothing(prob.pde.env.pml) && throw(ArgumentError(
        "assemble_maxwell: the ±z PML is not implemented for the Bloch (general-angle) formulation; use " *
        "Environment(pml=nothing) with tilted or explicitly-tilted FieldConfigs"))
    return nothing
end

"True when the physical density of `prob` needs a 3D Helmholtz filter solve (3D mode with the filter on)."
needs_filter_solve(prob::OptimizationProblem) = !prob.foundry_mode && prob.control.use_filter

function check_owned_solutions(solver, caller::String)
    returns_owned_solutions(solver) || throw(ArgumentError(
        "$caller: the solver backend $(typeof(solver)) declares `returns_owned_solutions` false; an export " *
        "needs a backend that creates, refactorizes, solves with and releases a factor without writing a " *
        "vector another factor's solve returned (see `AbstractSolver`). Nothing was solved, factorized or written."))
    return nothing
end

"Refuse a design vector whose length differs from the problem's design (it reads only its arguments)."
function check_export_design_length(prob::OptimizationProblem, p_opt::Vector{Float64}, caller::String)
    length(p_opt) == length(prob.p) || throw(DimensionMismatch(
        "$caller: the design vector has $(length(p_opt)) entries; this problem's design has $(length(prob.p))"))
    return nothing
end

"""
The physical density of `p` for an export: the 3D filter is factorized in a cache owned
by this call and released before it returns or throws, the filtered vector is copied out
of the solver's storage before that release, and the foundry grid is not written. `physical_density`, which the objective uses, does not
check the declaration.
"""
function export_density(p::Vector{Float64}, prob::OptimizationProblem)
    filter = solver_of(default_pool(prob.pool))
    needs_filter_solve(prob) && check_owned_solutions(filter.solver, "export_density")
    require_dct_filter_problem(prob, "export_density"; use_constraints=false)
    check_export_design_length(prob, p, "export_density")
    return export_density_with(p, prob, filter)
end

"The work of `export_density` after its checks, on the filter solver `filter` (from `solver_of`)."
function export_density_with(p::Vector{Float64}, prob::OptimizationProblem, filter)
    filter_cache = analysis_cache(filter)
    try
        return physical_density(p, prob; filter_cache, write_grid=false, copy_filtered=true)
    finally
        release_analysis_cache!(filter_cache)
    end
end

"An empty solver cache on a captured solver (`solver_of`), owned by one export."
analysis_cache(s) = SolverCache(s.solver; am_head=s.am_head)

"""
Release every factor held by an export's own solver cache. The filter factor goes through
`clear_filter_factor!`, which detaches the factor and its operator key before the release,
so the cache holds neither afterwards, also when the release throws.
"""
function release_analysis_cache!(cache::SolverCache)
    clear_filter_factor!(cache)
    for name in (:A_factor, :E2_factor)
        factor = getfield(cache, name)
        isnothing(factor) && continue
        setfield!(cache, name, nothing)
        release_factor!(cache.solver, factor)
    end
    return nothing
end

"""
    visualize_results(prob, p_opt; root)

Write the physical density and first input’s field for `p_opt` to `design.vtu` and `fields.vtu` on the same path
as `design_and_fields`. Foundry mode also writes `design_2d.png`.

It refuses what `design_and_fields` refuses for the first input configuration.
"""
function visualize_results(prob::OptimizationProblem, p_opt::Vector{Float64}; root::String=prob.root)
    # Before the announcement below and before any file is written, in the order of
    # `design_and_fields`, on solver objects read once here.
    config = prob.pde.inputs[1]
    solvers = export_solvers(prob, config)
    check_export_backend(prob, solvers, "visualize_results")
    require_dct_filter_problem(prob, "visualize_results"; use_constraints=false)
    check_export_design_length(prob, p_opt, "visualize_results")
    check_export_config(prob, config)
    println("Generating visualization artifacts in $root...")

    sim0 = default_sim(prob.sim)
    (; pf_vec, pt, E) = export_design_and_fields(prob, p_opt, config, solvers)

    # 1. Write the physical density (a plain function of position in the legacy foundry mode)
    pt_cf = pt isa Function ? CellField(pt, sim0.Ω_design) : pt
    writevtk(sim0.Ω_design, joinpath(root, "design"), cellfields=["p" => pt_cf])

    # 2. Write the fields of the first input configuration (nominal)
    Ω = sim_for(prob.sim, config).Ω

    writevtk(Ω, joinpath(root, "fields"), cellfields=[
        "E_real" => real(E),
        "E_imag" => imag(E),
        "E_sq" => sumabs2(E)
    ])

    println("  Saved design.vtu and fields.vtu")

    # 3. Save 2D copy if foundry mode (the filtered grid)
    if prob.foundry_mode
        save_design_2d_image(pf_vec, sim0.grid.x, sim0.grid.y; root=root)
    end
end

"""
    plot_iteration_history(g_history; root)

Plots the objective value vs iterations.
"""
function plot_iteration_history(g_history::Vector{Float64}; root::String)
    println("Plotting iteration history in $root...")
    fig = Figure(resolution=(800, 600), fontsize=20)
    ax = Axis(fig[1, 1], yscale=log10, title="Objective History", xlabel="Iteration", ylabel="g")

    # Plot history
    lines!(ax, 1:length(g_history), g_history, label="Objective", linewidth=2, color=:blue)

    axislegend(ax)
    save(joinpath(root, "history.png"), fig)
end

"""
    save_design_2d_image(p_vec, xs, ys; root)

Plots the 2D design parameters as a heatmap.
"""
function save_design_2d_image(p_vec::Vector{Float64}, xs::Vector{Float64}, ys::Vector{Float64}; root::String)
    println("Plotting 2D design image in $root...")
    nx = length(xs)
    ny = length(ys)

    # Reshape p_vec (assuming matching grid ordering)
    p_mat = reshape(p_vec, nx, ny)

    fig = Figure(resolution=(800, 600), fontsize=20)
    ax = Axis(fig[1, 1], title="Design Parameters (Foundry)", xlabel="x (nm)", ylabel="y (nm)", aspect=DataAspect())

    heatmap!(ax, xs, ys, p_mat, colormap=:grays, colorrange=(0, 1))
    Colorbar(fig[1, 2], limits=(0, 1), colormap=:grays, label="Material Density")

    save(joinpath(root, "design_2d.png"), fig)
end

end # module
