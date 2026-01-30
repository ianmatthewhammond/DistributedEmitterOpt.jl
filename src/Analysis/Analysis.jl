module Analysis

using JLD2
using Printf
using CairoMakie
import ..DistributedEmitterOpt: OptimizationProblem, FieldConfig, MaxwellProblem, evaluate, default_sim, sim_for, pool_for, get_cache!, build_phys_params, map_pt, assemble_maxwell, maxwell_lu!, assemble_source, maxwell_solve!, has_maxwell_factor, glc_solid, filter_grid, project_grid, pf_grid, project_ssp
using Gridap
using LinearAlgebra
using Statistics

export spectral_sweep, fabrication_sweep, visualize_results, plot_iteration_history, save_design_2d_image

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

Sweep wavelength around a center value, re-evaluating the objective.
Does not perform optimization, only analysis.
"""
function spectral_sweep(prob::OptimizationProblem, p_opt::Vector{Float64};
    center_λ::Float64=532.0,
    range_λ::Float64=50.0,
    points::Int=40,
    root::String=prob.root,
    save_plot::Bool=true
)
    # Generate wavelength list (log-spaced away from center to capture resonance)
    # Similar to Setup.jl logic
    e = range(-1, 1; length=points)
    offsets = sign.(e) .* range_λ .* ((10 .^ abs.(e) .- 1) ./ (10 - 1))
    λ_s = sort(unique(center_λ .+ offsets))

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
        new_inputs = [FieldConfig(λ, θ=inp.θ, pol=inp.polarization) for inp in original_inputs]

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

    gys = zeros(length(Rs))

    # Save original control
    original_R = prob.control.R_filter

    println("Starting fabrication sweep (R_filter) over $(length(Rs)) points...")

    for (i, R) in enumerate(Rs)
        # Update control
        # We assume R_filter stored as tuple
        prob.control.R_filter = (R, R, R)

        # Evaluate
        val, _ = evaluate(prob, p_opt)

        gys[i] = val
        @printf "  R = %.2f nm, g = %.4e\n" R val
    end

    # Restore
    prob.control.R_filter = original_R

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
    visualize_results(prob, p_opt; root)

Generate VTK and PNG outputs for the final design.
"""
function visualize_results(prob::OptimizationProblem, p_opt::Vector{Float64}; root::String=prob.root)
    println("Generating visualization artifacts in $root...")

    # 1. Get physical design pt on the mesh
    # Valid for both visualization and solving
    sim = prob.sim
    sim0 = default_sim(sim)

    local pt
    if prob.foundry_mode
        # Filter on grid
        pf_vec = filter_grid(p_opt, sim0, prob.control)

        # Interpolate to mesh (Piecewise Linear Pf)
        nx, ny = length(sim0.grid.x), length(sim0.grid.y)
        p_grid_mat = reshape(pf_vec, nx, ny)
        pf_vals = [pf_grid(node, p_grid_mat, sim0.grid.x, sim0.grid.y) for node in sim0.grid.nodes]
        pf = FEFunction(sim0.Pf, pf_vals)

        # SSP projection on mesh
        pt = project_ssp(pf, prob.control)
    else
        # 3D mode (placeholder implementation)
        pf_vec = p_opt # Assumes no filter for simplicity if not foundry
        pf = FEFunction(sim0.Pf, pf_vec)
        pt = project_ssp(pf, prob.control)
    end

    # Write optimized design
    writevtk(sim0.Ω_design, joinpath(root, "design"), cellfields=["p" => pt])

    # 2. Solve fields for the first input config (nominal)
    config = prob.pde.inputs[1]

    # Get solver cache
    pool_fc = pool_for(prob.pool, config)
    cache = get_cache!(pool_fc, config)

    # Assemble physics
    # Correct signature: build_phys_params(fc::FieldConfig, env::Environment, sim; α=0.0)
    phys = build_phys_params(config, prob.pde.env, sim0; α=prob.pde.α_loss)

    # Manual forward solve for single config
    # 1. Factorize (if needed)
    if !has_maxwell_factor(cache)
        # map_pt handles expanding p_phys if needed (usually identity for sim_base)
        # Use pt directly (already on mesh)
        # Note: map_pt is not needed if we use the mesh-defined pt, unless specific logic applies
        # But solve_forward uses map_pt(pt, sim_fc).
        # Since we just computed pt on sim0, we should pass it.
        # If sim_fc != sim0, we might need re-mapping, but typically sim0 is the one with the mesh.

        A = assemble_maxwell(pt, sim_for(sim, config), phys)
        maxwell_lu!(cache, A)
    end

    # 2. Solve
    source_y = (config.polarization == :y)
    b = assemble_source(sim_for(sim, config), phys; source_y)

    E_vec = maxwell_solve!(cache, b)
    uh = FEFunction(sim0.U, E_vec)  # Assuming sim0 for visualization

    # 3. Write fields
    Ω = sim0.Ω

    writevtk(Ω, joinpath(root, "fields"), cellfields=[
        "E_real" => real(uh),
        "E_imag" => imag(uh),
        "E_sq" => sumabs2(uh)
    ])

    println("  Saved design.vtu and fields.vtu")

    # 4. Save 2D copy if foundry mode
    if prob.foundry_mode
        # pf_vec computed above
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
