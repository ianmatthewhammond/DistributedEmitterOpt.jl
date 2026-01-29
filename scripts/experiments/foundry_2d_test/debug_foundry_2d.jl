using Pkg
Pkg.activate("/Users/ianhammond/GitHub/DistributedEmitterOpt.jl")

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random
using Test
using PyCall

# ═══════════════════════════════════════════════════════════════════════════════
# Utility: Test Gradient
# ═══════════════════════════════════════════════════════════════════════════════

function test_gradient(prob, p0; δ=1e-8, verbose=true)
    np = length(p0)
    grad = zeros(np)
    δp = randn(np) * δ

    # Forward + Adjoint
    g0 = objective_and_gradient!(grad, p0, prob)

    if any(isnan, grad)
        println("  ⚠️ Gradient contains NaNs!")
        println("  Norm: $(norm(grad))")
        println("  Count NaN: $(count(isnan, grad))")
    else
        println("  Gradient Norm: $(norm(grad))")
    end

    # Finite Difference
    grad_dummy = Float64[]
    g1 = objective_and_gradient!(grad_dummy, p0 + δp, prob)

    fd = g1 - g0
    adj = dot(grad, δp)
    rel_error = abs(fd - adj) / (abs(fd) + 1e-12)

    if verbose
        println("  g0 = $g0")
        println("  g1 = $g1")
        println("  FD  = $fd")
        println("  Adj = $adj")
        println("  Relative error = $(round(rel_error * 100, digits=4))%")
    end
    return rel_error
end

# ═══════════════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════════════

function _first_vtk_path(vtk_result)
    if vtk_result isa Tuple
        return _first_vtk_path(first(vtk_result))
    elseif vtk_result isa AbstractVector
        return first(vtk_result)
    else
        return vtk_result
    end
end

function save_foundry_grid_image(prob; outdir::String=@__DIR__, filename::String="foundry_grid.png")
    sim = prob.sim
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    grid_vals = reshape(prob.p, (nx, ny))

    plt = pyimport("matplotlib.pyplot")
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_vals', origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="p")
    plt.title("Foundry grid")
    png_path = joinpath(outdir, filename)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path
end

function write_projected_geometry_vtk(prob; outdir::String=@__DIR__)
    sim = prob.sim
    control = prob.control
    nx, ny = length(sim.grid.x), length(sim.grid.y)

    pf_vec = filter_grid(prob.p, sim, control)
    pt_vec = project_grid(pf_vec, sim, control)
    sim.grid.params[:, :] .= reshape(pt_vec, (nx, ny))

    pfield = interpolate(r -> pf_grid(r, sim.grid), sim.Pf)
    vtk_result = writevtk(
        sim.Ω_design,
        joinpath(outdir, "projected_geometry");
        cellfields=["p" => pfield]
    )
    return _first_vtk_path(vtk_result)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Setup: 2D Foundry Mode (Simple Periodic)
# ═══════════════════════════════════════════════════════════════════════════════

function build_foundry_experiment()
    # 1. Geometry & Mesh
    # Use standard symmetric geometry but fully periodic
    geo = SymmetricGeometry()
    # Scaled down for quicker test
    geo.L = 200.0
    geo.W = 200.0
    geo.l1 = 40.0
    geo.l2 = 20.0
    geo.l3 = 40.0

    meshfile = joinpath(@__DIR__, "mesh_foundry.msh")

    # CRITICAL FOR FOUNDRY: Full Periodicity (x and y)
    # Foundry mode typically projects a 2D pattern onto a periodic unit cell
    genperiodic(geo, meshfile; per_x=true, per_y=true)

    # 2. Simulation
    # foundry_mode=true: Enables 2D DOF grid
    # dir_x=false, dir_y=false: Full periodicity (no PEC walls)
    sim = build_simulation(meshfile;
        foundry_mode=true,
        dir_x=false,
        dir_y=false
    )

    # 3. Physics / Environment
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)

    # Simple Elastic Scattering
    λ = 532.0
    inputs = [FieldConfig(λ; θ=0.0, pol=:y)] # Pump Y
    outputs = FieldConfig[] # Elastic (same as input)

    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    # 4. Objective (Volume Integral)
    # Simple isotropic polarizability
    αₚ = Matrix{ComplexF64}(I, 3, 3)

    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=false
    )

    # 5. Control
    # Foundry mode typically uses DCT for parameterization, but we can test pixel-basis too.
    # We'll enable DCT as it's common for foundry.
    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0), # Filter radius
        use_dct=true,                # Enable DCT Parameterization
        use_projection=true,
        β=8.0,                       # Smooth gradients
        η=0.5,
        use_ssp=true,
        flag_volume=true
    )

    # 6. Solver
    solver = UmfpackSolver()

    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=true,
        control=control
    )

    return prob, meshfile
end

# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

println("\n=== Building 2D Foundry Experiment ===")
prob, mfile = build_foundry_experiment()

println("Problem built.")
println("  DOFs: $(length(prob.p)) (2D Grid)")
println("  Mesh: $mfile")

println("\n=== Testing Gradient ===")
Random.seed!(42)

# Initialize foundry grid with scaled x^2 + y^2 (centered, [0,1])
begin
    sim = prob.sim
    nx, ny = length(sim.grid.x), length(sim.grid.y)
    cx = (sim.grid.x[1] + sim.grid.x[end]) / 2
    cy = (sim.grid.y[1] + sim.grid.y[end]) / 2
    dx = maximum(abs.(sim.grid.x .- cx))
    dy = maximum(abs.(sim.grid.y .- cy))
    max_r2 = max(dx^2 + dy^2, eps())
    vals = [((x - cx)^2 + (y - cy)^2) / max_r2 for x in sim.grid.x, y in sim.grid.y]
    prob.p .= vec(vals)
end

# Check Jacobian
if any(isnan, prob.sim.grid.jacobian)
    println("⚠️ Jacobian contains NaNs!")
end

# Run test with larger perturbation to see if FD moves
test_gradient(prob, prob.p; δ=1e-8)

# Save outputs
grid_png = save_foundry_grid_image(prob; outdir=@__DIR__)
proj_vtu = write_projected_geometry_vtk(prob; outdir=@__DIR__)
println("Saved foundry grid image: $grid_png")
println("Saved projected geometry VTK: $proj_vtu")

# Cleanup
# rm(mfile; force=true) # Uncomment to keep mesh for inspection
