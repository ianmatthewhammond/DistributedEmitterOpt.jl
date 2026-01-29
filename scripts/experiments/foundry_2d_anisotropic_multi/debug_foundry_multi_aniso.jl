using Pkg
Pkg.activate("/Users/ianhammond/GitHub/DistributedEmitterOpt.jl")

using Revise
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
    plt.title("Foundry grid (p)")
    png_path = joinpath(outdir, filename)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    # Also save pf image
    control = prob.control
    pf_vec = filter_grid(prob.p, sim, control)
    pf_vals = reshape(pf_vec, (nx, ny))

    plt.figure(figsize=(6, 5))
    plt.imshow(pf_vals', origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="pf")
    plt.title("Filtered grid (pf)")
    png_path_pf = joinpath(outdir, "filtered_" * filename)
    plt.tight_layout()
    plt.savefig(png_path_pf, dpi=200)
    plt.close()

    return png_path
end

function write_projected_geometry_vtk(prob; outdir::String=@__DIR__)
    sim = prob.sim
    control = prob.control
    nx, ny = length(sim.grid.x), length(sim.grid.y)

    # 1. Filtered field (pf)
    pf_vec = filter_grid(prob.p, sim, control)
    sim.grid.params[:, :] .= reshape(pf_vec, (nx, ny))
    pf_field = interpolate(r -> pf_grid(r, sim.grid), sim.Pf)

    # 2. Projected field (pt)
    pt_vec = project_grid(pf_vec, sim, control)
    sim.grid.params[:, :] .= reshape(pt_vec, (nx, ny))
    pt_field = interpolate(r -> pf_grid(r, sim.grid), sim.Pf)

    vtk_result = writevtk(
        sim.Ω_design,
        joinpath(outdir, "projected_geometry");
        cellfields=["pt" => pt_field, "pf" => pf_field]
    )
    return _first_vtk_path(vtk_result)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Setup: 2D Foundry + Multi-Output + Anisotropic
# ═══════════════════════════════════════════════════════════════════════════════

function build_foundry_multi_aniso_experiment()
    # 1. Geometry & Mesh
    # Use the copied periodic mesh
    meshfile = joinpath(@__DIR__, "mesh_foundry.msh")

    # Ensure periodicity (x and y)
    # Note: genperiodic usually modifies the msh file in place or generates a new one. 
    # Since we copied it, we can run it here to be safe, or assume it's already good.
    # Let's run it to ensure the periodic tags are set up correctly.
    geo = SymmetricGeometry()
    geo.L = 200.0
    geo.W = 200.0
    geo.l1 = 40.0
    geo.l2 = 20.0
    geo.l3 = 40.0
    genperiodic(geo, meshfile; per_x=true, per_y=true)

    # 2. Simulation
    # foundry_mode=true: Enables 2D DOF grid
    # dir_x=false, dir_y=false: Full periodicity (no PEC walls)
    sim = build_simulation(meshfile;
        foundry_mode=true,
        dir_x=false,
        dir_y=true
    )

    # 3. Physics / Environment
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)

    # Complex Configuration
    # Pump: 532nm, Pol Y
    pump_config = FieldConfig(532.0; θ=0.0, pol=:y)

    # Emission Outputs: Mixed Wavelengths and Polarizations
    outputs = [
        FieldConfig(540.0; θ=0.0, pol=:x, weight=1.0),
        FieldConfig(532.0; θ=0.0, pol=:y, weight=0.5)
    ]

    pde = MaxwellProblem(env=env, inputs=[pump_config], outputs=outputs)

    # 4. Objective
    # Anisotropic Polarizability: x-only
    αₚ = zeros(ComplexF64, 3, 3)
    αₚ[1, 1] = 1.0 + 0.0im # [1 0 0; 0 0 0; 0 0 0]

    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=false
    )

    # 5. Control
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

println("\n=== Building Foundry Multi-Output Anisotropic (X-only) Experiment ===")
prob, mfile = build_foundry_multi_aniso_experiment()

println("Problem built.")
println("  DOFs: $(length(prob.p)) (2D Grid)")
println("  Mesh: $mfile")
println("  Outputs: $(length(prob.pde.outputs)) (540nm X, 550nm Y)")
println("  Polarizability: X-only diagonal")

println("\n=== Testing Gradient ===")
Random.seed!(42)

# Initialize foundry grid with something minimal but non-uniform
prob.p .= 0.3 .+ 0.4 .* rand(length(prob.p))

# Check Jacobian
if any(isnan, prob.sim.grid.jacobian)
    println("⚠️ Jacobian contains NaNs!")
end

# Run gradient test
rel_err = test_gradient(prob, prob.p; δ=1e-8)

println("\nTest Result:")
if rel_err < 1e-4
    println("✅ PASS (Rel Error < 1e-4)")
else
    println("❌ FAIL (Rel Error too high)")
end

# Save outputs
grid_png = save_foundry_grid_image(prob; outdir=@__DIR__)
proj_vtu = write_projected_geometry_vtk(prob; outdir=@__DIR__)
println("Saved foundry grid image: $grid_png")
println("Saved projected geometry VTK: $proj_vtu")
