"""
    Debug script: isotropic tensor, 3D DOFs, volume objective, elastic.

Run from REPL:
    julia> include("scripts/experiments/isotropic_3d_volume/debug_isotropic_3d_volume.jl")
"""

using Pkg; 
Pkg.activate(".")
Pkg.instantiate()
using Revise

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random
using PyCall

# Dir of file
const OUTDIR = dirname(@__FILE__)
const PERTURBATION = 1e-8
const TOL_RELATIVE = 1e-6

"""Build a minimal problem for the isotropic 3D DOF elastic volume test."""
function build_debug_objective(; outdir::String=OUTDIR)
    mkpath(outdir)
    # Coarse test geometry/mesh
    geo = SymmetricGeometry(532.0; L=100.0, W=100.0, hd=80.0, hsub=40.0)
    geo.l1 = 50.0
    geo.l2 = 30.0
    geo.l3 = 50.0
    geo.hair = 200.0
    geo.hs = 120.0
    geo.ht = 80.0

    meshfile = joinpath(outdir, "mesh.msh")
    genperiodic(geo, meshfile; per_x=true, per_y=true)

    # 3D DOF mode, normal incidence (default source_y=true)
    sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=true)

    # Isotropic polarizability tensor (identity)
    αₚ = Matrix{ComplexF64}(I, 3, 3)
    objective = SERSObjective(
        αₚ=αₚ,
        volume=true,
        surface=false,
        use_damage_model=false
    )

    env = Environment(mat_design="Ag", mat_fluid=1.33)
    inputs = [FieldConfig(532.0; θ=0.0, pol=:y)]
    outputs = FieldConfig[]  # Elastic scattering: outputs reuse inputs
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    control = Control(
        use_filter=true,
        R_filter=(15.0, 15.0, 15.0),
        use_dct=false,         # Helmholtz filter for 3D DOF mode
        use_projection=true,
        β=8.0,
        η=0.5,
        use_ssp=true,
        flag_volume=true,
        flag_surface=false,
        use_damage=false
    )

    solver = UmfpackSolver()
    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=false,
        control=control
    )
    Random.seed!(42)
    init_random!(prob)

    return prob
end

"""Run a finite-difference directional derivative check."""
function test_gradient(prob, p0; δ=PERTURBATION, verbose=true)
    np = length(p0)
    grad = zeros(np)
    δp = randn(np) * δ

    g0 = objective_and_gradient!(grad, p0, prob)
    g1 = objective_and_gradient!(Float64[], p0 + δp, prob)

    fd = g1 - g0
    adj = dot(grad, δp)
    rel_error = abs(fd - adj) / (abs(fd) + 1e-12)

    if verbose
        println("g0 = $g0")
        println("g1 = $g1")
        println("FD  = $fd")
        println("Adj = $adj")
        println("Relative error = $(round(rel_error * 100, digits=2))%")
    end

    return rel_error
end

"""Compute forward fields and projected design for visualization."""
function compute_forward_fields(prob, p)
    sim = prob.sim
    control = prob.control

    pf_vec = filter_helmholtz!(p, prob.pool.filter_cache, sim, control)
    pf = FEFunction(sim.Pf, pf_vec)
    pt = project_ssp(pf, control)

    fields = solve_forward!(prob.pde, pt, sim, prob.pool)
    Ep = first(values(fields))

    return Ep, pt
end

"""Write VTK files for fields and design; returns paths."""
function _first_vtk_path(vtk_result)
    if vtk_result isa Tuple
        return _first_vtk_path(first(vtk_result))
    elseif vtk_result isa AbstractVector
        return first(vtk_result)
    else
        return vtk_result
    end
end

function write_vtk_outputs(sim, Ep, pt; outdir::String=OUTDIR)
    E2 = DistributedEmitterOpt.sumabs2(Ep)

    fields_files = writevtk(
        sim.Ω,
        joinpath(outdir, "fields");
        cellfields=["E2" => E2]
    )

    design_files = writevtk(
        sim.Ω_design,
        joinpath(outdir, "design");
        cellfields=["p" => pt]
    )

    return _first_vtk_path(fields_files), _first_vtk_path(design_files)
end

"""Save a quick diagnostic image using pyvista (based on old repo Vis.jl)."""
function save_field_image(fields_vtu::String, design_vtu::String; outdir::String=OUTDIR)
    pv = pyimport("pyvista")
    try
        pv.start_xvfb()
    catch
        # Ignore if running with a display.
    end

    field_grid = pv.read(fields_vtu)
    design_grid = pv.read(design_vtu)

    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(1600, 800))
    plotter.subplot(0, 0)
    plotter.add_mesh(field_grid, scalars="E2", cmap="seismic", opacity=0.85)
    plotter.show_axes()

    plotter.subplot(0, 1)
    plotter.add_mesh(design_grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), opacity=0.85)
    plotter.show_axes()

    png_path = joinpath(outdir, "fields.png")
    plotter.show(screenshot=png_path)
    return png_path
end

println("\n=== Debug: isotropic, 3D DOF, volume objective, elastic ===")
println("Output directory: $(OUTDIR)")

prob = build_debug_objective()
p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
rel_err = test_gradient(prob, p0)
println("PASS = $(rel_err < TOL_RELATIVE)  (rel_err = $rel_err, tol = $TOL_RELATIVE)")

Ep, pt = compute_forward_fields(prob, p0)
fields_vtu, design_vtu = write_vtk_outputs(prob.sim, Ep, pt)
png_path = save_field_image(fields_vtu, design_vtu)

println("Saved mesh: $(joinpath(OUTDIR, "mesh.msh"))")
println("Saved VTK:  $fields_vtu")
println("Saved VTK:  $design_vtu")
println("Saved PNG:  $png_path")
