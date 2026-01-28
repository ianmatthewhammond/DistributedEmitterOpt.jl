"""
    Old-code sandbox 3D DOF test.

Run from REPL:
    julia> include("scripts/experiments/sandbox_3d_tests/old_sandbox_3d_test.jl")
"""

using Pkg

const OLD_ROOT = "/Users/ianhammond/GitHub/Emitter3DTopOpt"
const OUTDIR = @__DIR__
const PERTURBATION = 1e-8

Pkg.activate(OLD_ROOT)
using Revise

import Emitter3DTopOpt as e3
include(e3.includesolver("Umfpack"))
using .UmfpackSolver
include(e3.includescript("Setup"))
using .Setup
using LinearAlgebra
using PyCall

mkpath(OUTDIR)

const ALPHA_P = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)

function build_old_objective(; outdir::String=OUTDIR)
    reuse_y1 = UmfpackSolver.UMFPACK_Reuse()
    reuse_y2 = reuse_y1
    reuse_x = UmfpackSolver.UMFPACK_Reuse()

    obj = Setup.SetupAll(e3;
        root=outdir * "/",
        genmesh=true,
        meshfile="mesh.msh",

        # Physics (sandbox config)
        λ1=532.0,
        λ2=532.0,
        θ1=0.0,
        θ2=0.0,
        mat_m="Ag",
        mat_s="Ag",
        norder=0,
        qorder=4,
        α=0.0,
        αₚ=ALPHA_P,

        # Geometry (sandbox config)
        mesh_type="Box",
        L=150.0,
        W=150.0,
        hd=150.0,
        hsub=50.0,
        l1=30.0,
        l2=20.0,
        l3=30.0,
        full_cell=false,
        full_x=false,
        full_y=false,
        bidirectional=false,
        nonlocal=false,

        # Controls (sandbox config)
        flag_f=true,
        flag_t=true,
        flag_r=false,
        flag_c=false,
        flag_foundry=false, # 3D DOFs
        flagS=false,
        flagV=true,
        flag_nd=false,
        flag_e2=false,
        R_f=(20.0, 20.0, 20.0),
        R_er=0.0,
        R_nl=0.0,
        R_s=2.0,
        β=8.0,
        η=0.5,
        ηe=0.75,
        ηd=0.25,
        c0=-1.0,
        γ=1.0,
        Eₜₕ=10.0,
        subpixel=true,
        firststep=true,

        # Optimization (sandbox config)
        init=["rand"],
        β_list=[Inf],

        reuse_y1=reuse_y1,
        reuse_y2=reuse_y2,
        reuse_x=reuse_x
    )

    return obj
end

function test_gradient(obj, p0; δ=PERTURBATION, verbose=true)
    np = length(p0)
    grad = zeros(np)
    δp = randn(np) * δ

    g0 = e3.stepobjective(p0, grad; obj)
    g1 = e3.stepobjective(p0 + δp, Float64[]; obj)

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

"""Save VTK outputs using old code paths."""
function write_vtk_outputs(obj, p0; outdir::String=OUTDIR)
    e3.vtk_all(p0, obj; solve_flag=true)
    return joinpath(outdir, "fields.vtu"), joinpath(outdir, "design.vtu")
end

"""Save a quick diagnostic image using pyvista."""
function save_field_image(fields_vtu::String, design_vtu::String; outdir::String=OUTDIR)
    pv = pyimport("pyvista")
    try
        pv.start_xvfb()
    catch
    end

    field_grid = pv.read(fields_vtu)
    design_grid = pv.read(design_vtu)

    plotter = pv.Plotter(shape=(1, 2), off_screen=true, window_size=(1600, 800))
    plotter.subplot(0, 0)
    plotter.add_mesh(field_grid, scalars="absuh", cmap="seismic", opacity=0.85)
    plotter.show_axes()

    plotter.subplot(0, 1)
    plotter.add_mesh(design_grid, scalars="p", cmap="Greys", clim=(0.0, 1.0), opacity=0.85)
    plotter.show_axes()

    png_path = joinpath(outdir, "old_sandbox_fields.png")
    plotter.show(screenshot=png_path)
    return png_path
end

println("\n=== Old sandbox 3D DOF test ===")
println("Output directory: $(OUTDIR)")

obj = build_old_objective()
p0 = 0.4 .+ 0.2 .* rand(Float64, obj.sim_y.np)
rel_err = test_gradient(obj, p0)
println("PASS = $(rel_err < 1e-4)  (rel_err = $rel_err)")

fields_vtu, design_vtu = write_vtk_outputs(obj, p0)
png_path = save_field_image(fields_vtu, design_vtu)
mesh_path = joinpath(OUTDIR, "mesh.msh")

println("Saved mesh: $mesh_path")
println("Saved VTK:  $fields_vtu")
println("Saved VTK:  $design_vtu")
println("Saved PNG:  $png_path")
