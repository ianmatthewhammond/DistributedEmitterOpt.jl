"""
    Compare old vs new cache factors (sandbox 3D DOF config).

Run from REPL:
    julia> include("scripts/experiments/sandbox_3d_cache_compare/compare_old_new_caches.jl")
"""

using Pkg

const OLD_ROOT = "/Users/ianhammond/GitHub/Emitter3DTopOpt"
const OUTDIR = @__DIR__
const PERTURBATION = 1e-8

Pkg.activate(OLD_ROOT)
using Revise

import Emitter3DTopOpt as e3
include(e3.includesolver("Umfpack"))
import .UmfpackSolver as OldUmfpackSolver
include(e3.includescript("Setup"))
using .Setup

using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random
using SparseArrays

mkpath(OUTDIR)

const ALPHA_P = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)
const NF_SANDBOX = sqrt(1.77)

"""Generate a mesh using the new-code geometry path."""
function generate_new_mesh(; outdir::String=OUTDIR)
    hr = 532.0 / NF_SANDBOX / 2
    geo = SymmetricGeometry(532.0; L=150.0, W=150.0, hd=150.0, hsub=50.0)
    geo.ht = hr
    geo.hs = hr + hr
    geo.hair = hr + hr + hr
    geo.l1 = 30.0
    geo.l2 = 20.0
    geo.l3 = 30.0

    meshfile = joinpath(outdir, "new_mesh.msh")
    genperiodic(geo, meshfile; per_x=false, per_y=false)
    return meshfile
end

function build_old_objective(; outdir::String=OUTDIR)
    reuse_y1 = OldUmfpackSolver.UMFPACK_Reuse()
    reuse_y2 = reuse_y1
    reuse_x = OldUmfpackSolver.UMFPACK_Reuse()

    obj = Setup.SetupAll(e3;
        root=outdir * "/",
        genmesh=true,
        meshfile="old_mesh.msh",

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
        flag_foundry=false,
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

function build_new_problem(meshfile::String; outdir::String=OUTDIR)
    sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=true)

    objective = SERSObjective(
        αₚ=ALPHA_P,
        volume=true,
        surface=false,
        use_damage_model=false,
        E_threshold=10.0
    )

    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=NF_SANDBOX)
    inputs = [FieldConfig(532.0; θ=0.0, pol=:y)]
    outputs = FieldConfig[]
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=false,
        use_projection=true,
        β=8.0,
        η=0.5,
        use_ssp=true,
        R_ssp=2.0,
        flag_volume=true,
        flag_surface=false,
        use_damage=false,
        E_threshold=10.0
    )

    solver = DistributedEmitterOpt.UmfpackSolver()
    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=false,
        control=control
    )

    return prob
end

function describe_matrix(label, mat)
    if isnothing(mat)
        println("  $label: <none>")
        return
    end
    if mat isa AbstractMatrix
        nnz_val = mat isa SparseArrays.AbstractSparseMatrix ? nnz(mat) : "n/a"
        println("  $label: size=$(size(mat)) nnz=$nnz_val norm=$(norm(mat))")
    else
        println("  $label: $(summary(mat))")
    end
end

function describe_factor(label, factor)
    println("== $label ==")
    if isnothing(factor)
        println("  <none>")
        return
    end
    println("  type: $(typeof(factor))")
    for prop in (:L, :U, :F, :R, :p, :q)
        if hasproperty(factor, prop)
            describe_matrix(string(prop), getproperty(factor, prop))
        end
    end
end

function compare_vectors(label, a, b)
    if isnothing(a) || isnothing(b)
        println("== $label ==\n  one or both vectors missing")
        return
    end
    println("== $label ==")
    println("  a: length=$(length(a)) norm=$(norm(a))")
    println("  b: length=$(length(b)) norm=$(norm(b))")
    if length(a) == length(b)
        println("  norm(a-b) = $(norm(a - b))")
    end
end

println("\n=== Cache/LU comparison (sandbox 3D) ===")
println("Output directory: $(OUTDIR)")

new_meshfile = generate_new_mesh()
println("New mesh: $new_meshfile")

# Old
old_obj = build_old_objective()
Random.seed!(2)
p0_old = 0.4 .+ 0.2 .* rand(Float64, old_obj.sim_y.np)
grad_old = zeros(old_obj.sim_y.np)
g_old = e3.stepobjective(p0_old, grad_old; obj=old_obj)
_ = e3.filter!(p0_old, old_obj)  # populate filter LU

# New
new_prob = build_new_problem(new_meshfile)
Random.seed!(2)
p0_new = 0.4 .+ 0.2 .* rand(Float64, new_prob.sim.np)
grad_new = zeros(new_prob.sim.np)
g_new = objective_and_gradient!(grad_new, p0_new, new_prob)
_ = filter_helmholtz!(p0_new, new_prob.pool.filter_cache, new_prob.sim, new_prob.control)

println("\n=== Objective values ===")
println("old g = $g_old")
println("new g = $g_new")

# Pull caches
old_reuse = old_obj.reuse_y1
new_cache = isempty(new_prob.pool.caches) ? nothing : first(values(new_prob.pool.caches))

println("\n=== Maxwell LU factors ===")
describe_factor("old A (reuse_y1.Alu)", getproperty(old_reuse, :Alu))
describe_factor("new A (pool cache)", isnothing(new_cache) ? nothing : new_cache.A_factor)

println("\n=== Filter LU factors ===")
describe_factor("old F (reuse_y1.FAlu)", getproperty(old_reuse, :FAlu))
describe_factor("new F (pool.filter_cache)", new_prob.pool.filter_cache.F_factor)

println("\n=== Solution vectors ===")
compare_vectors("old vs new solution vectors", old_reuse.x, isnothing(new_cache) ? nothing : new_cache.x)
