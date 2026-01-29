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
import Gmsh: gmsh
using LinearAlgebra
using Random
using SparseArrays

mkpath(OUTDIR)

const ALPHA_P = Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)
const NF_SANDBOX = sqrt(1.77)

"""Generate a mesh using the new-code geometry path."""
function generate_new_mesh(; outdir::String=OUTDIR)
    hr = 532.0 / NF_SANDBOX / 2
    # Build geometry using the legacy struct layout, but call the copied generator in DistributedEmitterOpt.
    geo = DistributedEmitterOpt.SymmetricGeometry(150.0, 150.0, hr + hr + hr, hr + hr, hr, 150.0, 50.0, 0.0, 30.0, 20.0, 30.0, 0)
    meshfile = joinpath(outdir, "new_mesh.msh")
    DistributedEmitterOpt.genperiodic(geo, meshfile; per_x=false, per_y=false)
    return meshfile, geo, (false, false)
end

function describe_geometry(label, geo; per_x::Bool, per_y::Bool)
    z0 = -(geo.hair + geo.hd + geo.hsub) / 2
    println("== $label geometry ==")
    println("  L=$(geo.L) W=$(geo.W)")
    println("  hd=$(geo.hd) hsub=$(geo.hsub)")
    println("  ht=$(geo.ht) hs=$(geo.hs) hair=$(geo.hair)")
    println("  l1=$(geo.l1) l2=$(geo.l2) l3=$(geo.l3)")
    println("  per_x=$per_x per_y=$per_y z0=$z0")
end

function describe_physics(label, phys)
    if isnothing(phys)
        println("== $label physics ==\n  missing")
        return
    end
    println("== $label physics ==")
    println("  nf=$(phys.nf) θ=$(phys.θ) ω=$(phys.ω)")
end

function summarize_mesh(meshfile::String)
    if !isfile(meshfile)
        println("== mesh summary ($meshfile) ==\n  missing file")
        return
    end
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(meshfile)

    entities = gmsh.model.getEntities()
    counts = Dict(0 => 0, 1 => 0, 2 => 0, 3 => 0)
    for (dim, _) in entities
        counts[dim] = get(counts, dim, 0) + 1
    end

    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    types, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
    total_elems = sum(length.(elemTags))

    xmin = Inf
    ymin = Inf
    zmin = Inf
    xmax = -Inf
    ymax = -Inf
    zmax = -Inf
    coord_sum = 0.0
    for i in 1:3:length(coords)
        x = coords[i]
        y = coords[i + 1]
        z = coords[i + 2]
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        zmin = min(zmin, z)
        xmax = max(xmax, x)
        ymax = max(ymax, y)
        zmax = max(zmax, z)
        coord_sum += x + y + z
    end

    println("== mesh summary ($meshfile) ==")
    println("  entities: 0D=$(counts[0]) 1D=$(counts[1]) 2D=$(counts[2]) 3D=$(counts[3])")
    println("  nodes=$(length(nodeTags)) elements=$(total_elems)")
    println("  bbox=([$xmin,$xmax], [$ymin,$ymax], [$zmin,$zmax])")
    println("  coord_sum=$(coord_sum)")
    println("  elem_types=$(types)")
    println("  elem_counts=$(length.(elemTags))")

    gmsh.finalize()
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

new_meshfile, new_geo, (new_per_x, new_per_y) = generate_new_mesh()
println("New mesh: $new_meshfile")

# Old
old_obj = build_old_objective()
summarize_mesh(joinpath(OUTDIR, "old_mesh.msh"))
describe_geometry("old", old_obj.geometry; per_x=false, per_y=false)
describe_physics("old", old_obj.phys_1)
Random.seed!(2)
p0_old = 0.4 .+ 0.2 .* rand(Float64, old_obj.sim_y.np)
grad_old = zeros(old_obj.sim_y.np)
g_old = e3.stepobjective(p0_old, grad_old; obj=old_obj)
_ = e3.filter!(p0_old, old_obj)  # populate filter LU

# New
summarize_mesh(new_meshfile)
new_prob = build_new_problem(new_meshfile)
describe_geometry("new", new_geo; per_x=new_per_x, per_y=new_per_y)
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
