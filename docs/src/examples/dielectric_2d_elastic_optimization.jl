# # Dielectric 2D Foundry: Elastic Optimization
#md ```@meta
#md EditURL = "https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl/tree/main/docs/src/examples/dielectric_2d_elastic_optimization.jl"
#md ```
#
# 2D DOF optimization with a dielectric design material (real refractive index).

using DistributedEmitterOpt
using LinearAlgebra

# ## 1. Mesh + simulation (foundry mode)
λ = 1550.0
geo = SymmetricGeometry(λ; L=300.0, W=300.0, hd=120.0, hsub=60.0)
geo.l1 = 60.0
geo.l2 = 30.0
geo.l3 = 60.0

outdir = mktempdir()
meshfile = joinpath(outdir, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=true)

sim = build_simulation(meshfile; foundry_mode=true, dir_x=false, dir_y=false)

# ## 2. Physics (dielectric, elastic)
# Use real refractive indices for dielectrics.
env = Environment(mat_design=2.0, mat_substrate=1.45, mat_fluid=1.0)
inputs = [FieldConfig(λ; θ=0.0, pol=:y)]

# Empty outputs => elastic scattering
pde = MaxwellProblem(env=env, inputs=inputs, outputs=FieldConfig[])

objective = SERSObjective(
    αₚ=Matrix{ComplexF64}(I, 3, 3),
    volume=true,
    surface=false,
    use_damage_model=false
)

# ## 3. Controls
control = Control(
    use_filter=true,
    R_filter=(25.0, 25.0, 25.0),
    use_dct=true,
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true
)

# ## 4. Problem assembly
prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=true,
    control=control,
    root=outdir
)

init_uniform!(prob, 0.5)

# ## 5. Mini optimization (short run)
β_schedule = [8.0, 16.0]
max_iter = 5

(g_opt, p_opt) = optimize!(prob; max_iter=max_iter, β_schedule=β_schedule)

println("Final objective = ", g_opt)
