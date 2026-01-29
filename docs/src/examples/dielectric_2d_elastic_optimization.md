```@meta
EditURL = "https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl/tree/main/docs/src/examples/dielectric_2d_elastic_optimization.jl"
```

# Dielectric 2D foundry: elastic optimization

2D DOF optimization with a dielectric design material (real refractive index).

Tip: use the page's "Edit on GitHub" link to download the source `.jl` script.

```julia
using DistributedEmitterOpt
using LinearAlgebra

# Mesh + simulation (foundry mode)
λ = 1550.0
geo = SymmetricGeometry(λ; L=300.0, W=300.0, hd=120.0, hsub=60.0)
geo.l1 = 60.0
geo.l2 = 30.0
geo.l3 = 60.0

outdir = mktempdir()
meshfile = joinpath(outdir, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=true)

sim = build_simulation(meshfile; foundry_mode=true, dir_x=false, dir_y=false)

# Physics (dielectric, elastic)
env = Environment(mat_design=2.0, mat_substrate=1.45, mat_fluid=1.0)
inputs = [FieldConfig(λ; θ=0.0, pol=:y)]

pde = MaxwellProblem(env=env, inputs=inputs, outputs=FieldConfig[])

objective = SERSObjective(
    αₚ=Matrix{ComplexF64}(I, 3, 3),
    volume=true,
    surface=false,
    use_damage_model=false
)

# Controls
control = Control(
    use_filter=true,
    R_filter=(25.0, 25.0, 25.0),
    use_dct=true,
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true
)

# Assemble problem
prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=true,
    control=control,
    root=outdir
)

init_uniform!(prob, 0.5)

# Short optimization run (increase for real runs)
β_schedule = [8.0, 16.0]
max_iter = 5

(g_opt, p_opt) = optimize!(prob; max_iter=max_iter, β_schedule=β_schedule)

println("Final objective = ", g_opt)
```
