```@meta
EditURL = "https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl/tree/main/docs/src/examples/anisotropic_3d_inelastic_optimization.jl"
```

# Anisotropic 3D inelastic: mini optimization

3D DOF example with anisotropic polarizability and inelastic scattering.
Two emission outputs at the same wavelength but different polarizations.

Tip: use the page's "Edit on GitHub" link to download the source `.jl` script.

```julia
using DistributedEmitterOpt
using LinearAlgebra

# Mesh + simulation (3D DOF)
λ_pump = 532.0
λ_emission = 600.0

geo = SymmetricGeometry(λ_pump; L=100.0, W=100.0, hd=80.0, hsub=40.0)
geo.l1 = 50.0
geo.l2 = 30.0
geo.l3 = 50.0
geo.hair = 200.0
geo.hs = 120.0
geo.ht = 80.0

outdir = mktempdir()
meshfile = joinpath(outdir, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=false)  # PEC symmetry in Y

sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=true)

# Physics (inelastic, two outputs)
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
inputs = [FieldConfig(λ_pump; θ=0.0, pol=:y)]
outputs = [
    FieldConfig(λ_emission; θ=0.0, pol=:y, weight=1.0),
    FieldConfig(λ_emission; θ=0.0, pol=:x, weight=0.7)
]

pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

# Objective (anisotropic Raman tensor)
function anisotropic_tensor()
    α = ComplexF64[
        1.10+0.00im  0.05+0.02im  0.01-0.03im
        0.02-0.01im  0.95+0.00im  0.03+0.04im
        0.01+0.00im  0.02-0.02im  1.05+0.00im
    ]
    return (α + transpose(α)) / 2
end

objective = SERSObjective(
    αₚ=anisotropic_tensor(),
    volume=true,
    surface=false,
    use_damage_model=false
)

# Controls
control = Control(
    use_filter=true,
    R_filter=(20.0, 20.0, 20.0),
    use_dct=false,  # Helmholtz filter for 3D
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true
)

# Assemble problem
prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=false,
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
