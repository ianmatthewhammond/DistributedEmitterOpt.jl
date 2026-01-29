# # Anisotropic 3D Inelastic: Mini Optimization
#md ```@meta
#md EditURL = "https://github.com/ianmatthewhammond/DistributedEmitterOpt.jl/tree/main/docs/src/examples/anisotropic_3d_inelastic_optimization.jl"
#md ```
#
# 3D DOF example with anisotropic polarizability and inelastic scattering.
# We use two emission outputs at the same wavelength but different polarizations.

using DistributedEmitterOpt
using LinearAlgebra

# ## 1. Mesh + simulation (3D DOF)
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
# For 3D FE mode we use PEC symmetry in Y (per_y=false)
genmesh(geo, meshfile; per_x=true, per_y=false)

sim = build_simulation(meshfile; foundry_mode=false, dir_x=false, dir_y=true)

# ## 2. Physics (inelastic with two outputs)
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
inputs = [FieldConfig(λ_pump; θ=0.0, pol=:y)]
outputs = [
    FieldConfig(λ_emission; θ=0.0, pol=:y, weight=1.0),
    FieldConfig(λ_emission; θ=0.0, pol=:x, weight=0.7)
]

pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

# ## 3. Objective (anisotropic)
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

# ## 4. Controls
control = Control(
    use_filter=true,
    R_filter=(20.0, 20.0, 20.0),
    use_dct=false,  # Helmholtz filter in 3D
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true
)

# ## 5. Problem assembly
prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=false,
    control=control,
    root=outdir
)

init_uniform!(prob, 0.5)

# ## 6. Mini optimization (short run)
# Keep this tiny for demonstration; increase for real runs.
β_schedule = [8.0, 16.0]
max_iter = 5

(g_opt, p_opt) = optimize!(prob; max_iter=max_iter, β_schedule=β_schedule)

println("Final objective = ", g_opt)
