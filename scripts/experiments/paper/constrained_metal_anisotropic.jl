using DistributedEmitterOpt
using LinearAlgebra

# Paper configuration: constrained metal, anisotropic Raman tensor
# This mirrors figures/paper-figures-scripts/data/Constrained_Metal/isotropy_Anisotropic/output-post.txt.

# Output directory for mesh + optimization artifacts
name = "constrained_metal_anisotropic"
root = joinpath(@__DIR__, "runs", name)
mkpath(root)

# Geometry: SymmetricGeometry(L, W, hair, hs, ht, hd, hsub, dpml, l1, l2, l3, verbose)
geo = SymmetricGeometry(
    184.28757605350117, 184.28757605350117, 599.8135303462574,
    399.8756868975049, 199.93784344875246, 100.0, 50.0, 0.0,
    12.5, 2.9, 12.5, 0
)

# Mesh generation (periodic in x/y)
meshfile = joinpath(root, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=true)

# Simulation: foundry mode uses 2D design DOFs
sim = build_simulation(meshfile;
    foundry_mode=true,
    dir_x=false,
    dir_y=true,
    source_y=true
)

# PDE: elastic scattering at 532 nm, y-polarized
env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
inputs = [FieldConfig(532.0, θ=0.0, pol=:y)]
outputs = FieldConfig[]
pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs, α_loss=0.0)

# Objective: anisotropic Raman tensor (alpha_11 = 1.0, alpha_22 = alpha_33 = 0.0)
αₚ = diagm(ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im])
objective = SERSObjective(
    αₚ=αₚ,
    volume=true,
    surface=false,
    use_damage_model=false,
    E_threshold=10000.0
)

# Control: filter + SSP projection + linewidth constraints
control = Control(
    use_filter=true,
    R_filter=(20.0, 20.0, 20.0),
    use_dct=true,
    use_projection=true,
    β=8.0,
    η=0.5,
    use_ssp=true,
    R_ssp=1.5,
    use_constraints=true,
    η_erosion=0.75,
    η_dilation=0.25,
    b1=6.0e-6,
    c0=25600.0,
    use_damage=false,
    γ_damage=1.0,
    E_threshold=10000.0,
    flag_volume=true,
    flag_surface=false
)

# Optimization problem container
prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=true,
    control=control,
    root=root
)

# Initialization: uniform gray
init_uniform!(prob, 0.5)

# Beta continuation schedule (fixed across all paper scripts)
β_schedule = [8.0, 16.0, 32.0, Inf]
g_opt, p_opt = optimize!(prob;
    max_iter=200,
    β_schedule=β_schedule,
    use_constraints=true
)

println("Completed $name: g_opt = $g_opt, p_opt length = $(length(p_opt))")
