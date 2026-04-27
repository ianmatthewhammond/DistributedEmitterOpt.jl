using Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
using DistributedEmitterOpt
using DistributedEmitterOpt.Analysis
using LinearAlgebra

# Checkpoint-CCSAQ experiment: constrained metal nominal case using the
# standalone CCSAQ backend with checkpointing enabled. Identical physics
# and geometry to the paper configuration — only the optimizer backend differs.

name = "checkpoint_ccsaq_metal_nominal"
root = joinpath(@__DIR__, "runs", name)
mkpath(root)

# Geometry (identical to paper)
geo = SymmetricGeometry(
    184.28757605350117, 184.28757605350117, 599.8135303462574,
    399.8756868975049, 199.93784344875246, 100.0, 50.0, 0.0,
    25.5, 25.5, 25.5, 0
)

meshfile = joinpath(root, "mesh.msh")
genmesh(geo, meshfile; per_x=true, per_y=true)

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

# Objective: isotropic SERS
objective = SERSObjective(
    αₚ=Matrix{ComplexF64}(LinearAlgebra.I, 3, 3),
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

prob = OptimizationProblem(pde, objective, sim, UmfpackSolver();
    foundry_mode=true,
    control=control,
    root=root
)

init_uniform!(prob, 0.5)

# Beta continuation (paper schedule)
β_schedule = [8.0, 16.0, 32.0, Inf]

g_opt, p_opt = optimize!(prob;
    max_iter=200,
    β_schedule=β_schedule,
    use_constraints=true,
    backup=true,
    backup_every=20,
    backend=:standalone_ccsaq
)
g_history = prob.g_history

println("Completed $name: g_opt = $g_opt, p_opt length = $(length(p_opt))")

# Post-Processing
println("\n--- Starting Post-Processing ---")
Analysis.visualize_results(prob, p_opt; root=root)
Analysis.plot_iteration_history(g_history; root=root)
Analysis.spectral_sweep(prob, p_opt; center_λ=532.0, range_λ=50.0, root=root)
Analysis.fabrication_sweep(prob, p_opt; center_R=20.0, range_R=10.0, root=root)
println("--- Post-Processing Done ---")
