using DistributedEmitterOpt
using LinearAlgebra

# Paper configuration: constrained dielectric, inelastic (pump/emission split)
# This mirrors figures/paper-figures-scripts/data/Constrained_Dielectric/frequency_Shifted/output-post.txt.

# Output directory for mesh + optimization artifacts
name = "constrained_dielectric_frequency_shifted"
root = joinpath(@__DIR__, "runs", name)
mkpath(root)

# Geometry: SymmetricGeometry(L, W, hair, hs, ht, hd, hsub, dpml, l1, l2, l3, verbose)
geo = SymmetricGeometry(
    242.00213642879163, 242.00213642879163, 609.3970172032934,
    406.264678135529, 203.1323390677645, 200.0, 200.0, 0.0,
    10.0, 5.0, 10.0, 0
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

# PDE: pump at 532 nm, emission at 549 nm (inelastic)
env = Environment(mat_design="Si3N4", mat_substrate="SiO2", mat_fluid=1.33)
inputs = [FieldConfig(532.0, θ=0.0, pol=:y)]
outputs = [FieldConfig(549.0, θ=0.0, pol=:y)]
pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs, α_loss=0.0)

# Objective: isotropic SERS (identity polarizability), volume integral
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
    R_ssp=11.0,
    use_constraints=true,
    η_erosion=0.75,
    η_dilation=0.25,
    b1=1.0e-8,
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
# Alpha loss schedule for dielectric (4 epochs)
α_schedule = [0.1, 0.01, 0.001, 0.001]
g_opt, p_opt = optimize!(prob;
    max_iter=200,
    β_schedule=β_schedule,
    α_schedule=α_schedule,
    use_constraints=true
)

g_history = prob.g_history

println("Completed $name: g_opt = $g_opt, p_opt length = $(length(p_opt))")

# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------

println("\n--- Starting Post-Processing ---")

# 1. Visualize results (VTK)
Analysis.visualize_results(prob, p_opt; root=root)

# 2. Plot iteration history
Analysis.plot_iteration_history(g_history; root=root)

# 3. Spectral sweep (Robustness to wavelength shift)
# Center at 532 nm, +/- 50 nm range
Analysis.spectral_sweep(prob, p_opt; center_λ=532.0, range_λ=50.0, root=root)

# 4. Fabrication tolerance sweep (Robustness to filter radius / erosion)
# nominal R=20 nm, sweep +/- 10 nm
Analysis.fabrication_sweep(prob, p_opt; center_R=20.0, range_R=10.0, root=root)

println("--- Post-Processing Done ---")
