

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 5: Nominal Dielectric (2D Foundry)
# Config: Si3N4/SiO2, hd=200, hsub=200, Foundry Mode
cfg = MastersConfig(
    λ_pump=532.0,
    mat_design="Si3N4",
    mat_substrate="SiO2",
    hd=200.0,
    hsub=200.0,
    l2=5.0, # Coarser design res for dielectric
    foundry_mode=true
)

println("=== Running Masters Layout: Nominal Dielectric (2D) ===")
prob, mfile = setup_masters_problem(cfg)
println("Material (Design): $(prob.pde.env.mat_design)")
println("Foundry Mode: $(prob.foundry_mode)")

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
