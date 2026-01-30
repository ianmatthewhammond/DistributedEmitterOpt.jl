

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 7: Freeform Dielectric (3D)
# Config: Si3N4/SiO2, foundry_mode=false, full_cell=true
cfg = MastersConfig(
    λ_pump=532.0,
    mat_design="Si3N4",
    mat_substrate="SiO2",
    hd=200.0,
    hsub=200.0,
    foundry_mode=false, # 3D DOFs
    full_cell=true
)

println("=== Running Masters Layout: Freeform Dielectric (3D DOFs) ===")
prob, mfile = setup_masters_problem(cfg)
println("Foundry Mode: $(prob.foundry_mode)")
println("Material: $(prob.pde.env.mat_design)")

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
