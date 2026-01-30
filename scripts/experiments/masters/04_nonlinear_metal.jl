

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 4: Nonlinear Metal
# Config: Nonlinear damage model enabled, E_threshold=50.0
cfg = MastersConfig(
    λ_pump=532.0,
    mat_design="Ag",
    foundry_mode=true,
    nonlinear=true,
    E_threshold=50.0
)

println("=== Running Masters Layout: Nonlinear Metal ===")
prob, mfile = setup_masters_problem(cfg)
println("Damage Model: $(prob.objective.use_damage_model)")
println("E_threshold: $(prob.objective.E_threshold)")

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
