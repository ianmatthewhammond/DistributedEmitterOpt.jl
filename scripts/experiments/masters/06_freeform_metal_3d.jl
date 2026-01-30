

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 6: Freeform Metal (3D)
# Config: Ag/Ag, Foundry Mode = FALSE (3D DOFs)
cfg = MastersConfig(
    λ_pump=532.0,
    mat_design="Ag",
    foundry_mode=false # 3D DOFs
)

println("=== Running Masters Layout: Freeform Metal (3D DOFs) ===")
prob, mfile = setup_masters_problem(cfg)
println("Foundry Mode: $(prob.foundry_mode)")
println("DOFs: $(length(prob.p)) (Tetrahedral Elements)")

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
