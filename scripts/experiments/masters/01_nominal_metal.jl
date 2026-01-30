

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 1: Nominal Metal (Isotropic)
# Config: λ=532, Ag/Ag, Foundry Mode, Isotropic
cfg = MastersConfig(
    λ_pump=532.0,
    λ_emission=532.0,
    mat_design="Ag",
    foundry_mode=true,
    anisotropic=false
)

println("=== Running Masters Layout: Nominal Metal ===")
prob, mfile = setup_masters_problem(cfg)
println("Problem initialized. DOFs: $(length(prob.p))")

# Run short optimization
# Real thesis ran for 200 iters with beta ramp. Here we do a comprehensive check.
DistributedEmitterOpt.optimize!(prob; max_iter=10) # Short run for verification
println("Optimization complete.")

rm(mfile; force=true)
