

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 2: Anisotropic Metal
# Config: α=[1 0 0; 0 0 0; 0 0 0]
cfg = MastersConfig(
    λ_pump=532.0,
    λ_emission=532.0,
    mat_design="Ag",
    foundry_mode=true,
    anisotropic=true # Enables x-only polarizability
)

println("=== Running Masters Layout: Anisotropic Metal ===")
prob, mfile = setup_masters_problem(cfg)
println("Problem initialized. Polarizability: $(prob.objective.αₚ)")

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
