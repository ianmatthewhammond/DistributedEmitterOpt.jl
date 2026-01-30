

using DistributedEmitterOpt
using Gridap
include("common.jl")
using .MastersCommon

# Case 3: Splitting
# Config: λ_pump=532.0, λ_emission=549.0 (Stokes Shift)
cfg = MastersConfig(
    λ_pump=532.0,
    λ_emission=549.0,
    mat_design="Ag",
    foundry_mode=true,
    anisotropic=false
)

println("=== Running Masters Layout: Splitting (Stockes Shift) ===")
prob, mfile = setup_masters_problem(cfg)
println("Pump: $(prob.pde.inputs[1].wavelength) nm")
# Ensure outputs are configured
if isempty(prob.pde.outputs)
    # Manual update if common logic didn't catch it
    # But common.jl logic for splitting relies on λ_emission != λ_pump
    println("Outputs: $([o.wavelength for o in prob.pde.outputs]) nm")
end

DistributedEmitterOpt.optimize!(prob; max_iter=10)
println("Optimization complete.")

rm(mfile; force=true)
