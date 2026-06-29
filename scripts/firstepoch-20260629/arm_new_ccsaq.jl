# =============================================================================
# ARM 3 — NEW (DistributedEmitterOpt) backend=:standalone_ccsaq, first epoch at
# beta=8.  Same mesh + uniform-0.5 init as the other arms, but driven by the
# in-house CCSAQ optimizer.  Per-iteration snapshots come from the guarded
# _firstepoch_snapshot! hook (ENV FIRSTEPOCH_SNAP_DIR).  Quest tim-130116-82977.
# =============================================================================
using Pkg
using LinearAlgebra

include(joinpath(@__DIR__, "config.jl"))

Pkg.activate(NEW_ROOT)
Pkg.instantiate()
import DistributedEmitterOpt as DEO
include(joinpath(@__DIR__, "new_common.jl"))

const ARMDIR = joinpath(RUNROOT, "arm-new-ccsaq")
mkpath(ARMDIR)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE (run meshgen job first)"
cp(MESHFILE, joinpath(ARMDIR, "mesh.msh"); force=true)

# Enable the per-iteration snapshot hook for the standalone backend.
ENV["FIRSTEPOCH_SNAP_DIR"] = joinpath(ARMDIR, "snapshots")
mkpath(ENV["FIRSTEPOCH_SNAP_DIR"])

println("== ARM NEW-ccsaq (DEO backend=:standalone_ccsaq) =="); flush(stdout)
new_prob = build_new_problem(ARMDIR; meshfile=joinpath(ARMDIR, "mesh.msh"))
n = length(new_prob.p)
println("np = $n"); flush(stdout)

DEO.init_uniform!(new_prob, Float64(INIT_VALUE))
@assert new_prob.p == fill(0.5, n) "init is not uniform-0.5"

g_opt, p_opt = DEO.optimize!(new_prob;
    max_iter=MAX_ITER,
    β_schedule=[BETA],
    use_constraints=false,          # trap-3: keep constraints OFF in epoch 1
    tol=TOL,
    backend=:standalone_ccsaq)

# Save the normalized-g trajectory (per-iteration).
open(joinpath(ARMDIR, "history.csv"), "w") do io
    println(io, "iter,g")
    for (i, g) in enumerate(new_prob.g_history)
        println(io, string(i, ",", g))
    end
end

import JLD2
JLD2.save(joinpath(ARMDIR, "snapshots", "final.jld2"),
    Dict("g_opt" => g_opt, "p_opt" => collect(Float64, p_opt),
         "g_history" => collect(Float64, new_prob.g_history)))

println("ARM NEW-ccsaq complete: g_opt(norm)=$g_opt, iters=$(length(new_prob.g_history)) -> $ARMDIR")
