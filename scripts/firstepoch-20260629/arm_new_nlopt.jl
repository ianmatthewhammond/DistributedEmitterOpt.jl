# =============================================================================
# ARM 2 — NEW (DistributedEmitterOpt) backend=:nlopt first epoch at beta=8.
# PRIMARY head-to-head vs ARM OLD: SAME mesh, SAME uniform-0.5 init, SAME NLopt
# LD_CCSAQ loop; only the objective/gradient/filter assembly differs.
# Quest tim-130116-82977.
# =============================================================================
using Pkg
using LinearAlgebra

include(joinpath(@__DIR__, "config.jl"))

Pkg.activate(NEW_ROOT)
ensure_instantiated()
import DistributedEmitterOpt as DEO
include(joinpath(@__DIR__, "new_common.jl"))
include(joinpath(@__DIR__, "manual_loop.jl"))

const ARMDIR = joinpath(RUNROOT, "arm-new-nlopt")
mkpath(ARMDIR)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE (run meshgen job first)"
cp(MESHFILE, joinpath(ARMDIR, "mesh.msh"); force=true)

println("== ARM NEW-nlopt (DEO backend=:nlopt / NLopt LD_CCSAQ) =="); flush(stdout)
new_prob = build_new_problem(ARMDIR; meshfile=joinpath(ARMDIR, "mesh.msh"))
new_prob.control.β = BETA
n = length(new_prob.p)
println("np = $n"); flush(stdout)

g_norm = DEO.objective_and_gradient!(Float64[], zeros(n), new_prob)
println("g_norm (flat-substrate baseline) = $g_norm"); flush(stdout)

p0 = fill(Float64(INIT_VALUE), n)
@assert p0 == fill(0.5, n)

objfun! = (p, grad) -> DEO.objective_and_gradient!(grad, p, new_prob)

run_manual_epoch(objfun!, p0, g_norm;
    snapdir=joinpath(ARMDIR, "snapshots"),
    historypath=joinpath(ARMDIR, "history.csv"),
    max_iter=MAX_ITER, tol=TOL, label="NEW-nlopt")

println("ARM NEW-nlopt complete -> $ARMDIR")
