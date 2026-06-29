# =============================================================================
# ARM 1 — OLD (Emitter3DTopOpt, NLopt LD_CCSAQ) first epoch at beta=8.
# Loads the shared mesh (genmesh=false), starts from uniform-0.5, runs the
# shared manual NLopt epoch.  Quest tim-130116-82977.
# =============================================================================
using Pkg
using LinearAlgebra

include(joinpath(@__DIR__, "config.jl"))

Pkg.activate(OLD_ROOT)   # read-only main checkout; no instantiate
import Emitter3DTopOpt as e3
include(e3.includesolver("Umfpack")); import .UmfpackSolver as OldUmfpackSolver
include(e3.includescript("Setup")); using .Setup
include(joinpath(@__DIR__, "old_common.jl"))
include(joinpath(@__DIR__, "manual_loop.jl"))

const ARMDIR = joinpath(RUNROOT, "arm-old")
mkpath(ARMDIR)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE (run meshgen job first)"
cp(MESHFILE, joinpath(ARMDIR, "mesh.msh"); force=true)

println("== ARM OLD (Emitter3DTopOpt / NLopt LD_CCSAQ) =="); flush(stdout)
old_obj = build_old_objective(ARMDIR; meshfile="mesh.msh")
old_obj.control.β = BETA
n = old_obj.sim_y.np
println("np = $n"); flush(stdout)

g_norm = e3.stepobjective(zeros(n), Float64[]; obj=old_obj)
println("g_norm (flat-substrate baseline) = $g_norm"); flush(stdout)

p0 = fill(Float64(INIT_VALUE), n)
@assert p0 == fill(0.5, n)

objfun! = (p, grad) -> e3.stepobjective(p, grad; obj=old_obj)

run_manual_epoch(objfun!, p0, g_norm;
    snapdir=joinpath(ARMDIR, "snapshots"),
    historypath=joinpath(ARMDIR, "history.csv"),
    max_iter=MAX_ITER, tol=TOL, label="OLD")

println("ARM OLD complete -> $ARMDIR")
