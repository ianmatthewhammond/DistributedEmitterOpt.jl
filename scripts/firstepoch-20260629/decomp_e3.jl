# =============================================================================
# DECOMP (OLD / Emitter3D) — single-repo iter-1 evaluation at uniform-0.5.
# Memory-safe: holds only ONE simulation/factorization (the combined two-repo
# job OOM'd at ~183GB). Saves baseline g(p=0), raw g(p=0.5), gradient + ‖grad‖,
# and np for the parity check. Quest tim-130116-82977.
# =============================================================================
using Pkg
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "config.jl"))

Pkg.activate(OLD_ROOT)   # main Emitter3D checkout, read-only; no instantiate
import Emitter3DTopOpt as e3
include(e3.includesolver("Umfpack")); import .UmfpackSolver as OldUmfpackSolver
include(e3.includescript("Setup")); using .Setup
include(joinpath(@__DIR__, "old_common.jl"))
import JLD2

const DCDIR = joinpath(RUNROOT, "decomp")
const E3DIR = joinpath(DCDIR, "e3")
mkpath(E3DIR)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE"
cp(MESHFILE, joinpath(E3DIR, "mesh.msh"); force=true)

println("== DECOMP OLD (Emitter3D) single-repo =="); flush(stdout)
old_obj = build_old_objective(E3DIR; meshfile="mesh.msh")
old_obj.control.β = BETA
n = old_obj.sim_y.np
println("np_old = $n"); flush(stdout)

g_base = e3.stepobjective(zeros(n), Float64[]; obj=old_obj)
println("g_base_old (p=0) = $g_base"); flush(stdout)

p0 = fill(Float64(INIT_VALUE), n)
@assert p0 == fill(0.5, n)
grad = zeros(n)
g_raw = e3.stepobjective(p0, grad; obj=old_obj)
nrm = norm(grad)
@printf("g_raw_old(0.5)=%.12e  g_normd_old=%.12e  ||grad_old||=%.12e\n",
    g_raw, g_raw / g_base, nrm)
flush(stdout)

JLD2.save(joinpath(DCDIR, "e3_decomp.jld2"), Dict(
    "np" => n, "g_base" => g_base, "g_raw" => g_raw, "g_norm" => g_raw / g_base,
    "grad" => grad, "grad_norm" => nrm, "p0" => p0))
open(joinpath(DCDIR, "e3_decomp.txt"), "w") do io
    @printf(io, "np_old=%d\ng_base_old=%.12e\ng_raw_old=%.12e\ng_normd_old=%.12e\nnorm_grad_old=%.12e\n",
        n, g_base, g_raw, g_raw / g_base, nrm)
end
println("DECOMP OLD done -> $DCDIR/e3_decomp.{jld2,txt}")
