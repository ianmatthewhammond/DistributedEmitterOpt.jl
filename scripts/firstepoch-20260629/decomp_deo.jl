# =============================================================================
# DECOMP (NEW / DEO) — single-repo iter-1 eval at uniform-0.5, then COMBINE with
# the OLD side (loads the tiny e3_decomp.jld2, NOT the Emitter3D simulation) to
# produce the parity gate + the full iter-1 decomposition that explains
# normalized 0.305 (NEW) vs 0.537 (OLD). Memory-safe (one DEO simulation).
# Quest tim-130116-82977.
# =============================================================================
using Pkg
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "config.jl"))

Pkg.activate(NEW_ROOT)
ensure_instantiated()
import DistributedEmitterOpt as DEO
include(joinpath(@__DIR__, "new_common.jl"))
import JLD2

const DCDIR = joinpath(RUNROOT, "decomp")
const DEODIR = joinpath(DCDIR, "deo")
mkpath(DEODIR)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE"
cp(MESHFILE, joinpath(DEODIR, "mesh.msh"); force=true)

println("== DECOMP NEW (DEO) single-repo =="); flush(stdout)
new_prob = build_new_problem(DEODIR; meshfile=joinpath(DEODIR, "mesh.msh"))
new_prob.control.β = BETA
n = length(new_prob.p)
println("np_new = $n"); flush(stdout)

g_norm = DEO.objective_and_gradient!(Float64[], zeros(n), new_prob)
println("g_norm_new (p=0) = $g_norm"); flush(stdout)

p0 = fill(Float64(INIT_VALUE), n)
@assert p0 == fill(0.5, n)
grad = zeros(n)
g_raw = DEO.objective_and_gradient!(grad, p0, new_prob)
nrm = norm(grad)
@printf("g_raw_new(0.5)=%.12e  g_normd_new=%.12e  ||grad_new||=%.12e\n",
    g_raw, g_raw / g_norm, nrm)
flush(stdout)

# --- Combine with the OLD side (tiny load, no Emitter3D simulation) ----------
e3f = joinpath(DCDIR, "e3_decomp.jld2")
@assert isfile(e3f) "e3_decomp.jld2 missing — decomp_e3 must run first"
e3d = JLD2.load(e3f)
np_old   = e3d["np"]
g_base_o = e3d["g_base"]
g_raw_o  = e3d["g_raw"]
g_old_n  = e3d["g_norm"]
grad_old = e3d["grad"]
nrm_old  = e3d["grad_norm"]

parity_ok = (np_old == n)
g_new_n = g_raw / g_norm
cosg = dot(grad_old, grad) / max(nrm_old * nrm, 1e-300)
relg = norm(grad .- grad_old) / max(nrm_old, 1e-300)

open(joinpath(DCDIR, "decomp.txt"), "w") do io
    @printf(io, "ITER-1 DECOMPOSITION (uniform-0.5, beta=%g, alpha=0) — split single-repo\n", BETA)
    @printf(io, "PARITY: %s  (np_old=%d  np_new=%d)\n", parity_ok ? "PASS" : "FAIL", np_old, n)
    @printf(io, "g_base_old (p=0)    = %.12e\n", g_base_o)
    @printf(io, "g_norm_new (p=0)    = %.12e\n", g_norm)
    @printf(io, "baseline ratio n/o  = %.12e\n", g_norm / g_base_o)
    @printf(io, "g_raw_old (p=0.5)   = %.12e\n", g_raw_o)
    @printf(io, "g_raw_new (p=0.5)   = %.12e\n", g_raw)
    @printf(io, "g_raw ratio n/o     = %.12e\n", g_raw / g_raw_o)
    @printf(io, "g_normd_old         = %.12e\n", g_old_n)
    @printf(io, "g_normd_new         = %.12e\n", g_new_n)
    @printf(io, "g_normd ratio n/o   = %.12e\n", g_new_n / g_old_n)
    @printf(io, "||grad_old||        = %.12e\n", nrm_old)
    @printf(io, "||grad_new||        = %.12e\n", nrm)
    @printf(io, "cos(grad_o,grad_n)  = %.12e\n", cosg)
    @printf(io, "rel||dgrad||/||o||  = %.12e\n", relg)
end
JLD2.save(joinpath(DCDIR, "decomp.jld2"), Dict(
    "np_old" => np_old, "np_new" => n, "parity_ok" => parity_ok,
    "g_base_old" => g_base_o, "g_norm_new" => g_norm,
    "g_raw_old" => g_raw_o, "g_raw_new" => g_raw,
    "g_normd_old" => g_old_n, "g_normd_new" => g_new_n,
    "grad_norm_old" => nrm_old, "grad_norm_new" => nrm,
    "cos_grad" => cosg, "rel_grad" => relg,
    "grad_old" => grad_old, "grad_new" => grad, "p0" => p0))

# best-effort extras (written after the core numbers)
open(joinpath(DCDIR, "decomp.txt"), "a") do io
    println(io, "\n--- extra intermediates (best-effort) ---")
    try
        println(io, "NEW channels: inputs=", length(new_prob.pde.inputs),
                    " outputs=", length(new_prob.pde.outputs))
    catch e
        println(io, "channels err: ", e)
    end
    try
        sim0 = DEO.default_sim(new_prob.sim)
        pf = DEO.filter_grid(p0, sim0, new_prob.control)
        println(io, "NEW filtered pf @ 0.5: min=", minimum(pf), " max=", maximum(pf),
                    " mean=", sum(pf)/length(pf))
    catch e
        println(io, "filter err: ", e)
    end
end

println("\n===== ITER-1 DECOMPOSITION ====="); flush(stdout)
println(read(joinpath(DCDIR, "decomp.txt"), String))
parity_ok || error("PARITY FAIL: np_old=$np_old np_new=$n")
println("DECOMP NEW + combine done -> $DCDIR/decomp.{txt,jld2}")
