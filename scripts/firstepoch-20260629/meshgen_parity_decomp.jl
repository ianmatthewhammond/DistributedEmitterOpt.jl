# =============================================================================
# JOB 0 — mesh generation + parity GATE + iter-1 decomposition.
# Quest tim-130116-82977 (first-epoch beta=8 OLD-vs-NEW divergence study).
#
# 1. Generate the ONE shared paper-grade metal mesh (DEO genmesh).
# 2. Build OLD (Emitter3DTopOpt) and NEW (DEO) objectives from that SAME mesh.
# 3. PARITY GATE: report mesh node/element/physical-tag counts and assert the
#    two codes derive the SAME number of design DOFs (else snapshots can't align
#    -> error out; the human must inspect parity.txt and decide).
# 4. ITER-1 DECOMPOSITION (the sharpest test): at the shared uniform-0.5 design,
#    on the same mesh with α=0 and β=8, report raw g, normalized g, ||grad|| and
#    cos(grad_old,grad_new) for OLD vs NEW. Differences here localize the split
#    to objective/gradient/filter ASSEMBLY, independent of optimizer dynamics.
# =============================================================================

using Pkg
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "config.jl"))

mkpath(MESHDIR)
const OUTDIR = joinpath(RUNROOT, "meshgen")
mkpath(OUTDIR)

println("== first-epoch meshgen+parity+decomp ==")
println("OLD_ROOT = $OLD_ROOT")
println("NEW_ROOT = $NEW_ROOT")
println("RUNROOT  = $RUNROOT")
println("MESHFILE = $MESHFILE")
flush(stdout)

# --- OLD environment ---------------------------------------------------------
Pkg.activate(OLD_ROOT)
Pkg.instantiate()
import Emitter3DTopOpt as e3
include(e3.includesolver("Umfpack")); import .UmfpackSolver as OldUmfpackSolver
include(e3.includescript("Setup")); using .Setup
include(joinpath(@__DIR__, "old_common.jl"))

# --- NEW environment ---------------------------------------------------------
Pkg.activate(NEW_ROOT)
Pkg.instantiate()
import DistributedEmitterOpt as DEO
import JLD2
include(joinpath(@__DIR__, "new_common.jl"))

# --- 1. Use the pre-copied authentic paper mesh (login-safe cp; NO genmesh) ---
# The ONE shared mesh is the authentic 38MB paper mesh, copied into MESHDIR
# OUTSIDE Julia (login-safe). This job never generates a mesh on any node.
println("\n[1] using pre-copied shared mesh ..."); flush(stdout)
@assert isfile(MESHFILE) "shared mesh missing: $MESHFILE — copy the authentic " *
    "paper mesh in first (login-safe cp); this job does NOT genmesh."
println("    mesh present ($(round(filesize(MESHFILE)/2^20, digits=1)) MiB)")

# Tolerant .msh stat scan (handles gmsh 2.2 and 4.1 headers).
function mesh_stats(path)
    nnodes = -1; nelem = -1; phys = String[]
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        l = strip(lines[i])
        if l == "\$PhysicalNames"
            cnt = parse(Int, strip(lines[i+1]))
            for j in 1:cnt
                parts = split(strip(lines[i+1+j]))
                name = replace(join(parts[3:end], " "), "\"" => "")
                push!(phys, string("dim", parts[1], " tag", parts[2], " ", name))
            end
            i += 1 + cnt
        elseif l == "\$Nodes"
            hdr = split(strip(lines[i+1]))
            nnodes = length(hdr) >= 2 ? parse(Int, hdr[2]) : parse(Int, hdr[1])
            i += 1
        elseif l == "\$Elements"
            hdr = split(strip(lines[i+1]))
            nelem = length(hdr) >= 2 ? parse(Int, hdr[2]) : parse(Int, hdr[1])
            i += 1
        else
            i += 1
        end
    end
    return nnodes, nelem, phys
end
nnodes, nelem, physnames = mesh_stats(MESHFILE)

# --- 2. Build both objectives from the SAME mesh -----------------------------
# Keep MESHDIR write-clean: give OLD its own copy in OUTDIR (cp is file IO, not
# compute, and we are inside the allocation anyway).
cp(MESHFILE, joinpath(OUTDIR, "mesh.msh"); force=true)
println("\n[2] building OLD objective ..."); flush(stdout)
old_obj = build_old_objective(OUTDIR; meshfile="mesh.msh")
n_old = old_obj.sim_y.np

println("[2] building NEW problem ..."); flush(stdout)
new_prob = build_new_problem(OUTDIR; meshfile=MESHFILE)
n_new = length(new_prob.p)

# --- 3. PARITY GATE ----------------------------------------------------------
parity_ok = (n_old == n_new)
open(joinpath(OUTDIR, "parity.txt"), "w") do io
    println(io, "first-epoch parity report  (quest tim-130116-82977)")
    println(io, "mesh: $MESHFILE")
    println(io, "mesh file size (MiB): ", round(filesize(MESHFILE)/2^20, digits=2))
    println(io, "mesh nodes:    ", nnodes)
    println(io, "mesh elements: ", nelem)
    println(io, "physical groups (", length(physnames), "):")
    for p in physnames; println(io, "  ", p); end
    println(io, "design DOFs (np)  OLD: ", n_old)
    println(io, "design DOFs (np)  NEW: ", n_new)
    println(io, "PARITY: ", parity_ok ? "PASS (n_old == n_new)" : "FAIL (n_old != n_new)")
end
println("\n[3] PARITY ", parity_ok ? "PASS" : "FAIL", "  (n_old=$n_old, n_new=$n_new)")
println("    mesh nodes=$nnodes elements=$nelem  tags=$(length(physnames))")
for p in physnames; println("      ", p); end
flush(stdout)

if !parity_ok
    error("PARITY FAIL: OLD np=$n_old != NEW np=$n_new — snapshots cannot align. " *
          "Inspect $(joinpath(OUTDIR, "parity.txt")) and STOP.")
end

# --- 4. ITER-1 DECOMPOSITION at uniform-0.5 ----------------------------------
n = n_old
p0 = fill(Float64(INIT_VALUE), n)
@assert p0 == fill(0.5, n) "init is not uniform-0.5 (trap-2)"

old_obj.control.β = BETA
new_prob.control.β = BETA

println("\n[4] flat-substrate normalization baselines (p=0, alpha=0) ..."); flush(stdout)
g_norm_old = e3.stepobjective(zeros(n), Float64[]; obj=old_obj)
g_norm_new = DEO.objective_and_gradient!(Float64[], zeros(n), new_prob)

println("[4] objective + gradient at uniform-0.5 ..."); flush(stdout)
grad_old = zeros(n); g_raw_old = e3.stepobjective(p0, grad_old; obj=old_obj)
grad_new = zeros(n); g_raw_new = DEO.objective_and_gradient!(grad_new, p0, new_prob)

g_old_n = g_raw_old / g_norm_old
g_new_n = g_raw_new / g_norm_new
nrm_old = norm(grad_old); nrm_new = norm(grad_new)
cosg = dot(grad_old, grad_new) / max(nrm_old * nrm_new, 1e-300)
relg = norm(grad_new - grad_old) / max(nrm_old, 1e-300)

open(joinpath(OUTDIR, "decomp.txt"), "w") do io
    @printf(io, "ITER-1 DECOMPOSITION (uniform-0.5, beta=%g, alpha=0)\n", BETA)
    @printf(io, "np = %d   mesh nodes=%d elements=%d\n", n, nnodes, nelem)
    @printf(io, "g_norm_old        = %.12e\n", g_norm_old)
    @printf(io, "g_norm_new        = %.12e\n", g_norm_new)
    @printf(io, "g_norm ratio n/o  = %.12e\n", g_norm_new / g_norm_old)
    @printf(io, "g_raw_old         = %.12e\n", g_raw_old)
    @printf(io, "g_raw_new         = %.12e\n", g_raw_new)
    @printf(io, "g_raw ratio n/o   = %.12e\n", g_raw_new / g_raw_old)
    @printf(io, "g_raw rel diff    = %.12e\n", abs(g_raw_new - g_raw_old)/max(abs(g_raw_old),1e-300))
    @printf(io, "g_norm'd old      = %.12e\n", g_old_n)
    @printf(io, "g_norm'd new      = %.12e\n", g_new_n)
    @printf(io, "g_norm'd ratio    = %.12e\n", g_new_n / g_old_n)
    @printf(io, "||grad_old||      = %.12e\n", nrm_old)
    @printf(io, "||grad_new||      = %.12e\n", nrm_new)
    @printf(io, "cos(grad_o,grad_n)= %.12e\n", cosg)
    @printf(io, "rel||dgrad||/||o||= %.12e\n", relg)
end

JLD2.save(joinpath(OUTDIR, "decomp.jld2"), Dict(
    "np" => n, "mesh_nodes" => nnodes, "mesh_elements" => nelem,
    "physnames" => physnames,
    "g_norm_old" => g_norm_old, "g_norm_new" => g_norm_new,
    "g_raw_old" => g_raw_old, "g_raw_new" => g_raw_new,
    "g_old_norm" => g_old_n, "g_new_norm" => g_new_n,
    "grad_old" => grad_old, "grad_new" => grad_new,
    "grad_norm_old" => nrm_old, "grad_norm_new" => nrm_new,
    "cos_grad" => cosg, "rel_grad" => relg,
    "p0" => p0,
))

println("\n===== ITER-1 DECOMPOSITION =====")
println(read(joinpath(OUTDIR, "decomp.txt"), String))
println("DONE. mesh + parity.txt + decomp.{txt,jld2} written to $OUTDIR and $MESHDIR")
