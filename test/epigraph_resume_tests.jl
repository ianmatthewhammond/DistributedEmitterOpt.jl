# Checkpoint and resume for the epigraph worst case (plan section 5, item 3), through
# optimize! on the small foundry FEM problem:
#   1. a mid-epoch checkpoint file actually appears (JLD2 with tau/T + MMA state of
#      dims (np+1, nsamples)) — the KS safe_checkpoint would throw inside its try on an
#      np+1 x and never write;
#   2. kill-and-resume gives a BIT-identical x = [p; tau], g_history and p_best to an
#      uninterrupted run;
#   3. refusals, each with a pinned message: KS <-> epigraph, a changed sample set, an MMA
#      state of the wrong (n, m), a tau that is not the state's committed -minf
#      (NEGATIVE CONTROL for the exactness check), a checkpoint without tau/T.
using Test, DistributedEmitterOpt
using LinearAlgebra, Random, JLD2
include(joinpath(@__DIR__, "gradient_common.jl"))
const DEO = DistributedEmitterOpt
const MB = DEO.MmaccsaBackend

prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                          surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                          foundry_projection_mode=:current)
sim0 = DEO.default_sim(prob.sim)
z_lo, z_hi = DEO.getdesignz(sim0.labels, sim0.Ω)
const Z0 = (z_lo + z_hi) / 2
const SAMPLES = [(0.0, Z0), (tand(10.0), Z0)]
const NS = length(SAMPLES)
np = length(prob.p)
g_norm = DEO.flat_substrate_norm(prob)
p0 = clamp!(fill(0.5, np) .+ 0.2 .* randn(MersenneTwister(20260921), np), 0.0, 1.0)

const NEV = 8          # evals in the epoch
const EVERY = 2        # backup_every
const KILL = 6         # simulated kill after this many evals

common(; kw...) = (; max_iter=NEV, β_schedule=[8.0], backend=:standalone_ccsaq,
                   rho_init=1.0 / g_norm, robust_samples=SAMPLES, robust_mode=:epigraph, kw...)

function fresh!()
    prob.p .= p0
    empty!(prob.g_history); prob.iteration = 0; prob.g_best = -Inf; fill!(prob.p_best, 0.0)
end

# pinned-message refusal helpers
msg(f) = try f(); nothing catch err; sprint(showerror, err) end
refused_robust(f) = (m = msg(f); m !== nothing && occursin("robust sidewall configuration", m))

dir = mktempdir()

@testset "Epigraph checkpoint and resume" begin

    # ── A: uninterrupted reference ────────────────────────────────────────────────
    fresh!()
    ckA = joinpath(dir, "A.jld2")
    optimize!(prob; common(backup=true, backup_every=EVERY, backup_path=ckA)...)
    stA = DEO._EPIGRAPH_LAST_STATS[]
    xA, histA, pA = copy(stA.x_final), copy(prob.g_history), copy(prob.p)
    pbestA, gbestA = copy(prob.p_best), prob.g_best
    @test length(histA) == NEV && stA.checkpoint_failures == 0

    # ── B: killed after KILL evals ────────────────────────────────────────────────
    fresh!()
    ckB = joinpath(dir, "B.jld2")
    DEO._EPIGRAPH_ABORT_AFTER[] = KILL
    killed = try
        optimize!(prob; common(backup=true, backup_every=EVERY, backup_path=ckB)...)
        false
    catch err
        err isa DEO.EpigraphTestAbort
    finally
        DEO._EPIGRAPH_ABORT_AFTER[] = typemax(Int)
    end
    @test killed

    @testset "a mid-epoch checkpoint was actually written" begin
        @test isfile(ckB)
        d = JLD2.load(ckB)
        @test d["robust_mode"] == "epigraph"
        @test length(d["p"]) == np                       # p only, not [p; tau]
        @test 0 < d["epoch_evals"] < NEV
        @test haskey(d, "epigraph_tau") && haskey(d, "epigraph_T")
        @test isfile(d["mma_state_path"])
        @test MB.state_dims(d["mma_state_path"]) == (np + 1, NS)
        @test -MB.state_header(d["mma_state_path"]).minf == d["epigraph_tau"]
        @test length(d["g_history"]) == d["epoch_evals"]
        println("  checkpoint at eval $(d["epoch_evals"]): tau=$(d["epigraph_tau"]) T=$(d["epigraph_T"])")
    end

    # private copies for the refusal checks (each resume below reads its own copy)
    names = ("resume", "ks", "samples", "nm", "tau", "notau")
    cks = Dict(n => joinpath(dir, "B_$n.jld2") for n in names)
    foreach(n -> cp(ckB, cks[n]), names)

    @testset "kill-and-resume is bit-identical to the uninterrupted run" begin
        optimize!(prob; common(backup=false, resume_from=cks["resume"])...)
        stB = DEO._EPIGRAPH_LAST_STATS[]
        @test stB.x_final == xA                   # x = [p; tau], bitwise
        @test prob.g_history == histA
        @test prob.p == pA
        @test prob.p_best == pbestA && prob.g_best == gbestA
        @test stB.T == stA.T
        # the resumed epoch does not re-seed (no seed call): calls = (1 + NS) per eval, the
        # first eval after resume is a new point (a miss), every constraint call hits
        @test stB.hits + stB.misses == stB.nevals * (1 + NS)
        @test stB.solves == NS * stB.misses
        @test 1 <= stB.misses <= stB.nevals
        @test stB.hits >= NS * stB.nevals
    end

    @testset "refusals" begin
        # KS <-> epigraph
        @test refused_robust(() -> optimize!(prob; common(backup=false, resume_from=cks["ks"],
                                                          robust_mode=:ks)...))
        # a changed sample set
        @test refused_robust(() -> optimize!(prob; common(backup=false, resume_from=cks["samples"],
            robust_samples=[(0.0, Z0), (tand(5.0), Z0)])...))

        # an MMA state of the wrong (n, m): point the checkpoint at a real n=3, m=1 state
        st = MB.MmaState(); xs = [0.9, 0.1, 0.5]
        MB.minimize!(:ccsaq, (x, g) -> (isempty(g) || (g .= 2 .* x); sum(abs2, x)), xs;
            lb=zeros(3), ub=ones(3), maxeval=3, state=st,
            constraints=[((x, g) -> (isempty(g) || (g .= 1.0); sum(x) - 2.0), 1e-8)])
        foreign = joinpath(dir, "foreign.mma_state"); MB.save_state(foreign, st)
        d = JLD2.load(cks["nm"]); d["mma_state_path"] = foreign; JLD2.save(cks["nm"], d)
        m = msg(() -> optimize!(prob; common(backup=false, resume_from=cks["nm"])...))
        @test m !== nothing && occursin("was written for n=3, m=1", m) && occursin("n=$(np + 1), m=$NS", m)

        # NEGATIVE CONTROL for bit-exactness: tau off by one ulp is refused
        d = JLD2.load(cks["tau"]); d["epigraph_tau"] = nextfloat(d["epigraph_tau"]); JLD2.save(cks["tau"], d)
        m = msg(() -> optimize!(prob; common(backup=false, resume_from=cks["tau"])...))
        @test m !== nothing && occursin("not the MMA state's committed -minf", m)

        # a checkpoint without tau/T cannot rebuild x
        d = JLD2.load(cks["notau"]); delete!(d, "epigraph_tau"); JLD2.save(cks["notau"], d)
        m = msg(() -> optimize!(prob; common(backup=false, resume_from=cks["notau"])...))
        @test m !== nothing && occursin("lacks epigraph_T/epigraph_tau", m)
    end
end
println("EPIGRAPH-RESUME-TESTS-DONE")
