# Epigraph worst case over sidewall samples (EpigraphSidewall.jl + the standalone epigraph
# epoch in Optimizer.jl). Every testset runs the real FEM chain on the small foundry test
# problem, on the production-shaped path (:current foundry mode, :tv_void surface objective).
#
#   1. the cache is keyed on the VALUES of p (1-ulp change misses, a copy hits, in-place
#      mutation misses), with a cache-off negative control;
#   2. epigraph constraint gradients vs central FD, both samples and tau;
#   3. the linewidth wrapper passes p only and leaves grad[tau] == 0 (fake and real glc);
#   4. cache accounting through optimize!: solves = nsamples x misses <= nsamples x nevals,
#      every constraint call hits; NEGATIVE CONTROL: the uncached count (cache off) is
#      nsamples + nsamples(1+nsamples) x nevals, on an identical trajectory;
#   5. the constrained path (linewidth constraints on, m = nsamples + 2) runs from a feasible
#      start and ends feasible, and a constraint_scale that is not 1/b1 is refused by name;
#   6. a real negative CCSA return (strongly violated linewidth start) throws, pinned message.
using Test, DistributedEmitterOpt
using LinearAlgebra, Random
include(joinpath(@__DIR__, "gradient_common.jl"))
const DEO = DistributedEmitterOpt

prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                          surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                          foundry_projection_mode=:current)
sim0 = DEO.default_sim(prob.sim)
z_lo, z_hi = DEO.getdesignz(sim0.labels, sim0.Ω)
const Z0 = (z_lo + z_hi) / 2                     # mid-height pivot (mid-height reference)
const SAMPLES = [(0.0, Z0), (tand(10.0), Z0)]
const NS = length(SAMPLES)
np = length(prob.p)
g_norm = DEO.flat_substrate_norm(prob)
p0 = clamp!(fill(0.5, np) .+ 0.2 .* randn(MersenneTwister(20260921), np), 0.0, 1.0)
println("np=$np  z=[$z_lo, $z_hi]  Z0=$Z0  g_norm=$g_norm")

@testset "Epigraph worst case" begin

@testset "cache keyed on p VALUES" begin
    c = DEO.EpigraphCache(np, NS)
    x = [copy(p0); 1.0]
    gs1, _, hit = DEO.epigraph_samples!(c, x, prob, SAMPLES)
    @test !hit && c.solves == NS
    gs1 = copy(gs1)
    @test gs1[1] != gs1[2]                      # the samples are genuinely different objectives
    @test DEO.epigraph_samples!(c, x, prob, SAMPLES)[3]            # same buffer, same values
    @test DEO.epigraph_samples!(c, copy(x), prob, SAMPLES)[3]      # different object, same values
    x[end] = 5.0
    @test DEO.epigraph_samples!(c, x, prob, SAMPLES)[3]            # tau is not part of the key
    @test c.solves == NS
    # IN-PLACE 1-ulp change of one entry of the SAME buffer: an identity/pointer-keyed cache
    # would hit here (this is exactly what the C side does to its x buffer).
    x[3] = nextfloat(x[3])
    @test !DEO.epigraph_samples!(c, x, prob, SAMPLES)[3]
    @test c.solves == 2NS
    # the key is a copy: mutating the caller's buffer after a solve must not move the key
    x[4] += 0.25
    @test c.key[4] == p0[4]
    # NEGATIVE CONTROL: cache off -> the identical point is recomputed
    DEO._EPIGRAPH_CACHE_ENABLED[] = false
    try
        x[4] -= 0.25                                              # back to the cached values
        @test !DEO.epigraph_samples!(c, x, prob, SAMPLES)[3]
        @test c.solves == 3NS
    finally
        DEO._EPIGRAPH_CACHE_ENABLED[] = true
    end
end

@testset "constraint gradients vs central FD (both samples, and tau)" begin
    c = DEO.EpigraphCache(np, NS)
    T, τ0 = DEO.epigraph_seed!(c, copy(p0), prob, SAMPLES, g_norm)
    @test T == minimum(c.gs) / g_norm && 0 < τ0 < 1
    x = [copy(p0); τ0]
    d = randn(MersenneTwister(7), np); d ./= norm(d)
    cons = [DEO.epigraph_constraint(c, i, prob, SAMPLES, g_norm, T) for i in 1:NS]
    grads = [zeros(np + 1) for _ in 1:NS]
    vals = [cons[i](x, grads[i]) for i in 1:NS]
    @test all(vals .< 0)                                   # strictly feasible start
    @test vals ≈ τ0 .- c.gs ./ (g_norm * T)
    fdv = zeros(NS)
    for i in 1:NS
        f(t) = cons[i]([p0 .+ t .* d; τ0], Float64[])
        cd(h) = (f(h) - f(-h)) / (2h)
        δ = 1e-4
        fd = (4cd(δ / 2) - cd(δ)) / 3                      # Richardson
        fdv[i] = fd
        adj = dot(grads[i][1:np], d)
        println("  constraint $i: adj=$adj fd=$fd rel=$(abs(adj - fd) / abs(fd))")
        @test isapprox(adj, fd; rtol=1e-4)
        @test grads[i][np+1] == 1.0
        # tau: the constraint is exactly linear in tau
        @test cons[i]([p0; τ0 + 0.125], Float64[]) - cons[i]([p0; τ0 - 0.125], Float64[]) ≈ 0.25 atol=1e-12
    end
    # NEGATIVE CONTROL: the two samples' gradients are not interchangeable
    @test !isapprox(dot(grads[1][1:np], d), fdv[2]; rtol=1e-3)
end

@testset "linewidth wrapper: p only, grad[tau] == 0" begin
    seen = Ref(0)
    fake = (p, g) -> begin
        seen[] = length(p)
        isempty(g) || (g .= 1:length(p))
        2.0
    end
    w = DEO.epigraph_linewidth(fake, np, 3.0)
    x = [copy(p0); 0.7]
    g = fill(NaN, np + 1)                                  # a reused C buffer holds junk
    @test w(x, g) == 6.0
    @test seen[] == np
    @test g[1:np] == 3.0 .* (1:np)
    @test g[np+1] === 0.0
    @test w(x, Float64[]) == 6.0                           # value-only call
    # the real constraint, through the wrapper vs direct
    cs = 1 / prob.control.b1
    gd = zeros(np)
    vd = glc_solid(copy(p0), gd; sim=sim0, control=prob.control)
    ww = DEO.epigraph_linewidth((p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), np, cs)
    gw = fill(NaN, np + 1)
    vw = ww(x, gw)
    @test vw == cs * vd
    @test gw[1:np] == cs .* gd
    @test gw[np+1] === 0.0
end

@testset "cache accounting through optimize!, vs the uncached count" begin
    nev = 5
    run() = begin
        prob.p .= p0
        optimize!(prob; max_iter=nev, β_schedule=[8.0], backend=:standalone_ccsaq,
                  rho_init=1.0 / g_norm, backup=false,
                  robust_samples=SAMPLES, robust_mode=:epigraph)
        (DEO._EPIGRAPH_LAST_STATS[], copy(prob.g_history), copy(prob.p))
    end
    st, hist, pfin = run()
    @test st.nevals == nev
    # Cache accounting. Calls: 1 seed + (1 objective + NS constraints) per eval. Every
    # constraint call hits (it follows the objective at the same x), and so does the first
    # objective call (the seed filled the cache). An objective call at a trial point that
    # differs from the last one only in tau ALSO hits — CCSA does produce such points
    # (measured: 4 resumed evals needed 2 solve sets) — so solves <= NS x nevals, with
    # equality only when every trial point moves p.
    @test st.hits + st.misses == 1 + st.nevals * (1 + NS)
    @test st.solves == NS * st.misses
    @test st.misses <= st.nevals
    @test st.hits >= NS * st.nevals + 1
    # g_history is the TRUE worst case at every evaluated point
    @test hist[end-nev+1:end] == st.worst
    @test st.worst == [minimum(g) / g_norm for g in st.gs]
    @test st.checkpoint_failures == 0
    @test pfin != p0                            # it moved

    # NEGATIVE CONTROL: cache off. The counter must see the uncached count, and the
    # trajectory must be the same (the cache returns what a fresh solve returns).
    DEO._EPIGRAPH_CACHE_ENABLED[] = false
    st0, hist0, pfin0 = try run() finally DEO._EPIGRAPH_CACHE_ENABLED[] = true end
    @test st0.nevals == nev
    @test st0.solves == NS + NS * (1 + NS) * nev   # seed + (objective + NS constraints) per eval
    @test st0.solves != st.solves
    @test isapprox(hist0[end-nev+1:end], hist[end-nev+1:end]; rtol=1e-12)
    @test isapprox(pfin0, pfin; rtol=1e-12)
    println("  cached solves=$(st.solves)  uncached=$(st0.solves)  bitwise-same trajectory: $(hist0 == hist && pfin0 == pfin)")
end

@testset "linewidth-constrained path: linewidth constraints on (m = nsamples + 2)" begin
    b1_saved = prob.control.b1
    Ms = glc_solid(copy(p0), Float64[]; sim=sim0, control=prob.control) + b1_saved
    Mv = glc_void(copy(p0), Float64[]; sim=sim0, control=prob.control) + b1_saved
    try
        prob.p .= p0
        # constraint_scale must be 1/b1, else the normalized tolerance means something else
        err = try
            optimize!(prob; max_iter=2, β_schedule=[8.0], backend=:standalone_ccsaq,
                      rho_init=1.0 / g_norm, backup=false, use_constraints=true,
                      constraint_scale=1.0, robust_samples=SAMPLES, robust_mode=:epigraph)
            nothing
        catch e; e end
        @test err isa ArgumentError && occursin("constraint_scale == 1/b1", sprint(showerror, err))

        # plumbing: eps chosen so the start is feasible (M/eps - 1 = -0.5); 4 constraints,
        # the cache still does NS solves per eval, and the linewidth constraints hold at the end
        eps = 2max(Ms, Mv)
        prob.control.b1 = eps
        prob.p .= p0
        optimize!(prob; max_iter=3, β_schedule=[8.0], backend=:standalone_ccsaq,
                  rho_init=1.0 / g_norm, backup=false, use_constraints=true,
                  constraint_scale=1 / eps, robust_samples=SAMPLES, robust_mode=:epigraph)
        st = DEO._EPIGRAPH_LAST_STATS[]
        @test st.m == NS + 2 && st.nevals == 3
        @test st.solves == NS * st.misses && st.misses <= st.nevals
        @test glc_solid(copy(prob.p), Float64[]; sim=sim0, control=prob.control) / eps <= 1e-3
        @test glc_void(copy(prob.p), Float64[]; sim=sim0, control=prob.control) / eps <= 1e-3
    finally
        prob.control.b1 = b1_saved
    end
end

@testset "a negative CCSA return code throws (the backend does not)" begin
    # A REAL failure, not a simulated one: linewidth constraints from the grey start at the
    # default b1 = 1e-8 are violated by a large normalized amount;
    # CCSA's dual line search then returns ret = -1 after the first evaluation. The KS path
    # ignores ret; the epigraph path must refuse to report a design. The failure comes from
    # the L-BFGS dual solver's line search, so this test asks for it explicitly.
    viol = (glc_solid(copy(p0), Float64[]; sim=sim0, control=prob.control)) / prob.control.b1
    @test viol > 1e5                     # the scenario is the one this test is about
    prob.p .= p0
    m = try
        optimize!(prob; max_iter=3, β_schedule=[8.0], backend=:standalone_ccsaq,
                  rho_init=1.0 / g_norm, backup=false, use_constraints=true,
                  constraint_scale=1 / prob.control.b1, robust_samples=SAMPLES, robust_mode=:epigraph,
                  dual_algorithm=:lbfgs)
        nothing
    catch e; sprint(showerror, e) end
    @test m !== nothing && occursin("returned ret=-", m) && occursin("refusing to report a design", m)
end

@testset "the default (MMA) dual gets past the infeasible start that stops the L-BFGS dual" begin
    # Same scenario as the test above: strongly violated linewidth constraints at the grey
    # start. There the L-BFGS dual returns ret = -1 after one evaluation (that test is the
    # control). The default MMA dual must run the whole budget and move the committed
    # iterate towards feasibility. Four calls cover both the legacy density rule and
    # the design-region grid sizing rule; retain the same tenfold reduction check.
    vs0 = glc_solid(copy(p0), Float64[]; sim=sim0, control=prob.control) / prob.control.b1
    vv0 = glc_void(copy(p0), Float64[]; sim=sim0, control=prob.control) / prob.control.b1
    @test min(vs0, vv0) > 1e5
    prob.p .= p0
    m = try
        optimize!(prob; max_iter=4, β_schedule=[8.0], backend=:standalone_ccsaq,
                  rho_init=1.0 / g_norm, backup=false, use_constraints=true,
                  constraint_scale=1 / prob.control.b1, robust_samples=SAMPLES, robust_mode=:epigraph)
        nothing
    catch e; sprint(showerror, e) end
    @test m === nothing
    st = DistributedEmitterOpt._EPIGRAPH_LAST_STATS[]
    @test st.nevals == 4
    x = st.x_final[1:np]
    vs = glc_solid(copy(x), Float64[]; sim=sim0, control=prob.control) / prob.control.b1
    vv = glc_void(copy(x), Float64[]; sim=sim0, control=prob.control) / prob.control.b1
    @test vs < vs0 / 10 && vv < vv0 / 10
end

@testset "argument guards" begin
    e1 = try optimize!(prob; max_iter=1, β_schedule=[8.0], backend=:nlopt,
                       robust_samples=SAMPLES, robust_mode=:epigraph); nothing catch e; e end
    @test e1 isa ArgumentError && occursin("standalone backend", sprint(showerror, e1))
    e2 = try optimize!(prob; max_iter=1, β_schedule=[8.0], backend=:standalone_ccsaq,
                       robust_mode=:epigraph); nothing catch e; e end
    @test e2 isa ArgumentError && occursin("requires robust_samples", sprint(showerror, e2))
    e3 = try optimize!(prob; max_iter=1, β_schedule=[8.0], backend=:standalone_ccsaq,
                       robust_samples=SAMPLES, robust_mode=:softmax); nothing catch e; e end
    @test e3 isa ArgumentError && occursin("robust_mode must be", sprint(showerror, e3))
end

end
println("EPIGRAPH-TESTS-DONE")
