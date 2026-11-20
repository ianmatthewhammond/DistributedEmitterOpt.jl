# The 3D Helmholtz filter caches the factorization of its matrix, which depends on the three
# filter radii, the filtered FE space and the two integration measures. Changing
# `control.R_filter` on a problem (as `fabrication_sweep` does, or as `init_random!` does with
# its fixed 20 nm radius) must give the same filtered design, filter adjoint and objective as a
# cache that has never been used.
#
# Reference values ("cold") come from a new SolverCache on the same simulation, or from the same
# problem with a new SolverCache installed as its filter cache, so the warm and cold paths
# assemble and factorize the same matrix and should agree to round-off. Each warm cache is
# primed explicitly at the radius the case needs.
#
# Assertions marked "regression" check a result or a cache state that a factor built for
# another operator would change. The others check that the comparisons can detect a
# difference, or that the factor is still reused when the operator is unchanged.
using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

isdefined(@__MODULE__, :build_test_problem) || include("gradient_common.jl")

const HFR = DistributedEmitterOpt
const HFR_RTOL = 1e-12

hfr_relerr(a, b) = norm(a .- b) / norm(b)

"""Filter `p` at radii `R` on a new cache (no factor carried over from anything else)."""
function hfr_cold_filter(p, sim, control, R)
    ctrl = deepcopy(control)
    ctrl.R_filter = R
    return filter_helmholtz!(p, HFR.SolverCache(UmfpackSolver()), sim, ctrl)
end

"""Filter adjoint of `v` at radii `R` on a new cache (forward call first, to factorize)."""
function hfr_cold_adjoint(v, np, sim, control, R)
    ctrl = deepcopy(control)
    ctrl.R_filter = R
    cache = HFR.SolverCache(UmfpackSolver())
    filter_helmholtz!(zeros(np), cache, sim, ctrl)
    return filter_helmholtz_adjoint!(v, cache, sim, ctrl)
end

"""A new, empty cache on the problem's solver backend."""
hfr_new_cache(prob) = HFR.SolverCache(HFR.default_pool(prob.pool).solver)

"""
Run `f()` with `c` installed as the problem's filter cache, then put the previous cache back.
Every problem-level path (initialization, evaluation, sweeps) filters with that cache.
"""
function hfr_with_filter_cache(f, prob, c)
    pool = HFR.default_pool(prob.pool)
    previous = pool.filter_cache
    pool.filter_cache = c
    try
        return f()
    finally
        pool.filter_cache = previous
    end
end

"""Factor handle of the test backend below; `live` is false once its release was attempted."""
mutable struct HFRHandle
    F::Any
    live::Bool
end

function hfr_live_factor(h::HFRHandle)
    h.live || error("HFRReleaseFails: solve with a released factor")
    return h.F
end

"""
Test backend: UMFPACK factors, but releasing a factor marks the handle released and then
throws, as a native cleanup failure would. A solve with a released handle throws.
"""
struct HFRReleaseFails <: HFR.AbstractSolver end
HFR.filter_lu!(::HFRReleaseFails, A) = HFRHandle(HFR.filter_lu!(UmfpackSolver(), A), true)
HFR.solve!(::HFRReleaseFails, h::HFRHandle, b::Vector) = HFR.solve!(UmfpackSolver(), hfr_live_factor(h), b)
HFR.solve_adjoint!(::HFRReleaseFails, h::HFRHandle, b::Vector) =
    HFR.solve_adjoint!(UmfpackSolver(), hfr_live_factor(h), b)
function HFR.release_factor!(::HFRReleaseFails, h::HFRHandle)
    h.live = false
    error("HFRReleaseFails: release failed")
end

@testset "Helmholtz filter factor follows the radius" begin
    prob = build_test_problem()          # 3D DOF mode; init_random! factorizes at R = 20
    sim = HFR.default_sim(prob.sim)
    cache = HFR.default_pool(prob.pool).filter_cache
    ctrl = prob.control
    np = length(prob.p)
    p = rand(MersenneTwister(7), np)     # fixed nonuniform design
    R1 = (10.0, 10.0, 10.0)
    R2 = (30.0, 30.0, 30.0)
    R0 = (20.0, 20.0, 20.0)

    fwd = Dict(R => hfr_cold_filter(p, sim, ctrl, R) for R in (R0, R1, R2))
    nf = length(fwd[R1])
    v = randn(MersenneTwister(11), nf)   # fixed adjoint seed on the filtered space
    adj = Dict(R => hfr_cold_adjoint(v, np, sim, ctrl, R) for R in (R1, R2))

    # The radii must matter on this mesh, or the comparisons below prove nothing.
    @test hfr_relerr(fwd[R2], fwd[R1]) > 1e-3
    @test hfr_relerr(adj[R2], adj[R1]) > 1e-3

    @testset "forward at R1, R2, R1" begin
        # The cache starts with the 20 nm factor that init_random! built in build_test_problem.
        for (step, R) in enumerate((R1, R2, R1))
            ctrl.R_filter = R
            @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), fwd[R]) < HFR_RTOL     # regression: factor of another radius reused
        end
    end

    @testset "adjoint at R1, R2, R1" begin
        # Adjoint calls alone, with no forward call at the new radius in between.
        for (step, R) in enumerate((R1, R2, R1))
            ctrl.R_filter = R
            @test hfr_relerr(filter_helmholtz_adjoint!(v, cache, sim, ctrl), adj[R]) < HFR_RTOL   # regression: factor of another radius reused
        end
        # On an empty cache the adjoint factorizes on its own (ctrl is at R1 here).
        @test hfr_relerr(filter_helmholtz_adjoint!(v, HFR.SolverCache(UmfpackSolver()), sim, ctrl), adj[R1]) < HFR_RTOL   # regression: adjoint on an empty cache
    end

    @testset "factor reused at a fixed radius" begin
        # Reuse is kept when nothing changes.
        ctrl.R_filter = R2
        a = filter_helmholtz!(p, cache, sim, ctrl)
        F = cache.F_factor
        b = filter_helmholtz!(p, cache, sim, ctrl)
        @test cache.F_factor === F
        @test b == a
        filter_helmholtz_adjoint!(v, cache, sim, ctrl)
        @test cache.F_factor === F
    end

    @testset "anisotropic radius change" begin
        Ra = (20.0, 20.0, 10.0)
        ref = hfr_cold_filter(p, sim, ctrl, Ra)
        @test hfr_relerr(ref, fwd[R0]) > 1e-6      # the z radius matters
        ctrl.R_filter = R0
        @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), fwd[R0]) < HFR_RTOL   # starting radius
        ctrl.R_filter = Ra
        @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), ref) < HFR_RTOL       # regression: isotropic factor reused after a z-radius change
    end

    @testset "initialization radius vs optimization radius" begin
        # init_random! filters at its fixed 20 nm radius whatever the optimization radius is.
        # Each case starts from its own cache, installed as the problem's filter cache.
        ctrl.R_filter = R1

        # (a) Initialization on an empty cache, then the filter at R1 on the same cache.
        c_a = hfr_new_cache(prob)
        hfr_with_filter_cache(() -> init_random!(prob; seed=3), prob, c_a)
        @test hfr_relerr(filter_helmholtz!(p, c_a, sim, ctrl), fwd[R1]) < HFR_RTOL     # regression: 20 nm factor reused at R1

        # (b) A cache primed at R1 (the configured radius, held fixed), then initialization,
        # compared with initialization on an empty cache.
        c_b = hfr_new_cache(prob)
        filter_helmholtz!(p, c_b, sim, ctrl)                # c_b holds the R1 factor
        @test HFR.has_filter_factor(c_b)
        p_warm = copy(hfr_with_filter_cache(() -> init_random!(prob; seed=3), prob, c_b).p)
        p_cold = copy(hfr_with_filter_cache(() -> init_random!(prob; seed=3), prob, hfr_new_cache(prob)).p)
        @test hfr_relerr(p_warm, p_cold) < HFR_RTOL                                   # regression: initialization filtered with the R1 factor

        # (c) After (b), c_b filters at R1 as a cold cache does.
        @test hfr_relerr(filter_helmholtz!(p, c_b, sim, ctrl), fwd[R1]) < HFR_RTOL     # regression: 20 nm factor reused at R1
    end

    @testset "objective and gradient after a radius change" begin
        # The path a radius sweep takes: evaluate at R1, then at R2 on the same problem, compared
        # with an evaluation at R2 whose filter cache starts empty.
        c = hfr_new_cache(prob)
        ctrl.R_filter = R1
        g_warm, ∇g_warm = hfr_with_filter_cache(prob, c) do
            HFR.evaluate(prob, p)                           # c now holds the R1 factor
            ctrl.R_filter = R2
            HFR.evaluate(prob, p)
        end
        g_cold, ∇g_cold = hfr_with_filter_cache(() -> HFR.evaluate(prob, p), prob, hfr_new_cache(prob))
        @test isapprox(g_warm, g_cold; rtol=1e-10)                                   # regression: R1 factor reused at R2
        @test hfr_relerr(∇g_warm, ∇g_cold) < 1e-10                                   # regression: R1 factor reused at R2
    end

    @testset "length-scale constraints after a radius change (glc_solid_fe, glc_void_fe)" begin
        # The path optimize! and linewidth_feasibility take in 3D DOF mode: the constraints filter
        # with the problem's filter cache. Here that cache is a new one, primed at R1.
        c = hfr_new_cache(prob)
        ctrl.R_filter = R1
        obj = (; sim, control=ctrl, cache_pump=c)
        glc_solid_fe(p, Float64[], obj)                     # c now holds the R1 factor
        @test HFR.has_filter_factor(c)
        ctrl.R_filter = R2
        gs, gv = zeros(np), zeros(np)
        Ms, Mv = glc_solid_fe(p, gs, obj), glc_void_fe(p, gv, obj)
        cold = (; sim, control=ctrl, cache_pump=HFR.SolverCache(UmfpackSolver()))
        gs_c, gv_c = zeros(np), zeros(np)
        Ms_c, Mv_c = glc_solid_fe(p, gs_c, cold), glc_void_fe(p, gv_c, cold)
        @test isapprox(Ms, Ms_c; rtol=1e-10)                # regression: R1 factor reused at R2
        @test isapprox(Mv, Mv_c; rtol=1e-10)                # regression: R1 factor reused at R2
        @test hfr_relerr(gs, gs_c) < 1e-10                  # regression: R1 factor reused at R2
        @test hfr_relerr(gv, gv_c) < 1e-10                  # regression: R1 factor reused at R2
    end

    @testset "fabrication_sweep evaluates each radius with its own filter" begin
        # The sweep changes control.R_filter on the problem between evaluations (10 and 30 nm
        # here). Its filter cache is a new one, primed at R1 = 10 nm before the sweep.
        c = hfr_new_cache(prob)
        ctrl.R_filter = R1
        Rs, gys = hfr_with_filter_cache(prob, c) do
            HFR.evaluate(prob, p)                           # c now holds the R1 factor
            HFR.Analysis.fabrication_sweep(prob, p; center_R=20.0, range_R=10.0, points=2,
                                           root=mktempdir(), save_plot=false)
        end
        @test prob.control.R_filter == R1                   # restored after the sweep
        @test !isapprox(gys[1], gys[2]; rtol=1e-6)          # regression: both points scored with one factor
        for (R, g) in zip(Rs, gys)
            ctrl.R_filter = (R, R, R)
            g_cold, _ = hfr_with_filter_cache(() -> HFR.evaluate(prob, p), prob, hfr_new_cache(prob))
            @test isapprox(g, g_cold; rtol=1e-10)           # regression: sweep point scored with another radius's factor
        end
        ctrl.R_filter = R1
    end

    @testset "factor rebuilt for another filtered space" begin
        # The key holds the filtered FE space as well as the radii. A second problem built the same
        # way has its own space object; at the same radius the cache must not hand it the factor
        # built on the first space. The second problem also has its own measures, so this case is
        # caught by the space check or the measure checks; the testset "space check alone" below
        # isolates the space check.
        prob2 = build_test_problem()
        sim2 = HFR.default_sim(prob2.sim)
        p2 = rand(MersenneTwister(13), length(prob2.p))
        ctrl.R_filter = R1
        filter_helmholtz!(p, cache, sim, ctrl)
        F = cache.F_factor
        @test cache.F_key.space === sim.Pf
        out2 = filter_helmholtz!(p2, cache, sim2, ctrl)
        @test cache.F_factor !== F                          # regression: factor of another space reused
        @test cache.F_key.space === sim2.Pf
        @test hfr_relerr(out2, hfr_cold_filter(p2, sim2, ctrl, R1)) < HFR_RTOL
        @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), fwd[R1]) < HFR_RTOL
        @test cache.F_key.space === sim.Pf
    end

    @testset "old checkpoint: design-only import, then the current radius" begin
        # A checkpoint as main writes it holds no filter radius, so one written by a run whose cache
        # held a factor for other radii cannot be told apart from any other; exact resume of such
        # files is not refused by this change. The migration path the PR names is the design-only
        # import followed by a fresh run, which filters with the factor for the current radius.
        path = joinpath(mktempdir(), "checkpoint_main_keys.jld2")
        p_ck = rand(MersenneTwister(17), np)
        HFR.JLD2.save(path, Dict{String,Any}("p" => p_ck, "g" => 1.5, "p_best" => p_ck, "g_best" => 1.5,
                                            "g_history" => [1.0, 1.5],
                                            "dct_filter_revision" => HFR._checkpoint_filter_revision()))
        ctrl.R_filter = R1
        filter_helmholtz!(p, cache, sim, ctrl)              # cache holds the R1 factor
        HFR.load_checkpoint_design!(prob, path)
        @test prob.p == p_ck
        @test isempty(prob.g_history) && prob.iteration == 0 && prob.g_best == -Inf
        ctrl.R_filter = R2
        @test hfr_relerr(filter_helmholtz!(prob.p, cache, sim, ctrl), hfr_cold_filter(p_ck, sim, ctrl, R2)) < HFR_RTOL   # regression: R1 factor reused at R2
    end

    @testset "factor rebuilt for a new space object on the same measures, space check alone" begin
        # The two-problem case above also changes the measures, so it no longer isolates the space
        # check. Here only sim.Pf is replaced (same model, same DOFs, same measures).
        old_space = sim.Pf
        try
            ctrl.R_filter = R1
            filter_helmholtz!(p, cache, sim, ctrl)
            F = cache.F_factor
            sim.Pf = HFR.TestFESpace(sim.model, HFR.ReferenceFE(HFR.lagrangian, Float64, 1);
                                     vector_type=Vector{Float64})
            out = filter_helmholtz!(p, cache, sim, ctrl)
            @test cache.F_factor !== F                          # regression: factor of the replaced space reused
            @test cache.F_key.space === sim.Pf
            @test hfr_relerr(out, fwd[R1]) < HFR_RTOL            # same operator numerically
        finally
            sim.Pf = old_space
        end
    end

    @testset "factor rebuilt when an integration measure is replaced" begin
        # The filter matrix integrates its gradient term over sim.dΩ_design and its mass term over
        # sim.dΩ. Assigning a new measure on a live simulation must not reuse the old factor.
        old_design, old_full = sim.dΩ_design, sim.dΩ
        try
            ctrl.R_filter = R1
            filter_helmholtz!(p, cache, sim, ctrl)              # factor for R1 with the original measures
            sim.dΩ_design = sim.dΩ                              # gradient term over the whole domain
            ref = hfr_cold_filter(p, sim, ctrl, R1)
            @test hfr_relerr(ref, fwd[R1]) > 1e-3               # the measure matters on this mesh
            @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), ref) < HFR_RTOL     # regression: stale factor reused after a measure change
            @test cache.F_key.dΩ_design === sim.dΩ_design

            # A new mass-term measure (another quadrature degree; exact for P1 mass either way)
            F = cache.F_factor
            sim.dΩ = HFR.Measure(sim.Ω, 2)
            out = filter_helmholtz!(p, cache, sim, ctrl)
            @test cache.F_factor !== F                          # regression: same factor reused
            @test cache.F_key.dΩ === sim.dΩ
            @test hfr_relerr(out, hfr_cold_filter(p, sim, ctrl, R1)) < HFR_RTOL
        finally
            sim.dΩ_design, sim.dΩ = old_design, old_full
        end
        # Back on the original measures the factor is rebuilt and equals the cold filter again
        @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), fwd[R1]) < HFR_RTOL
    end

    @testset "fabrication_sweep restores radius, score and gradient on both exits" begin
        # Populated state: the problem has evaluated at R0 (20 nm, not a sweep point) and carries a
        # history and an iteration count. The sweep below visits 10 and 30 nm.
        hist0, it0 = copy(prob.g_history), prob.iteration
        try
            ctrl.R_filter = R0
            HFR.evaluate(prob, p)
            push!(prob.g_history, 0.25, 0.5)
            prob.iteration = 7
            g0, ∇g0, hist1 = prob.g, copy(prob.∇g), copy(prob.g_history)
            @test !isempty(prob.∇g)

            # Normal exit
            Rs, gys = HFR.Analysis.fabrication_sweep(prob, p; center_R=20.0, range_R=10.0, points=2,
                                                     root=mktempdir(), save_plot=false)
            @test !isapprox(gys[end], g0; rtol=1e-6)            # the last point's score is not the caller's
            @test prob.control.R_filter == R0
            @test prob.g == g0                                   # regression: 30 nm score left behind
            @test prob.∇g == ∇g0                                 # regression: 30 nm gradient left behind
            @test prob.g_history == hist1 && prob.iteration == 7

            # Exit by an error inside the loop: the rebuild at the first point (10 nm) throws, because the
            # gradient-term measure is missing.
            old_design = sim.dΩ_design
            try
                sim.dΩ_design = nothing
                @test_throws Exception HFR.Analysis.fabrication_sweep(prob, p; center_R=20.0, range_R=10.0,
                                                                      points=2, root=mktempdir(), save_plot=false)
            finally
                sim.dΩ_design = old_design
            end
            @test prob.control.R_filter == R0                    # regression: radius left at 10 nm
            @test prob.g == g0
            @test prob.∇g == ∇g0
            @test prob.g_history == hist1 && prob.iteration == 7
        finally
            ctrl.R_filter = R0
            empty!(prob.g_history); append!(prob.g_history, hist0)
            prob.iteration = it0
        end
    end

    @testset "a release error leaves no factor marked valid" begin
        bad = HFR.SolverCache(HFRReleaseFails())
        c1 = deepcopy(ctrl)
        c1.R_filter = R1
        @test hfr_relerr(filter_helmholtz!(p, bad, sim, c1), fwd[R1]) < HFR_RTOL
        @test HFR.has_filter_factor(bad) && bad.F_key.R == R1
        h1 = bad.F_factor
        @test_throws ErrorException HFR.clear_filter_factor!(bad)
        @test !h1.live                                           # the test backend marked the handle released
        @test !HFR.has_filter_factor(bad)                       # regression: factor left marked valid after a release error
        @test isnothing(bad.F_key)                               # regression: factor left marked valid after a release error
        # A retry at the old radius factorizes again; the backend throws on a solve with h1.
        @test hfr_relerr(filter_helmholtz!(p, bad, sim, c1), fwd[R1]) < HFR_RTOL     # regression: released handle selected again
        @test bad.F_factor !== h1 && bad.F_factor.live
        @test bad.F_key.R == R1
        # A radius change releases through the same path: the error propagates, the cache is empty.
        c1.R_filter = R2
        @test_throws ErrorException filter_helmholtz!(p, bad, sim, c1)
        @test !HFR.has_filter_factor(bad) && isnothing(bad.F_key)  # regression: factor left marked valid after a release error
    end

    @testset "installing a factor directly clears the key" begin
        ctrl.R_filter = R1
        filter_helmholtz!(p, cache, sim, ctrl)                  # factor and key for R1
        HFR.filter_lu!(cache, 2 .* HFR.assemble_filter_matrix(R1, sim))   # some other matrix, no key
        @test isnothing(cache.F_key)                             # regression: key left set after a direct install
        @test hfr_relerr(filter_helmholtz!(p, cache, sim, ctrl), fwd[R1]) < HFR_RTOL     # regression: result half the expected value from the stale key
        @test cache.F_key.R == R1
    end

    @testset "empty! on a cache pool clears the filter factor in every slot" begin
        # Any SolverCache of the pool can be passed to the filter, including its eigen slot.
        # After an in-place edit of a filter measure (not detected by the key), empty! on the
        # pool must leave every slot to factorize the edited operator. The edit doubles the
        # reference quadrature weights of sim.dΩ_design in place and is undone at the end.
        pool = HFR.default_pool(prob.pool)
        ctrl.R_filter = R1
        w = sim.dΩ_design.quad.cell_weight[1]
        w0 = copy(w)
        try
            for slot in (pool.eigen_cache, pool.filter_cache)
                filter_helmholtz!(p, slot, sim, ctrl)       # each slot holds a factor for R1
                @test HFR.has_filter_factor(slot)
            end
            w .*= 2
            ref = hfr_cold_filter(p, sim, ctrl, R1)
            @test hfr_relerr(ref, fwd[R1]) > 1e-3           # the in-place edit changes the operator
            empty!(pool)
            for slot in (pool.eigen_cache, pool.filter_cache)
                @test !HFR.has_filter_factor(slot)          # regression: filter factor left in a pool slot
                @test isnothing(slot.F_key)                 # regression: filter key left in a pool slot
                @test hfr_relerr(filter_helmholtz!(p, slot, sim, ctrl), ref) < HFR_RTOL   # regression: stale factor reused after empty!
            end
        finally
            w .= w0
            empty!(pool)                                    # drop the factors built on the edited weights
        end
        @test hfr_relerr(filter_helmholtz!(p, pool.eigen_cache, sim, ctrl), fwd[R1]) < HFR_RTOL   # regression: factor of the edited weights kept after empty!
    end
    ctrl.R_filter = R0
end
