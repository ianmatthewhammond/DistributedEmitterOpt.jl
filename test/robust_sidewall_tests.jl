# Worst-case design over a range of sidewall slopes: the aggregation, its
# gradient, and an end-to-end FD check on a real FEM problem.
using DistributedEmitterOpt
using LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "gradient_common.jl"))

const DEO = DistributedEmitterOpt

@testset "Robust sidewall (worst case over a slope range)" begin

    @testset "sample-set construction" begin
        S = sidewall_samples(angles_deg=[80, 85, 90], z0s=[0.0])
        @test length(S) == 3
        @test S[end] == (0.0, 0.0)                      # 90° is the vertical extrusion
        @test isapprox(S[1][1], 1/tan(deg2rad(80)); atol=1e-12)
        @test isapprox(S[2][1], 1/tan(deg2rad(85)); atol=1e-12)

        # both z₀ conventions, but the vertical sample is not solved twice
        S2 = sidewall_samples(angles_deg=[80, 85, 90], z0s=[0.0, 70.0])
        @test length(S2) == 5
        @test count(x -> x[1] == 0.0, S2) == 1
        @test allunique(S2)

        # 85° over hd=70nm: the two z₀ conventions differ by 6.1 nm of linewidth
        @test isapprox(1/tan(deg2rad(85)) * 70.0, 6.12; atol=0.01)
    end

    @testset "ks_softmin: bound, weights, limit, derivative" begin
        rng = MersenneTwister(4242)
        for _ in 1:400
            gs = 100 .* (0.5 .+ rand(rng, rand(rng, 2:6)))
            sc = 100.0                      # fixed reference scale, NOT from gs
            for sharp in (5.0, 20.0, 100.0)
                g, w = ks_softmin(gs, sharp; scale=sc)
                @test all(w .>= 0)
                @test isapprox(sum(w), 1.0; atol=1e-12)
                @test g <= minimum(gs) + 1e-12          # never optimistic
                @test g > minimum(gs) - abs(minimum(gs))   # and not wildly loose
            end
            # → min as sharpness → ∞
            @test isapprox(ks_softmin(gs, 1e6; scale=sc)[1], minimum(gs); rtol=1e-6)
            # exact mode picks the argmin and puts all weight there
            g∞, w∞ = ks_softmin(gs, Inf)
            @test g∞ == minimum(gs)
            @test w∞[argmin(gs)] == 1.0
            @test sum(w∞) == 1.0
        end

        # the weights ARE the derivative ∂g_KS/∂gᵢ — check against central FD
        # The weights must be the derivative for EVERY component including the
        # argmin — the component an offset-derived sharpness gets wrong.
        gs = [1.0, 1.12, 0.94, 1.31]
        for sharp in (5.0, 20.0, 60.0), sc in (1.0, 2.5)
            g_agg, w = ks_softmin(gs, sharp; scale=sc)
            # Weights span ten decades, so score each against the FD's OWN noise
            # floor rather than one absolute tolerance: a central difference of a
            # quantity of size |g| resolves no better than eps(|g|)/δ, and below
            # that the "FD value" is ulp quantization. (Two components previously
            # failed here at w ~ 2e-10 against an FD of exactly 2.5*eps() — the
            # instrument, not the derivative.) Richardson-extrapolate to keep
            # truncation well under that floor, and check the unresolvable
            # components collectively via sum(w) == 1 instead.
            δ = 1e-5
            floor_fd = 8 * eps(abs(g_agg)) / δ
            for i in eachindex(gs)
                f(x) = begin
                    t = copy(gs); t[i] = x; ks_softmin(t, sharp; scale=sc)[1]
                end
                c(h) = (f(gs[i]+h) - f(gs[i]-h)) / (2h)
                fd = (4*c(δ/2) - c(δ)) / 3
                max(abs(w[i]), abs(fd)) < floor_fd && continue
                @test isapprox(w[i], fd; rtol=1e-6, atol=floor_fd)
            end
            @test isapprox(sum(w), 1.0; atol=1e-12)
            # specifically: the argmin's weight, where the old bug lived
            _, w2 = ks_softmin(gs, sharp; scale=sc)
            @test w2[argmin(gs)] == maximum(w2)
        end

        @testset "scale is rejected when it cannot define a sharpness" begin
            @test_throws ArgumentError ks_softmin([1.0, 2.0], 5.0; scale=0.0)
            @test_throws ArgumentError ks_softmin([1.0, 2.0], 5.0; scale=-1.0)
            @test_throws ArgumentError ks_softmin([1.0, 2.0], 5.0; scale=NaN)
            # exact-min mode needs no scale at all
            @test ks_softmin([3.0, 1.0, 2.0], Inf)[1] == 1.0
        end
    end

    @testset "FEM: one vertical sample reduces to the plain objective" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=true)
        np = length(DEO.default_sim(prob.sim).grid.params)
        rng = MersenneTwister(11)
        p = clamp!(fill(0.5, np) .+ 0.2 .* randn(rng, np), 0.0, 1.0)

        gp = zeros(np); g_plain = objective_and_gradient!(gp, p, prob)
        gr = zeros(np)
        g_rob, gs, w = robust_sidewall_objective!(gr, p, prob, [(0.0, 0.0)];
                                                   sharpness=50.0, scale=abs(g_plain))
        @test gs[1] == g_plain          # bit-identical: slope 0 dispatches to the old path
        @test g_rob == g_plain
        @test gr == gp
        @test w == [1.0]
        # the control must come back exactly as it was
        @test prob.control.use_sidewall == false
        @test prob.control.sidewall_slope == 0.0
    end

    @testset "FEM: samples differ, and the worst case is the worst" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=true)
        np = length(DEO.default_sim(prob.sim).grid.params)
        rng = MersenneTwister(11)
        p = clamp!(fill(0.5, np) .+ 0.2 .* randn(rng, np), 0.0, 1.0)

        S = sidewall_samples(angles_deg=[78, 84, 90], z0s=[0.0])
        g0 = objective_and_gradient!(Float64[], p, prob)   # vertical: the run's scale
        g_rob, gs, w = robust_sidewall_objective!(Float64[], p, prob, S;
                                                  sharpness=50.0, scale=abs(g0),
                                                  verbose=true)
        # Every sample must be a genuinely different design. If a cache froze
        # between samples these would be equal.
        @test allunique(gs)
        @test length(unique(round.(gs, sigdigits=10))) == length(gs)
        @test g_rob <= minimum(gs) + 1e-12
        @test all(isfinite, gs)
        println("  worst case = $g_rob over samples $gs")
    end

    @testset "FEM: robust gradient vs central FD" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=true)
        np = length(DEO.default_sim(prob.sim).grid.params)
        rng = MersenneTwister(11)
        p = clamp!(fill(0.5, np) .+ 0.2 .* randn(rng, np), 0.0, 1.0)
        S = sidewall_samples(angles_deg=[78, 84, 90], z0s=[0.0])
        # scale is fixed for the whole check, exactly as g_norm would be in a run
        sc = abs(objective_and_gradient!(Float64[], p, prob))

        for sharp in (20.0, 50.0)
            grad = zeros(np)
            robust_sidewall_objective!(grad, p, prob, S; sharpness=sharp, scale=sc)
            @test all(isfinite, grad)
            @test norm(grad) > 0

            d = randn(rng, np); d ./= norm(d)
            δ = 1e-6
            gp, _, _ = robust_sidewall_objective!(Float64[], p .+ δ .* d, prob, S;
                                                   sharpness=sharp, scale=sc)
            gm, _, _ = robust_sidewall_objective!(Float64[], p .- δ .* d, prob, S;
                                                   sharpness=sharp, scale=sc)
            fd  = (gp - gm) / (2δ)
            adj = dot(grad, d)
            rel = abs(adj - fd) / max(abs(fd), 1e-300)
            println("  sharpness $sharp:  adj = $adj   fd = $fd   rel = $rel")
            @test rel < TOL_RELATIVE
        end
    end

    @testset "FEM: control is restored even when a sample throws" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=true)
        np = length(DEO.default_sim(prob.sim).grid.params)
        @test_throws Exception robust_sidewall_objective!(
            Float64[], fill(NaN, np - 1), prob, [(0.1, 0.0)])   # wrong length ⇒ throws
        @test prob.control.use_sidewall == false
        @test prob.control.sidewall_slope == 0.0
        @test prob.control.sidewall_z0 == 0.0
    end
end
