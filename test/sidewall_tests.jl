# Linear sidewall extrusion: the flag-off path must be untouched, and the
# flag-on adjoint must agree with finite differences on a real FEM problem.
using DistributedEmitterOpt
using LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "gradient_common.jl"))

const DEO = DistributedEmitterOpt

@testset "Linear sidewalls" begin

    # ── the pointwise projection ─────────────────────────────────────────────
    @testset "projection: slope=0 is bit-identical to the z-invariant SSP" begin
        R, β, η = 8.0, 8.0, 0.5
        rng = MersenneTwister(20260917)
        for _ in 1:2000
            ρ̃ = rand(rng)*1.4 - 0.2
            ∇ρ̃ = DEO.VectorValue(randn(rng)*0.05, randn(rng)*0.05, 0.0)
            z = rand(rng)*70
            a = DEO.smooth_project(ρ̃, ∇ρ̃; R, β, η)
            b = DEO.smooth_project_sidewall(ρ̃, ∇ρ̃, z; R, β, η, slope=0.0, z0=0.0)
            @test a === b     # identical bits, not just approximately equal
        end
    end

    @testset "projection: the wall leans, flat regions do not" begin
        R, β, η = 8.0, 8.0, 0.5
        s, z0 = 0.0875, 0.0
        # a wall: |∇ρ̃| = a, so the η-crossing must shift by exactly s·Δz
        a = 0.01
        ∇ρ̃ = DEO.VectorValue(a, 0.0, 0.0)
        cross(z) = begin
            lo, hi = -100.0, 100.0
            f(x) = DEO.smooth_project_sidewall(η + a*x, ∇ρ̃, z; R, β, η, slope=s, z0) - 0.5
            for _ in 1:200
                m = (lo+hi)/2
                f(lo)*f(m) <= 0 ? (hi = m) : (lo = m)
            end
            (lo+hi)/2
        end
        x0 = cross(0.0)
        for z in (10.0, 35.0, 70.0)
            @test isapprox(cross(z) - x0, s*z; atol=1e-8)
        end
        # flat: no wall ⇒ no z dependence at all
        for ρ̃ in (0.0, 0.35, 1.0)
            flat = DEO.VectorValue(0.0, 0.0, 0.0)
            v0 = DEO.smooth_project_sidewall(ρ̃, flat, 0.0;  R, β, η, slope=0.3, z0=0.0)
            v1 = DEO.smooth_project_sidewall(ρ̃, flat, 70.0; R, β, η, slope=0.3, z0=0.0)
            @test v0 === v1
        end
    end

    @testset "partials vs Richardson-extrapolated FD" begin
        R, β, η, z0 = 8.0, 8.0, 0.5, 0.0
        rng = MersenneTwister(1234)
        central(f,x,δ) = (f(x+δ)-f(x-δ))/(2δ)
        rich(f,x,δ) = (4*central(f,x,δ/2) - central(f,x,δ))/3
        worst_ρ = worst_g = 0.0
        for s in (0.0, 0.0875, -0.15), _ in 1:500
            ρ̃ = rand(rng)*1.4 - 0.2
            gx = 10.0^(rand(rng)*2 - 2.5)
            ∇ρ̃ = DEO.VectorValue(gx, 0.0, 0.0)
            z = rand(rng)*70
            a, c = DEO.sidewall_partials(ρ̃, ∇ρ̃, z; R, β, η, slope=s, z0)
            fρ = rich(x -> DEO.smooth_project_sidewall(x, ∇ρ̃, z; R,β,η,slope=s,z0), ρ̃, 1e-4)
            fg = rich(x -> DEO.smooth_project_sidewall(ρ̃, DEO.VectorValue(x,0.0,0.0), z;
                                                       R,β,η,slope=s,z0), gx, 1e-4*gx)
            abs(fρ) > 1e-4 && (worst_ρ = max(worst_ρ, abs(a-fρ)/abs(fρ)))
            abs(fg) > 1e-4 && (worst_g = max(worst_g, abs(c-fg)/abs(fg)))
        end
        println("  sidewall_partials worst rel err: ∂ρ̃ $worst_ρ  ∂|∇ρ̃| $worst_g")
        @test worst_ρ < 1e-5
        @test worst_g < 1e-5
    end

    # ── end to end on a real FEM problem ─────────────────────────────────────
    @testset "FEM: flag off is bit-identical to the current code" begin
        Random.seed!(20260917)
        p_ref = nothing; g_ref = nothing; grad_ref = nothing
        for use in (false, true)          # `true` here means the flag is on with slope 0
            prob = build_test_problem(foundry_mode=true, flag_volume=true,
                                      use_sidewall=use, sidewall_slope=0.0, sidewall_z0=0.0)
            np = length(DEO.default_sim(prob.sim).grid.params)
            p = fill(0.5, np); rng = MersenneTwister(7); p .+= 0.2 .* randn(rng, np)
            clamp!(p, 0.0, 1.0)
            grad = zeros(np)
            g = objective_and_gradient!(grad, p, prob)
            if use == false
                g_ref, grad_ref, p_ref = g, copy(grad), copy(p)
            else
                @test p == p_ref
                @test g == g_ref              # bit-identical objective
                @test grad == grad_ref        # bit-identical gradient
            end
        end
    end

    @testset "FEM: sloped walls move the design and the adjoint is right" begin
        Random.seed!(20260917)
        prob0 = build_test_problem(foundry_mode=true, flag_volume=true, use_sidewall=false)
        np = length(DEO.default_sim(prob0.sim).grid.params)
        rng = MersenneTwister(7)
        p = clamp!(fill(0.5, np) .+ 0.2 .* randn(rng, np), 0.0, 1.0)
        g_straight = objective_and_gradient!(Float64[], p, prob0)

        prob = build_test_problem(foundry_mode=true, flag_volume=true,
                                  use_sidewall=true, sidewall_slope=0.15, sidewall_z0=0.0)
        grad = zeros(np)
        g_sloped = objective_and_gradient!(grad, p, prob)
        println("  g straight = $g_straight   g sloped = $g_sloped")
        @test isfinite(g_sloped)
        @test all(isfinite, grad)
        @test g_sloped != g_straight          # the taper must actually do something
        @test norm(grad) > 0

        rel, _ = test_gradient(prob, p; δ=1e-8)
        @test rel < TOL_RELATIVE
    end

    @testset "FEM: re-entrant slope also differentiates" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=true,
                                  use_sidewall=true, sidewall_slope=-0.15, sidewall_z0=0.0)
        np = length(DEO.default_sim(prob.sim).grid.params)
        rng = MersenneTwister(7)
        p = clamp!(fill(0.5, np) .+ 0.2 .* randn(rng, np), 0.0, 1.0)
        rel, g = test_gradient(prob, p; δ=1e-8)
        @test isfinite(g)
        @test rel < TOL_RELATIVE
    end
end
