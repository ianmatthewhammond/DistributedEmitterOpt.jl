# Sidewalls combined with the surface objective and active damage.
# Combination checks supplement the separate projection and surface tests.
using DistributedEmitterOpt
using LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "gradient_common.jl"))

const DEO = DistributedEmitterOpt

design_np(prob) = length(DEO.default_sim(prob.sim).grid.params)

"""
    design_z0(prob) -> Float64

Use the measured substrate interface as the pivot: the drawn design is the footprint,
and material recedes above it. The pivot is an absolute physical coordinate, so a value
outside the design layer can turn a taper into an unintended dilation.
"""
function design_z0(prob)
    sim0 = DEO.default_sim(prob.sim)
    z_lo, _ = DEO.getdesignz(sim0.labels, sim0.Ω)
    return z_lo
end

function blob_design(np::Int; seed=20260917)
    rng = MersenneTwister(seed)
    n = isqrt(np)
    @assert n*n == np "expected a square design grid, got np=$np"
    P = zeros(n, n)
    for _ in 1:6
        cx, cy, r, a = rand(rng)*n, rand(rng)*n, 0.08n + 0.12n*rand(rng), 0.5 + 0.5rand(rng)
        for j in 1:n, i in 1:n
            P[i,j] += a * exp(-((i-cx)^2 + (j-cy)^2) / (2r^2))
        end
    end
    return clamp!(vec(P ./ maximum(P)), 0.0, 1.0)
end

@testset "Sidewalls × tv_void × damage" begin

    @testset "tv_void + sidewall: gradient" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                  surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                                  use_sidewall=true, sidewall_slope=0.15)
        prob.control.sidewall_z0 = design_z0(prob)
        p = blob_design(design_np(prob))
        grad = zeros(length(p))
        g = objective_and_gradient!(grad, p, prob)
        println("  tv_void + sidewall (z0=$(prob.control.sidewall_z0)): g = $g,  |∇g| = $(norm(grad))")
        @test isfinite(g) && g > 0
        @test all(isfinite, grad)
        @test norm(grad) > 0

        idx = partialsortperm(abs.(grad), 1:4, rev=true)
        dirs = Vector{Float64}[]
        for i in idx
            d = zeros(length(p)); d[i] = 1.0; push!(dirs, d)
        end
        rng = MersenneTwister(3); dd = randn(rng, length(p)); dd ./= norm(dd); push!(dirs, dd)
        for (adj, fd, rel) in central_fd_check(prob, p, grad, dirs; δ=1e-5)
            println("    adj=$adj  fd=$fd  rel=$rel")
            @test rel < TOL_RELATIVE
        end
    end

    @testset "tv_void + sidewall: the taper actually changes the shell" begin
        Random.seed!(20260917)
        p = nothing; gs = Float64[]
        for s in (0.0, 0.10, 0.20)
            prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                      surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                                      use_sidewall=(s != 0.0), sidewall_slope=s)
            prob.control.sidewall_z0 = design_z0(prob)
            p === nothing && (p = blob_design(design_np(prob)))
            push!(gs, objective_and_gradient!(Float64[], p, prob))
        end
        println("  tv_void g vs slope 0 / 0.10 / 0.20: $gs")
        @test all(isfinite, gs)
        @test allunique(gs)          # the shell must see the taper
    end

    @testset "tv_void + damage + sidewall: gradient" begin
        Random.seed!(20260917)
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                  surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                                  use_damage=true, γ_damage=0.4, E_threshold=1e-3,
                                  use_sidewall=true, sidewall_slope=0.15)
        prob.control.sidewall_z0 = design_z0(prob)
        p = blob_design(design_np(prob))
        grad = zeros(length(p))
        g = objective_and_gradient!(grad, p, prob)
        println("  tv_void + damage + sidewall: g = $g,  |∇g| = $(norm(grad))")
        @test isfinite(g) && g > 0
        @test all(isfinite, grad)
        @test norm(grad) > 0

        idx = partialsortperm(abs.(grad), 1:3, rev=true)
        dirs = Vector{Float64}[]
        for i in idx
            d = zeros(length(p)); d[i] = 1.0; push!(dirs, d)
        end
        rng = MersenneTwister(5); dd = randn(rng, length(p)); dd ./= norm(dd); push!(dirs, dd)
        for (adj, fd, rel) in central_fd_check(prob, p, grad, dirs; δ=1e-5)
            println("    adj=$adj  fd=$fd  rel=$rel")
            @test rel < TOL_RELATIVE
        end
        # E_threshold low enough that damage is genuinely active, else this test
        # is only re-testing the damage-off path.
        prob_nodmg = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                        surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                                        use_sidewall=true, sidewall_slope=0.15)
        prob_nodmg.control.sidewall_z0 = design_z0(prob_nodmg)
        g_nodmg = objective_and_gradient!(Float64[], p, prob_nodmg)
        println("  damage on $g  vs off $g_nodmg  (ratio $(g/g_nodmg))")
        @test g != g_nodmg
    end

    @testset "tv_void + sidewall: flag off is bit-identical" begin
        Random.seed!(20260917)
        p = nothing; ref = nothing
        for use in (false, true)
            prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                      surface_mode=:tv_void, t_shell=10.0, ε_grad=1e-6,
                                      use_sidewall=use, sidewall_slope=0.0)
            prob.control.sidewall_z0 = design_z0(prob)
            p === nothing && (p = blob_design(design_np(prob)))
            grad = zeros(length(p))
            g = objective_and_gradient!(grad, p, prob)
            if ref === nothing
                ref = (g, copy(grad))
            else
                @test g == ref[1]
                @test grad == ref[2]
            end
        end
    end
end
