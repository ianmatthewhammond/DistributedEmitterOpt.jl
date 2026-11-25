# `spectral_sweep` evaluates the design at each wavelength on the caller's problem, and each evaluation overwrites
# `prob.g` and `prob.∇g`. On return, normal or by an error, both must hold the values they had before the call,
# and `prob.∇g` must be the caller's vector.
using DistributedEmitterOpt
using Random
using Test

isdefined(@__MODULE__, :build_test_problem) || include("gradient_common.jl")

const SSS = DistributedEmitterOpt

@testset "spectral_sweep restores score and gradient on both exits" begin
    prob = build_test_problem()                     # 3D sandbox, Ag, built at 532 nm
    p = rand(MersenneTwister(23), length(prob.p))   # fixed nonuniform design
    pde0 = prob.pde
    try
        # Populated state at 532 nm, which is not a sweep point below
        SSS.evaluate(prob, p)
        g0, ∇g0, ∇g_ref = prob.g, copy(prob.∇g), prob.∇g
        @test !isempty(prob.∇g)

        # Normal exit: the sweep visits 512 and 552 nm
        λs, gys = SSS.Analysis.spectral_sweep(prob, p; center_λ=532.0, range_λ=20.0, points=2,
                                              root=mktempdir(), save_plot=false)
        @test λs ≈ [512.0, 552.0]
        @test !isapprox(gys[end], g0; rtol=1e-6)    # the last point's score is not the caller's
        @test prob.g == g0                          # regression: 552 nm score left behind
        @test prob.∇g == ∇g0                        # regression: 552 nm gradient left behind
        @test prob.∇g === ∇g_ref                    # restored in place, not replaced

        # Exit by an error inside the loop: the sweep visits 600 and 2000 nm. The Ag table ends below 2000 nm, so
        # the second evaluation throws after the first has overwritten the score and gradient.
        @test resolve_index("Ag", 600.0) isa ComplexF64
        @test_throws BoundsError resolve_index("Ag", 2000.0)
        @test_throws Exception SSS.Analysis.spectral_sweep(prob, p; center_λ=1300.0, range_λ=700.0, points=2,
                                                           root=mktempdir(), save_plot=false)
        @test prob.g == g0                          # regression: 600 nm score left behind
        @test prob.∇g == ∇g0                        # regression: 600 nm gradient left behind
        @test prob.∇g === ∇g_ref
    finally
        # The sweep does not restore the wavelength after an error (tracked separately)
        prob.pde = pde0
        empty!(prob.pool)
    end
end
