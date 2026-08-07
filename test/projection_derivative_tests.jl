"""
Projection derivative tests — the SSP flat-field branch.

Regression test for the ×4 flat-field gradient bug: the degenerate branch of
`DSP_dpf` (‖∇ρ̃‖ < 1e-8) returned `∂projection_∂pf(ρ̃, β, η)` WITHOUT the `* p`
test-function factor that the non-degenerate branch carries. On P1 tets each
nodal entry then received the full cell integral instead of its 1/4 share —
a gradient exactly 4× too large wherever the filtered field is flat (in
particular at uniform initialization, where it fires on every design cell and
locks CCSA into permanent step rejection: actual/predicted decrease = 1/4 < 1/2).

Root-caused 2026-07-20 (continue-figures quest); frozen-run forensics and the
escape-race mechanism 2026-08-07 (engaging-deo quest).

Run with: julia --project test/projection_derivative_tests.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Test

const DEO = DistributedEmitterOpt

@testset "DSP_dpf flat-field branch" begin
    R, β, η = 1.5, 8.0, 0.5
    ρ̃ = 0.5

    @testset "carries the test-function factor p" begin
        # In the flat-field limit the SSP derivative must reduce to the plain
        # projection derivative TIMES the test function p — the same factor the
        # non-degenerate branch applies to both of its projected terms.
        for p in (0.25, 0.5, 1.0)
            got = DEO.DSP_dpf(p, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
            want = DEO.∂projection_∂pf(ρ̃, β, η) * p
            @test got ≈ want rtol = 1e-14
        end
        # Linearity in p (the bug made this constant in p):
        d1 = DEO.DSP_dpf(1.0, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
        d4 = DEO.DSP_dpf(0.25, ρ̃, [0.0, 0.0, 0.0]; R, β, η)
        @test d1 ≈ 4 * d4 rtol = 1e-14
    end

    @testset "continuity across the degeneracy threshold" begin
        # The two branches must agree (to the smoothing scale) as ‖∇ρ̃‖ crosses
        # 1e-8. With the missing *p the jump is a factor 1/p — for p=0.25 a 4×
        # discontinuity this test catches immediately.
        p = 0.25
        n̂ = [1.0, 0.0, 0.0]
        below = DEO.DSP_dpf(p, ρ̃, 0.99e-8 * n̂; R, β, η)
        above = DEO.DSP_dpf(p, ρ̃, 1.01e-8 * n̂; R, β, η)
        @test below ≈ above rtol = 1e-6
    end
end
