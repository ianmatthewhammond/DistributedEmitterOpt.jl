# Gradient Tests for DistributedEmitterOpt.jl (new architecture)
#
# Finite-difference validation of adjoint gradients across configurations:
# - 2D DOF (foundry) vs 3D DOF (mesh) modes
# - Volume vs Surface objectives
# - With/without damage model
# - Elastic vs Inelastic scattering
# - Isotropic vs Anisotropic polarizability tensor
#
# Run with: julia --project test/gradient_tests.jl
# (helpers live in gradient_common.jl so other test files can reuse them)

include(joinpath(@__DIR__, "gradient_common.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
# Test suite
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistributedEmitterOpt Gradient Tests" begin

    # ══════════════════════════════════════════════════════════════════════════════
    # Primary Configurations (3D DOF, Matching Experiments)
    # ══════════════════════════════════════════════════════════════════════════════

    @testset "3D Elastic Baseline (Standard Sandbox)" begin
        println("\n=== 3D Elastic Baseline ===")
        # Matches new_sandbox: 3D, Volume, Elastic, Isotropic
        prob = build_test_problem(foundry_mode=false, flag_volume=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D Inelastic Scattering" begin
        println("\n=== 3D Inelastic ===")
        # Matches inelastic: Pump != Emission
        prob = build_test_problem(
            foundry_mode=false,
            flag_volume=true,
            λ_pump=532.0,
            λ_emission=600.0
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D Oblique Incidence (Shifted Nabla)" begin
        println("\n=== 3D Oblique Incidence (Shifted Nabla) ===")
        prob = build_test_problem(
            foundry_mode=false,
            θ=20.0,
            flag_volume=true
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D Complex Configuration" begin
        println("\n=== 3D Complex (Mixed Pol/Weights) ===")
        # Matches complex_multi_output
        prob = build_test_problem(
            foundry_mode=false,
            flag_volume=true,
            complex_config=true
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    # ══════════════════════════════════════════════════════════════════════════════
    # Variant Configurations (Surface, Damage, Anisotropic)
    # ══════════════════════════════════════════════════════════════════════════════

    @testset "3D Surface Objective" begin
        println("\n=== 3D Surface Objective ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D With Damage Model" begin
        println("\n=== 3D With Damage ===")
        prob = build_test_problem(foundry_mode=false, use_damage=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D Anisotropic (Elastic)" begin
        println("\n=== 3D Anisotropic (Elastic) ===")
        prob = build_test_problem(foundry_mode=false, isotropic=false)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D Anisotropic + Multi Output" begin
        println("\n=== 3D Anisotropic + Multi Output ===")
        prob = build_test_problem(
            foundry_mode=false,
            isotropic=false,
            complex_config=true
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    # ══════════════════════════════════════════════════════════════════════════════
    # Legacy/Foundry Mode Regression
    # ══════════════════════════════════════════════════════════════════════════════

    @testset "2D Foundry Mode (Legacy)" begin
        println("\n=== 2D Foundry Mode ===")
        prob = build_test_problem(foundry_mode=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0; δ=1e-8)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D Foundry Inelastic" begin
        println("\n=== 2D Foundry Inelastic ===")
        prob = build_test_problem(
            foundry_mode=true,
            λ_pump=532.0,
            λ_emission=600.0
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0; δ=1e-8)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D Foundry Anisotropic" begin
        println("\n=== 2D Foundry Anisotropic ===")
        prob = build_test_problem(foundry_mode=true, isotropic=false)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0; δ=1e-8)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D Foundry Anisotropic + Multi Output" begin
        println("\n=== 2D Foundry Anisotropic + Multi Output ===")
        prob = build_test_problem(
            foundry_mode=true,
            isotropic=false,
            complex_config=true
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0; δ=1e-8)
        @test rel_err < TOL_RELATIVE
    end
end

println("\nGradient tests complete!")
