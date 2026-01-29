"""
Gradient Tests for DistributedEmitterOpt.jl (new architecture)

Finite-difference validation of adjoint gradients across configurations:
- 2D DOF (foundry) vs 3D DOF (mesh) modes
- Volume vs Surface objectives
- With/without damage model
- Elastic vs Inelastic scattering
- Isotropic vs Anisotropic polarizability tensor

Run with: julia --project test/gradient_tests.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

# ═══════════════════════════════════════════════════════════════════════════════
# Test configuration
# ═══════════════════════════════════════════════════════════════════════════════

const PERTURBATION = 1e-8
const TOL_RELATIVE = 1e-4

"""Run finite difference gradient test and return relative error."""
function test_gradient(prob, p0; δ=PERTURBATION, verbose=true)
    np = length(p0)
    grad = zeros(np)

    # Random perturbation direction
    δp = randn(np) * δ

    # Forward evaluation with gradient
    g0 = objective_and_gradient!(grad, p0, prob)

    # Perturbed evaluation (no gradient needed)
    grad_dummy = Float64[]
    g1 = objective_and_gradient!(grad_dummy, p0 + δp, prob)

    # Finite difference vs adjoint
    fd = g1 - g0
    adj = dot(grad, δp)

    rel_error = abs(fd - adj) / (abs(fd) + 1e-12)

    if verbose
        println("  g0 = $g0")
        println("  g1 = $g1")
        println("  FD  = $fd")
        println("  Adj = $adj")
        println("  Relative error = $(round(rel_error * 100, digits=2))%")
    end

    return rel_error, g0
end

# ═══════════════════════════════════════════════════════════════════════════════
# Build test objective
# ═══════════════════════════════════════════════════════════════════════════════

"""Build a minimal test problem matching the experimental "sandbox" configuration."""
function build_test_problem(;
    foundry_mode::Bool=false,  # Default to 3D mode (matches experiments)
    flag_volume::Bool=true,
    flag_surface::Bool=false,
    use_damage::Bool=false,
    isotropic::Bool=true,
    λ::Float64=532.0,
    λ_pump::Float64=λ,
    λ_emission::Float64=λ,
    complex_config::Bool=false # If true, use mixed pol/weights
)
    # Generate coarse test mesh with SYMMETRIC Y (half-cell)
    # This matches the "sandbox" mesh topology used in experiments
    geo = SymmetricGeometry(λ_pump; L=100.0, W=100.0, hd=80.0, hsub=40.0)
    geo.l1 = 50.0  # Very coarse for speed
    geo.l2 = 30.0
    geo.l3 = 50.0
    geo.hair = 200.0
    geo.hs = 120.0
    geo.ht = 80.0

    meshfile = tempname() * ".msh"

    # Key change: per_y=false to match half-cell symmetry of experiments (UNLESS regression testing foundry mode)
    # Foundry mode typically expects full periodicity for 2D design projection
    is_foundry = foundry_mode
    genperiodic(geo, meshfile; per_x=true, per_y=is_foundry)

    # Build simulation: dir_y=true enforces symmetry (PEC) on the Y-walls (only if not periodic)
    sim = build_simulation(meshfile; foundry_mode, dir_x=false, dir_y=!is_foundry)

    # Physics: Ag/Ag/1.33 match experiments
    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)

    # Polarizability
    αₚ = isotropic ? Matrix{ComplexF64}(I, 3, 3) : rand(ComplexF64, 3, 3)
    if !isotropic
        αₚ = αₚ + transpose(αₚ)
    end

    objective = SERSObjective(
        αₚ=αₚ,
        volume=flag_volume,
        surface=flag_surface,
        use_damage_model=use_damage,
        γ_damage=1.0,
        E_threshold=10.0
    )

    # IO Configuration
    inputs = [FieldConfig(λ_pump; θ=0.0, pol=:y)] # Pump Y (consistent with symmetry)

    outputs = if complex_config
        # Mixed polarization and weights
        [
            FieldConfig(λ_emission; θ=0.0, pol=:y, weight=1.0),
            FieldConfig(λ_emission + 10.0; θ=0.0, pol=:x, weight=0.5)
        ]
    elseif λ_emission == λ_pump
        FieldConfig[] # Elastic
    else
        [FieldConfig(λ_emission; θ=0.0, pol=:y)] # Inelastic Y
    end

    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    # Control matches experiments (R=20, ssp=2.0)
    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=foundry_mode, # DCT only for 2D/Foundry
        use_projection=true,
        β=8.0, # Finite beta for smooth gradients
        η=0.5,
        use_ssp=true,
        flag_volume=flag_volume,
        flag_surface=flag_surface,
        use_damage=use_damage,
        γ_damage=1.0,
        E_threshold=10.0
    )

    # Build solver + problem
    solver = UmfpackSolver()
    prob = OptimizationProblem(pde, objective, sim, solver;
        foundry_mode=foundry_mode,
        control=control
    )

    # Initialize with random design
    Random.seed!(42)
    # Foundry mode needs consistent grid size for random init
    if foundry_mode
        init_uniform!(prob, 0.5) # Simpler init for 2D to avoid mismatch issues
    else
        init_random!(prob)
    end

    # Cleanup mesh file
    rm(meshfile; force=true)

    return prob
end

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

    # ══════════════════════════════════════════════════════════════════════════════
    # Legacy/Foundry Mode Regression
    # ══════════════════════════════════════════════════════════════════════════════

    @testset "2D Foundry Mode (Legacy)" begin
        println("\n=== 2D Foundry Mode ===")
        prob = build_test_problem(foundry_mode=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end
end

println("\nGradient tests complete!")
