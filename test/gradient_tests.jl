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

"""Build a minimal test problem with given configuration."""
function build_test_problem(;
    foundry_mode::Bool=true,
    flag_volume::Bool=true,
    flag_surface::Bool=false,
    use_damage::Bool=false,
    isotropic::Bool=true,
    λ::Float64=532.0,
    λ_pump::Float64=λ,
    λ_emission::Float64=λ
)
    # Generate coarse test mesh
    geo = SymmetricGeometry(λ_pump; L=100.0, W=100.0, hd=80.0, hsub=40.0)
    geo.l1 = 50.0  # Very coarse
    geo.l2 = 30.0
    geo.l3 = 50.0
    geo.hair = 200.0
    geo.hs = 120.0
    geo.ht = 80.0

    meshfile = tempname() * ".msh"
    genperiodic(geo, meshfile; per_x=true, per_y=true)

    # Build simulation
    sim = build_simulation(meshfile; foundry_mode, dir_x=false, dir_y=true)

    # Build PDE + objective
    αₚ = isotropic ? Matrix{ComplexF64}(I, 3, 3) : rand(ComplexF64, 3, 3)
    if !isotropic
        αₚ = αₚ + transpose(αₚ)  # Symmetric
    end

    objective = SERSObjective(
        αₚ=αₚ,
        volume=flag_volume,
        surface=flag_surface,
        use_damage_model=use_damage,
        γ_damage=1.0,
        E_threshold=10.0
    )

    env = Environment(mat_design="Ag", mat_fluid=1.33)
    inputs = [FieldConfig(λ_pump; θ=0.0, pol=:y)]
    outputs = λ_emission == λ_pump ? FieldConfig[] : [FieldConfig(λ_emission; θ=0.0, pol=:y)]
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    # Control
    control = Control(
        use_filter=true,
        R_filter=(15.0, 15.0, 15.0),
        use_dct=foundry_mode,
        use_projection=true,
        β=8.0,
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
    init_random!(prob)

    # Cleanup mesh file
    rm(meshfile; force=true)

    return prob
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test suite
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistributedEmitterOpt Gradient Tests" begin

    @testset "2D DOF (Foundry) Mode - Volume Objective" begin
        println("\n=== 2D DOF Mode, Volume Objective ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=true, flag_surface=false)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D DOF (Foundry) Mode - Surface Objective" begin
        println("\n=== 2D DOF Mode, Surface Objective ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D DOF Mode - Volume + Surface Combined" begin
        println("\n=== 2D DOF Mode, Volume + Surface ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=true, flag_surface=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D DOF Mode - With Damage Model" begin
        println("\n=== 2D DOF Mode, With Damage ===")
        prob = build_test_problem(foundry_mode=true, use_damage=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D DOF (Mesh) Mode - Volume Objective" begin
        println("\n=== 3D DOF Mode, Volume Objective ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=true, flag_surface=false)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D DOF Mode - Surface Objective" begin
        println("\n=== 3D DOF Mode, Surface Objective ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D DOF Mode - With Damage Model" begin
        println("\n=== 3D DOF Mode, With Damage ===")
        prob = build_test_problem(foundry_mode=false, use_damage=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D DOF Mode - Inelastic Volume Objective" begin
        println("\n=== 2D DOF Mode, Inelastic Volume ===")
        prob = build_test_problem(
            foundry_mode=true,
            flag_volume=true,
            flag_surface=false,
            λ_pump=532.0,
            λ_emission=600.0
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D DOF Mode - Inelastic Surface Objective" begin
        println("\n=== 3D DOF Mode, Inelastic Surface ===")
        prob = build_test_problem(
            foundry_mode=false,
            flag_volume=false,
            flag_surface=true,
            λ_pump=532.0,
            λ_emission=600.0
        )
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    # @testset "Anisotropic Polarizability" begin
    #     println("\n=== Anisotropic Mode ===")
    #     prob = build_test_problem(isotropic=false)
    #     p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
    #     rel_err, _ = test_gradient(prob, p0)
    #     @test rel_err < TOL_RELATIVE
    # end
end

println("\nGradient tests complete!")
