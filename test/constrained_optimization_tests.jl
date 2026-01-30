"""
Constrained Optimization Schedule Tests

Runs a short multi-epoch optimization with constraints only in the final epoch.
Covers 2D (foundry) and 3D (FE) problems with anisotropy and multiple outputs.

Run with: julia --project test/constrained_optimization_tests.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

const BETAS = [8.0, 16.0, 32.0, Inf]
const MAX_ITER = 5
const TOL = 1e-6

function anisotropic_tensor()
    α = ComplexF64[
        1.10+0.00im 0.05+0.02im 0.01-0.03im
        0.02-0.01im 0.95+0.00im 0.03+0.04im
        0.01+0.00im 0.02-0.02im 1.05+0.00im
    ]
    return (α + transpose(α)) / 2
end

function build_multi_output_problem(; foundry_mode::Bool)
    geo = if foundry_mode
        g = SymmetricGeometry()
        g.L = 200.0
        g.W = 200.0
        g.l1 = 40.0
        g.l2 = 20.0
        g.l3 = 40.0
        g
    else
        g = SymmetricGeometry(532.0; L=100.0, W=100.0, hd=80.0, hsub=40.0)
        g.l1 = 50.0
        g.l2 = 30.0
        g.l3 = 50.0
        g.hair = 200.0
        g.hs = 120.0
        g.ht = 80.0
        g
    end

    meshfile = tempname() * ".msh"
    genmesh(geo, meshfile; per_x=false, per_y=false)

    env = Environment(mat_design="Ag", mat_substrate="Ag", mat_fluid=1.33)
    inputs = [FieldConfig(532.0; θ=0.0, pol=:y)]
    outputs = [
        FieldConfig(600.0; θ=0.0, pol=:y, weight=1.0),
        FieldConfig(650.0; θ=0.0, pol=:x, weight=0.7),
    ]
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=outputs)

    objective = SERSObjective(
        αₚ=anisotropic_tensor(),
        volume=true,
        surface=false,
        use_damage_model=false
    )

    control = Control(
        use_filter=true,
        R_filter=(20.0, 20.0, 20.0),
        use_dct=foundry_mode,
        use_projection=true,
        β=8.0,
        η=0.5,
        use_ssp=true,
        flag_volume=true
    )

    solver = UmfpackSolver()
    prob = OptimizationProblem(pde, objective, meshfile, solver;
        per_x=false,
        per_y=false,
        foundry_mode=foundry_mode,
        control=control
    )

    init_uniform!(prob, 0.5)
    rm(meshfile; force=true)
    return prob
end

function run_schedule!(prob)
    for β in BETAS
        prob.control.β = β
        use_constraints = (β == Inf)
        g_opt, p_opt, _ = DistributedEmitterOpt.run_epoch!(prob, MAX_ITER, use_constraints, TOL)
        prob.p .= p_opt
        prob.g = g_opt
        @test isfinite(g_opt)
        @test length(p_opt) == length(prob.p)
    end
end

@testset "Constrained Optimization Schedule" begin
    Random.seed!(42)

    @testset "2D Foundry" begin
        prob = build_multi_output_problem(foundry_mode=true)
        run_schedule!(prob)
    end

    @testset "3D FE" begin
        prob = build_multi_output_problem(foundry_mode=false)
        run_schedule!(prob)
    end
end

println("\nConstrained optimization schedule tests complete!")
