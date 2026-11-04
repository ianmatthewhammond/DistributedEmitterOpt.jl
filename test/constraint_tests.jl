"""
Constraint Tests for DistributedEmitterOpt.jl

Validate constraint values and gradients for:
- 2D DOF (foundry) grid constraints
- 3D DOF (mesh) FE constraints

Run with: julia --project test/constraint_tests.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

# Import helpers to access simulation from bundles
import DistributedEmitterOpt: default_sim, default_pool

const PERTURBATION = 1e-6
const TOL_RELATIVE = 1e-3

"""Directional derivative check for constraint callback."""
function check_constraint_grad(f, p0; δ=PERTURBATION)
    n = length(p0)
    grad = zeros(Float64, n)
    val0 = f(p0, grad)
    d = randn(n)
    d ./= norm(d)
    val1 = f(p0 .+ δ .* d, Float64[])
    fd = (val1 - val0) / δ
    adj = dot(grad, d)
    rel = abs(fd - adj) / max(abs(fd), 1e-12)
    return rel, fd, adj
end

"""Build a small simulation for constraint testing."""
function build_constraint_problem(; foundry_mode::Bool)
    # Geometry
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
    pde = MaxwellProblem(env=env, inputs=inputs, outputs=FieldConfig[])

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
    prob = OptimizationProblem(pde, SERSObjective(), meshfile, solver;
        per_x=false,
        per_y=false,
        foundry_mode=foundry_mode,
        control=control
    )

    rm(meshfile; force=true)
    return prob
end

"""Full period of a mirror-walled quarter cell: reflected across both walls, wall nodes once."""
function mirror_period(f)
    n, m = size(f)
    half = vcat(reverse(f[2:n-1, :]; dims=1), f)
    return hcat(reverse(half[:, 2:m-1]; dims=2), half)
end

@testset "Constraints Tests" begin
    Random.seed!(42)

    @testset "2D indicator at cell walls" begin
        ind = DistributedEmitterOpt.indicator_solid
        res, c0 = 0.8, 64 * 3.0^2
        # A smooth field with large gradients normal to every wall.
        n, m = 31, 23
        f = [0.5 + 0.4 * sin(0.3i + 0.1) * cos(0.25j - 0.2) for i in 1:n, j in 1:m]

        # Mirror walls: the quarter cell's indicator is the mirrored period's (periodic) one.
        quarter = ind(f, c0, identity, identity, res; mirror=true)
        period = ind(mirror_period(f), c0, identity, identity, res; mirror=false)
        @test quarter ≈ period[n-1:end, m-1:end] atol = 1e-14

        # Periodic cell: translating the field translates the indicator, along x and y.
        g = rand(n, m)
        base = ind(g, c0, identity, identity, res; mirror=false)
        for s in ((5, 0), (0, 7), (3, 11))
            @test ind(circshift(g, s), c0, identity, identity, res; mirror=false) ≈ circshift(base, s) atol = 1e-14
        end
    end

    @testset "2D Foundry Constraints" begin
        prob = build_constraint_problem(foundry_mode=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))

        # Use default_sim to get Simulation from SimulationBundle
        sim0 = default_sim(prob.sim)

        rel_s, _, _ = check_constraint_grad(
            (p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), p0
        )
        @test rel_s < TOL_RELATIVE

        rel_v, _, _ = check_constraint_grad(
            (p, g) -> glc_void(p, g; sim=sim0, control=prob.control), p0
        )
        @test rel_v < TOL_RELATIVE
    end

    @testset "3D FE Constraints" begin
        prob = build_constraint_problem(foundry_mode=false)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))

        # Use default_sim and default_pool to access Simulation and cache
        sim0 = default_sim(prob.sim)
        pool0 = default_pool(prob.pool)
        obj = (; sim=sim0, control=prob.control, cache_pump=pool0.filter_cache)

        rel_s, _, _ = check_constraint_grad(
            (p, g) -> glc_solid_fe(p, g, obj), p0
        )
        @test rel_s < TOL_RELATIVE

        rel_v, _, _ = check_constraint_grad(
            (p, g) -> glc_void_fe(p, g, obj), p0
        )
        @test rel_v < TOL_RELATIVE
    end
end

println("\nConstraint tests complete!")

