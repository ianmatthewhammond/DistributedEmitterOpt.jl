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
function build_constraint_problem(; foundry_mode::Bool, R_filter::Float64=20.0)
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
        R_filter=(R_filter, R_filter, R_filter),
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

    @testset "2D DCT filter walls, tiled and untiled axes" begin
        D = DistributedEmitterOpt
        res = 1.0
        n, m = 9, 21
        x = [0.5 + 0.3 * sin(0.7i + 0.2) * cos(0.4j - 0.1) + 0.1 * cos(1.3i * j / m) for i in 1:n, j in 1:m]

        # A radius above half the x width and below half the y width: convolvedcti tiles x only.
        h = D.conic_filter(5.0, n - 1.0, m - 1.0, res)
        tiled = D.dcti_tiled((n, m), size(h))
        @test tiled == (true, false)
        f = D.convolvedcti(x; h)
        fh = D.convolvedcti(x; h, halo=true)
        @test size(fh) == (n + 2, m)
        @test fh[2:end-1, :] ≈ f atol = 1e-14
        # Across a tiled wall the filtered neighbour is not the mirror image of the inner node.
        @test maximum(abs.(fh[1, :] .- f[2, :])) > 1e-3
        @test maximum(abs.(fh[end, :] .- f[end-1, :])) > 1e-3

        # The stencil: the halo along the tiled axis, mirror reflection along the untiled one.
        g = D.gradient_sq(fh, res; mirror=true, halo=tiled)
        fy = hcat(f[:, 2:2], f, f[:, end-1:end-1])
        gx = (fh[3:end, :] .- fh[1:end-2, :]) .* (res / 2)
        gy = (fy[:, 3:end] .- fy[:, 1:end-2]) .* (res / 2)
        @test g ≈ gx .^ 2 .+ gy .^ 2 atol = 1e-14

        # Independent reference for the halo: filter the enlarged tiled array explicitly and read the
        # nodes next to the centre copy. Odd and even sizes; x-only, y-only and both-axis tiling.
        for (sx, sy, R) in ((9, 21, 5.0), (8, 20, 5.0), (21, 9, 5.0), (20, 8, 5.0), (7, 6, 4.0))
            xr = [0.5 + 0.3 * sin(0.7i + 0.2) * cos(0.4j - 0.1) + 0.1 * cos(1.3i * j / sy) for i in 1:sx, j in 1:sy]
            hr = D.conic_filter(R, sx - 1.0, sy - 1.0, res)
            npx, npy = D.dcti_tiles((sx, sy), size(hr))
            big = repeat(xr, outer=(npx, npy))
            hp = D.properpaddcti(hr, size(big))
            freq = D.DCT_I(hp)
            full = D.iDCT_I(D.DCT_I(big) .* (freq ./ maximum(freq)))
            ox, oy = (npx - 1) * sx ÷ 2, (npy - 1) * sy ÷ 2
            hx, hy = npx > 1, npy > 1
            ref = full[ox+1-hx:ox+sx+hx, oy+1-hy:oy+sy+hy]
            @test D.convolvedcti(xr; h=hr, halo=true) ≈ ref atol = 1e-12
        end

        # An untiled cell gets no halo, so the wall stencil there is the mirror one.
        h_small = D.conic_filter(2.0, n - 1.0, m - 1.0, res)
        @test D.dcti_tiled((n, m), size(h_small)) == (false, false)
        @test D.convolvedcti(x; h=h_small, halo=true) == D.convolvedcti(x; h=h_small)

        # Zygote's gradient through the halo matches central finite differences.
        thr = p -> 1 ./ (1 .+ exp.(-8 .* (p .- 0.5)))
        filt = p -> D.convolvedcti(p; h, halo=true)
        common = (; c0=64 * 5.0^2, filter_f=filt, threshold_f=thr, resolution=res, mirror=true, halo=tiled)
        for M in (p -> D.control_solid(p; ηe=0.75, common...), p -> D.control_void(p; ηd=0.25, common...))
            gz = D.Zygote.gradient(M, x)[1]
            for idx in (CartesianIndex(1, 1), CartesianIndex(1, 11), CartesianIndex(n, 5), CartesianIndex(5, m))
                e = zeros(n, m); e[idx] = 1e-6
                fd = (M(x .+ e) - M(x .- e)) / 2e-6
                @test isapprox(gz[idx], fd; rtol=1e-5, atol=1e-12)
            end
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

        # A filter radius wide enough that convolvedcti tiles the cell: walls use the halo.
        prob_t = build_constraint_problem(foundry_mode=true, R_filter=150.0)
        sim_t = default_sim(prob_t.sim)
        p_t = 0.4 .+ 0.2 .* rand(length(prob_t.p))
        rel_ts, _, _ = check_constraint_grad((p, g) -> glc_solid(p, g; sim=sim_t, control=prob_t.control), p_t)
        @test rel_ts < TOL_RELATIVE
        rel_tv, _, _ = check_constraint_grad((p, g) -> glc_void(p, g; sim=sim_t, control=prob_t.control), p_t)
        @test rel_tv < TOL_RELATIVE
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

