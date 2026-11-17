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

    @testset "2D DCT filter: mirror walls for a kernel of any size" begin
        D = DistributedEmitterOpt
        res = 1.0
        field(sx, sy) = [0.5 + 0.3 * sin(0.7i + 0.2) * cos(0.4j - 0.1) + 0.1 * cos(1.3i * j / sy) for i in 1:sx, j in 1:sy]

        # Reference, independent of any transform: the conic kernel (1 − r/R for r < R, as conic_filter builds it)
        # summed directly over the cell continued by its mirror images (reflected inner nodes, wall nodes once).
        function mirror_conic(x, R)
            sx, sy = size(x)
            fold(i, n) = (m = mod(i - 1, 2 * (n - 1)); m <= n - 1 ? m + 1 : 2 * (n - 1) - m + 1)
            k = ceil(Int, R * res)
            w = [(di, dj, 1 - hypot(di, dj) / (R * res)) for di in -k:k for dj in -k:k if hypot(di, dj) < R * res]
            total = sum(t[3] for t in w)
            return [sum(c * x[fold(i + di, sx), fold(j + dj, sy)] for (di, dj, c) in w) / total for i in 1:sx, j in 1:sy]
        end

        # Kernels that fit the cell, that exceed half of it along x, along y or along both, and that exceed the
        # whole cell (several mirror images overlap). Odd and even node counts. The cases above half a cell width
        # are the ones the filter used to tile by repetition; they fail on the code before this fix.
        cases = ((9, 21, 2.0), (9, 21, 5.0), (8, 20, 5.0), (21, 9, 5.0), (20, 8, 5.0), (7, 6, 4.0), (7, 6, 9.0), (6, 9, 13.5), (5, 5, 30.0),
            # two-node cells (the smallest a DCT-I allows), with kernels that fold over several periods
            (2, 7, 3.0), (5, 2, 4.0), (2, 2, 2.5), (3, 2, 7.5))
        for (sx, sy, R) in cases
            x = field(sx, sy)
            h = D.conic_filter(R, sx - 1.0, sy - 1.0, res)
            f = D.convolvedcti(x; h)
            @test size(f) == (sx, sy)
            @test f ≈ mirror_conic(x, R) atol = 1e-12
            # No axis is tiled and no halo is added, whatever the kernel.
            @test D.dcti_tiles((sx, sy), size(h)) == (1, 1)
            @test D.dcti_tiled((sx, sy), size(h)) == (false, false)
            @test D.convolvedcti(x; h, halo=true) == f
            # The adjoint is the transpose.
            a, b = rand(sx, sy), rand(sx, sy)
            @test dot(D.convolvedcti(a; h), b) ≈ dot(a, D.convolvedcti_adjoint(b; h)) rtol = 1e-12
            # A constant stays constant: the kernel is normalized after folding.
            @test D.convolvedcti(fill(0.37, sx, sy); h) ≈ fill(0.37, sx, sy) atol = 1e-13
        end

        # The forward pipeline of the filter before the fix (filter revision 1), copied literally: tile the cell by
        # repetition, pad the kernel to the tiled size, filter, return the middle copy.
        function revision1_convolvedcti(x; h)
            sx, sy = size(x)
            npx, npy = D.legacy_dcti_tiles((sx, sy), size(h))
            xt = repeat(x, outer=(npx, npy))
            xt = D.edgepaddcti(xt, ((0, 0), (0, 0)))
            hp = D.properpaddcti(h, (npx * sx, npy * sy))
            freq = D.DCT_I(hp)
            freq = freq ./ maximum(freq)
            return D.centeredfft(D.iDCT_I(D.DCT_I(xt) .* freq), (sx, sy))
        end

        # A kernel that fits the cell (2k − 1 ≤ n on both axes) is filtered as before the fix: the same arrays go
        # through the same transforms. Compared to round-off, not bit for bit: FFTW plans are made afresh for each
        # array, and their choice may depend on memory alignment.
        for (sx, sy, R) in ((9, 21, 2.0), (31, 23, 5.0), (12, 12, 3.0))
            x = field(sx, sy)
            h = D.conic_filter(R, sx - 1.0, sy - 1.0, res)
            @test all(2 .* size(h) .- 1 .<= (sx, sy))
            @test D.legacy_dcti_tiles((sx, sy), size(h)) == (1, 1)
            freq = D.DCT_I(D.properpaddcti(h, (sx, sy)))
            @test D.convolvedcti(x; h) ≈ D.iDCT_I(D.DCT_I(x) .* (freq ./ maximum(freq))) atol = 1e-14
            @test D.convolvedcti(x; h) ≈ revision1_convolvedcti(x; h) atol = 1e-14
        end

        # A kernel that does not fit: revision 1 tiled the cell (the rule kept in legacy_dcti_tiles) and was wrong
        # by far more than round-off; that is the fault this test set guards against.
        for (sx, sy, R, tiles) in ((9, 21, 5.0, (3, 1)), (7, 6, 9.0, (3, 5)))
            x = field(sx, sy)
            h = D.conic_filter(R, sx - 1.0, sy - 1.0, res)
            @test D.legacy_dcti_tiles((sx, sy), size(h)) == tiles
            @test maximum(abs, revision1_convolvedcti(x; h) .- mirror_conic(x, R)) > 1e-3
        end
        @test D.legacy_dcti_tiles((2, 2), size(D.conic_filter(2.5, 1.0, 1.0, res))) == (3, 3)

        # A DCT-I needs two nodes along each axis; one node is refused instead of read past the kernel.
        @test_throws ArgumentError D.convolvedcti(rand(1, 5); h=D.conic_filter(2.0, 4.0, 4.0, res))
        @test_throws ArgumentError D.convolvedcti(rand(5, 1); h=D.conic_filter(2.0, 4.0, 4.0, res))
        @test_throws ArgumentError D.convolvedcti_adjoint(rand(1, 1); h=D.conic_filter(2.0, 4.0, 4.0, res))
        @test_throws ArgumentError D.fold_axis_dcti(rand(4, 4), 1, 1)

        # Mirror consistency: a cell doubled by reflection across a wall (the wall node once) filters to the
        # doubled filtered cell. Repetition instead of reflection breaks this for a kernel above half the cell.
        for (sx, sy, R) in ((9, 21, 5.0), (21, 9, 5.0), (7, 6, 9.0))
            x = field(sx, sy)
            h = D.conic_filter(R, sx - 1.0, sy - 1.0, res)
            f = D.convolvedcti(x; h)
            xd = vcat(x, reverse(x[1:end-1, :]; dims=1))
            fd = D.convolvedcti(xd; h=D.conic_filter(R, size(xd, 1) - 1.0, sy - 1.0, res))
            @test fd[1:sx, :] ≈ f atol = 1e-12
            @test fd ≈ vcat(f, reverse(f[1:end-1, :]; dims=1)) atol = 1e-12
        end

        # The wall stencil of the length-scale indicator is the mirror one on every axis, and Zygote's gradient
        # through the filter with a long kernel matches central finite differences.
        n, m = 9, 21
        x = field(n, m)
        h = D.conic_filter(5.0, n - 1.0, m - 1.0, res)
        f = D.convolvedcti(x; h)
        g = D.gradient_sq(f, res; mirror=true, halo=D.dcti_tiled((n, m), size(h)))
        fx = vcat(f[2:2, :], f, f[end-1:end-1, :])
        fy = hcat(f[:, 2:2], f, f[:, end-1:end-1])
        @test g ≈ ((fx[3:end, :] .- fx[1:end-2, :]) .* (res / 2)) .^ 2 .+ ((fy[:, 3:end] .- fy[:, 1:end-2]) .* (res / 2)) .^ 2 atol = 1e-14
        # Both penalties are active on this field and must say so before their gradients are compared: the solid
        # penalty min(f − ηe, 0)² acts where the filtered field is below ηe, the void penalty min(ηd − f, 0)² where
        # it is above ηd. The filtered fields below lie between 0.38 and 0.70, inside (ηd, ηe) = (0.25, 0.75).
        # Three kernels: above half the cell along x only; longer than the whole cell on both axes, so several
        # mirror images fold onto each node (7 × 6 nodes, R = 9); and the same on 6 × 9 nodes with R = 13.5.
        thr = p -> 1 ./ (1 .+ exp.(-8 .* (p .- 0.5)))
        for (n, m, R) in ((9, 21, 5.0), (7, 6, 9.0), (6, 9, 13.5))
            x = field(n, m)
            h = D.conic_filter(R, n - 1.0, m - 1.0, res)
            filt = p -> D.convolvedcti(p; h, halo=true)
            f = filt(x)
            @test 0.25 < minimum(f) && maximum(f) < 0.75
            common = (; c0=64 * R^2, filter_f=filt, threshold_f=thr, resolution=res, mirror=true, halo=(false, false))
            for M in (p -> D.control_solid(p; ηe=0.75, common...), p -> D.control_void(p; ηd=0.25, common...))
                gz = D.Zygote.gradient(M, x)[1]
                @test M(x) > 1e-3
                @test maximum(abs, gz) > 1e-4
                for idx in (CartesianIndex(1, 1), CartesianIndex(1, m - 1), CartesianIndex(n, 3), CartesianIndex(4, m))
                    e = zeros(n, m); e[idx] = 1e-6
                    fd = (M(x .+ e) - M(x .- e)) / 2e-6
                    @test abs(fd) > 1e-6
                    @test isapprox(gz[idx], fd; rtol=1e-5, atol=1e-12)
                end
            end
        end
    end

    @testset "2D Foundry Constraints" begin
        prob = build_constraint_problem(foundry_mode=true)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))

        # Use default_sim to get Simulation from SimulationBundle
        sim0 = default_sim(prob.sim)

        rel_s, fd_s, adj_s = check_constraint_grad(
            (p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), p0
        )
        @test rel_s < TOL_RELATIVE

        rel_v, fd_v, adj_v = check_constraint_grad(
            (p, g) -> glc_void(p, g; sim=sim0, control=prob.control), p0
        )
        @test rel_v < TOL_RELATIVE
        # Not a comparison of zeros: both penalties and both directional derivatives are nonzero for a design
        # between 0.4 and 0.6 (the solid penalty acts below η_erosion, the void penalty above η_dilation).
        @test glc_solid(p0, Float64[]; sim=sim0, control=prob.control) + prob.control.b1 > 0
        @test glc_void(p0, Float64[]; sim=sim0, control=prob.control) + prob.control.b1 > 0
        @test fd_s != 0 && adj_s != 0 && fd_v != 0 && adj_v != 0
        # This radius fits the cell, so checkpoints written before the kernel was folded stay valid ...
        @test !DistributedEmitterOpt.dcti_fold_changes_filter(sim0, prob.control)

        # A filter radius above half the cell: the kernel is folded onto the cell, walls stay mirrors.
        prob_t = build_constraint_problem(foundry_mode=true, R_filter=150.0)
        sim_t = default_sim(prob_t.sim)
        p_t = 0.4 .+ 0.2 .* rand(length(prob_t.p))
        rel_ts, fd_ts, adj_ts = check_constraint_grad((p, g) -> glc_solid(p, g; sim=sim_t, control=prob_t.control), p_t)
        @test rel_ts < TOL_RELATIVE
        rel_tv, fd_tv, adj_tv = check_constraint_grad((p, g) -> glc_void(p, g; sim=sim_t, control=prob_t.control), p_t)
        @test rel_tv < TOL_RELATIVE
        @test glc_solid(p_t, Float64[]; sim=sim_t, control=prob_t.control) + prob_t.control.b1 > 0
        @test glc_void(p_t, Float64[]; sim=sim_t, control=prob_t.control) + prob_t.control.b1 > 0
        @test fd_ts != 0 && adj_ts != 0 && fd_tv != 0 && adj_tv != 0
        # ... and this one does not: the old filter tiled this cell, so its optimizer state must not be resumed.
        @test DistributedEmitterOpt.dcti_fold_changes_filter(sim_t, prob_t.control)
        # Whether that matters for a run: only if the filter is called. The objective calls it with use_filter;
        # the length-scale constraints call it always.
        matters = DistributedEmitterOpt._filter_revision_matters
        was = prob_t.control.use_filter
        prob_t.control.use_filter = true
        @test matters(prob_t; use_constraints=false) && matters(prob_t; use_constraints=true) && matters(prob_t)
        prob_t.control.use_filter = false
        @test !matters(prob_t; use_constraints=false)      # filter off, no constraints: old checkpoints stay valid
        @test matters(prob_t; use_constraints=true)        # filter off, constraints on: they filter
        @test matters(prob_t)                              # a caller that does not say is treated as constrained
        prob_t.control.use_filter = was
        # A kernel that fits never matters.
        @test !matters(prob; use_constraints=true) && !matters(prob; use_constraints=false)
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

