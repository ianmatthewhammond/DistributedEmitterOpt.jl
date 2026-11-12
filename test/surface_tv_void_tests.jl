# Gradient tests for the :tv_void surface measure (src/Objectives/SurfaceShell.jl).
#
# Run with: julia --project test/surface_tv_void_tests.jl
include(joinpath(@__DIR__, "gradient_common.jl"))

@testset "Surface :tv_void gradient tests" begin
    @testset "3D tv_void (elastic, isotropic)" begin
        println("\n=== 3D tv_void ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, g0 = test_gradient(prob, p0)
        @test isfinite(g0) && g0 > 0
        @test rel_err < TOL_RELATIVE
    end

    # Damage cases must be ACTIVE: with the helper defaults (γ=1, E_threshold=10) the test fields sit
    # below threshold and dmg ≡ 1 to all digits, so such a test only exercises the code path. Here
    # E_threshold=0, γ=4 suppresses the objective by a large factor, which the test asserts.
    @testset "3D tv_void + active damage model (elastic)" begin
        println("\n=== 3D tv_void + active damage (elastic) ===")
        prob_off = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void)
        prob     = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void,
                                      use_damage=true, γ_damage=4.0, E_threshold=0.0)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        g_off = objective_and_gradient!(Float64[], copy(p0), prob_off)
        rel_err, g_dmg = test_gradient(prob, p0)
        println("  damage suppression g_dmg/g_off = $(g_dmg / g_off)")
        @test isfinite(g_dmg) && g_dmg > 0
        @test g_dmg < 0.9 * g_off          # damage is actually doing something
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D tv_void + active damage model (inelastic, pump != emission)" begin
        println("\n=== 3D tv_void + active damage (inelastic) ===")
        prob_off = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void,
                                      λ_pump=532.0, λ_emission=600.0)
        prob     = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void,
                                      λ_pump=532.0, λ_emission=600.0, use_damage=true, γ_damage=4.0, E_threshold=0.0)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        g_off = objective_and_gradient!(Float64[], copy(p0), prob_off)
        rel_err, g_dmg = test_gradient(prob, p0)
        println("  damage suppression g_dmg/g_off = $(g_dmg / g_off)")
        @test g_dmg < 0.9 * g_off
        @test rel_err < TOL_RELATIVE
    end

    @testset "3D tv_void inelastic (pump != emission)" begin
        println("\n=== 3D tv_void inelastic ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void,
                                  λ_pump=532.0, λ_emission=600.0)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, _ = test_gradient(prob, p0)
        @test rel_err < TOL_RELATIVE
    end

    @testset "2D foundry tv_void (:current projection path, FEM SSP)" begin
        # NB: build_test_problem defaults to foundry_projection_mode=:current (pt is an FEFunction).
        println("\n=== 2D foundry tv_void (:current) ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true, surface_mode=:tv_void)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        rel_err, g0 = test_gradient(prob, p0; δ=1e-8)
        @test isfinite(g0) && g0 > 0
        @test rel_err < TOL_RELATIVE
    end

    # Stateful cache regression across flat normalization and changing designs.
    # With foundry_projection_mode=:legacy, every evaluation writes
    # sim.grid.params in place and hands the objective a closure `pt`. optimize! first
    # evaluates p ≡ 0 for flat_substrate_norm. The old ShellCache keyed χ_s on `pt` identity;
    # successive legacy closures are ===-equal, so χ_s stayed at the p ≡ 0 shell all run while
    # the adjoint differentiated a moving shell. This testset replays that exact sequence.
    @testset "legacy foundry: stateful flat→design sequence (cache + central FD)" begin
        println("\n=== legacy foundry stateful sequence ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                  surface_mode=:tv_void, foundry_projection_mode=:legacy)
        @test prob.control.foundry_projection_mode == :legacy
        np = length(prob.p)
        rng = MersenneTwister(20260912)

        # χ_s DOF values of the cached shell (helper added with the fix; fall back to the raw cache
        # so this file also runs — and fails — against pre-fix sources)
        chi_s = obj -> isdefined(DistributedEmitterOpt, :shell_chi_s_values) ?
            DistributedEmitterOpt.shell_chi_s_values(obj) :
            copy(DistributedEmitterOpt.get_free_dof_values(obj.shell_cache[].χs))

        # 1. flat normalization call, exactly as flat_substrate_norm does
        g_flat = objective_and_gradient!(Float64[], zeros(np), prob)
        χs_flat = chi_s(prob.objective)
        @test isfinite(g_flat) && g_flat > 0
        @test χs_flat !== nothing

        # 2. a SUBSTANTIALLY different design: a smooth blob of material on the grid
        sim0 = DistributedEmitterOpt.default_sim(prob.sim)
        nx, ny = length(sim0.grid.x), length(sim0.grid.y)
        xs, ys = sim0.grid.x, sim0.grid.y
        cx, cy = (xs[1] + xs[end]) / 2, (ys[1] + ys[end]) / 2
        w = 0.35 * (xs[end] - xs[1])
        P1 = [clamp(0.9 * exp(-((x - cx)^2 + (y - cy)^2) / w^2) + 0.05 * (rand(rng) - 0.5), 0.0, 1.0)
              for x in xs, y in ys]
        p1 = vec(P1)
        @test maximum(abs.(p1)) > 0.5          # really is a different design from p ≡ 0

        grad1 = zeros(np)
        g1 = objective_and_gradient!(grad1, copy(p1), prob)
        χs_1 = chi_s(prob.objective)
        println("  g(flat) = $g_flat   g(design) = $g1")

        # 3. the shell geometry MUST have moved with the design
        rel_move = norm(χs_1 .- χs_flat) / norm(χs_flat)
        println("  |χ_s(design) − χ_s(flat)| / |χ_s(flat)| = $rel_move")
        @test rel_move > 1e-2
        @test isfinite(g1) && g1 > 0 && !isapprox(g1, g_flat; rtol=1e-6)

        # 4. order independence: a FRESH problem that never saw p ≡ 0 gives the same objective
        prob_fresh = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                        surface_mode=:tv_void, foundry_projection_mode=:legacy)
        g1_fresh = objective_and_gradient!(Float64[], copy(p1), prob_fresh)
        println("  g(design | fresh objective) = $g1_fresh")
        @test isapprox(g1, g1_fresh; rtol=1e-10)

        # 5. central FD vs adjoint AT THE SECOND DESIGN, after the flat call: several coordinates
        #    (the largest-|∇g| ones, so the relative error is meaningful) + one dense direction.
        idx = sortperm(abs.(grad1); rev=true)[1:4]
        dirs = Vector{Float64}[]
        for i in idx
            e = zeros(np); e[i] = 1.0; push!(dirs, e)
        end
        push!(dirs, randn(rng, np) ./ sqrt(np))          # dense, unit-ish
        res = central_fd_check(prob, p1, grad1, dirs; δ=1e-5)
        for (k, (adj, fd, rel)) in enumerate(res)
            label = k <= length(idx) ? "coord $(idx[k])" : "dense"
            println("  $label: adj = $adj  fd = $fd  rel = $rel")
            @test rel < TOL_RELATIVE
        end

        # 6. and the gradient at the base point is still the same function after all those FD calls
        grad1b = zeros(np)
        g1b = objective_and_gradient!(grad1b, copy(p1), prob)
        @test g1b == g1
        @test grad1b == grad1
    end

    @testset "legacy foundry: pt closure is a snapshot, not an alias of sim.grid.params" begin
        println("\n=== legacy closure snapshot ===")
        prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                  surface_mode=:tv_void, foundry_projection_mode=:legacy)
        sim0 = DistributedEmitterOpt.default_sim(prob.sim)
        np = length(prob.p)
        pt_a = DistributedEmitterOpt.build_foundry_pf(fill(0.2, np), sim0, prob.control)
        r = DistributedEmitterOpt.VectorValue(sim0.grid.x[end ÷ 2], sim0.grid.y[end ÷ 2], 0.0)
        va = pt_a(r)
        pt_b = DistributedEmitterOpt.build_foundry_pf(fill(0.8, np), sim0, prob.control)
        @test pt_a(r) == va                      # old closure unaffected by the new write
        @test pt_b(r) ≈ 0.8 atol=1e-12
        @test va ≈ 0.2 atol=1e-12
    end

    @testset "tv_void: finite value and gradient on flat designs (ε_grad floor)" begin
        println("\n=== tv_void flat-region finiteness ===")
        for (mode, tag) in ((:legacy, "legacy"), (:current, "current"))
            prob = build_test_problem(foundry_mode=true, flag_volume=false, flag_surface=true,
                                      surface_mode=:tv_void, foundry_projection_mode=mode)
            np = length(prob.p)
            for (pflat, name) in ((zeros(np), "p≡0"), (ones(np), "p≡1"))
                grad = zeros(np)
                g = objective_and_gradient!(grad, pflat, prob)
                println("  [$tag] $name: g = $g  |∇g|_∞ = $(maximum(abs.(grad)))")
                @test isfinite(g) && g > 0
                @test all(isfinite, grad)
            end
        end
    end

    @testset "flat substrate baseline is nonzero for tv_void" begin
        println("\n=== tv_void baseline at p ≡ 0 ===")
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void)
        g_flat = objective_and_gradient!(Float64[], zeros(length(prob.p)), prob)
        println("  g(p≡0) = $g_flat")
        @test g_flat > 0
        prob_band = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:band)
        g_band = objective_and_gradient!(Float64[], zeros(length(prob_band.p)), prob_band)
        println("  band g(p≡0) = $g_band  (expected 0)")
        @test abs(g_band) < 1e-12
    end

    @testset "fail-closed guards" begin
        println("\n=== tv_void guards ===")
        # invalid parameters are rejected at construction, not silently ignored
        @test_throws ArgumentError SERSObjective(surface=true, surface_mode=:typo)
        @test_throws ArgumentError SERSObjective(surface_mode=:tv_void, t_shell=0.0)
        @test_throws ArgumentError SERSObjective(surface_mode=:tv_void, t_shell=Inf)
        @test_throws ArgumentError SERSObjective(surface_mode=:tv_void, ε_grad=-1.0)
        # ε_grad = 0 is a 0/0 in û wherever χ_s is flat: rejected for :tv_void, still allowed (unused) for :band
        @test_throws ArgumentError SERSObjective(surface_mode=:tv_void, ε_grad=0.0)
        @test_throws ArgumentError SERSObjective(surface_mode=:tv_void, ε_grad=NaN)
        @test SERSObjective(surface_mode=:band, ε_grad=0.0).ε_grad == 0.0

        # a mesh without a "Substrate" region must not silently drop the substrate from χ
        prob = build_test_problem(foundry_mode=false, flag_volume=false, flag_surface=true, surface_mode=:tv_void)
        p0 = 0.4 .+ 0.2 .* rand(length(prob.p))
        g_ok = objective_and_gradient!(Float64[], copy(p0), prob)
        sim0 = DistributedEmitterOpt.default_sim(prob.sim)   # prob.sim is a SimulationBundle
        @test sim0.dΩ_substrate !== nothing
        sim0.dΩ_substrate = nothing; sim0.Ω_substrate = nothing
        prob.objective.shell_cache[] = nothing
        @test_throws ErrorException DistributedEmitterOpt.validate_objective(prob.objective, prob.sim)
        @test_throws ErrorException OptimizationProblem(prob.pde, prob.objective, prob.sim, UmfpackSolver();
                                                        foundry_mode=prob.foundry_mode, control=prob.control)
        @test_throws ErrorException objective_and_gradient!(Float64[], copy(p0), prob)
        # ... unless a substrate-free geometry is requested explicitly
        obj_free = SERSObjective(αₚ=prob.objective.αₚ, volume=false, surface=true, surface_mode=:tv_void,
                                 t_shell=prob.objective.t_shell, require_substrate=false)
        prob_free = OptimizationProblem(prob.pde, obj_free, prob.sim, UmfpackSolver();
                                        foundry_mode=prob.foundry_mode, control=prob.control)
        g_free = objective_and_gradient!(Float64[], copy(p0), prob_free)
        println("  g with substrate = $g_ok   without (explicit) = $g_free")
        @test isfinite(g_free) && g_free > 0 && g_free != g_ok
    end
end
