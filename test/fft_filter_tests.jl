"""
2D filter selection tests for DistributedEmitterOpt.jl

On the 2D design grid (foundry mode) the only filter is the DCT-I conic filter. `Control(use_dct=false)` used to
select a periodic FFT convolution that no longer ran (it stopped on undefined helpers); it is now refused with an
`ArgumentError` that points to `use_dct=true`, in the design filter, its adjoint and both linewidth constraints.
The default path (`use_dct=true`) is checked against the DCT-I filter it calls.

Run with: julia --project test/fft_filter_tests.jl
"""

using DistributedEmitterOpt
using LinearAlgebra
using Random
using Test

# A foundry problem on a real OptimizationProblem: no mesh, no solve; `sim.grid` and `np` are set by hand.
struct FilterSelectionObjective <: DistributedEmitterOpt.ObjectiveFunction end

function filter_selection_sim(nx, ny)
    sim = DistributedEmitterOpt.Simulation()
    sim.grid = DistributedEmitterOpt.FoundryGrid(collect(range(0.0, nx - 1.0; length=nx)),
        collect(range(0.0, ny - 1.0; length=ny)), zeros(nx, ny), Any[])
    sim.np = nx * ny
    sim.nV = 3
    sim.objective_normalization = 1.0
    return sim
end

caught_error(f) = try
    f()
    nothing
catch e
    e
end

# A refused control on a real foundry problem of the same grid, and an eigen problem beside it.
filter_selection_control(; kwargs...) = Control(; R_filter=(3.0, 3.0, 3.0), use_dct=false, kwargs...)
filter_selection_problem(sim, root; kwargs...) =
    OptimizationProblem(MaxwellProblem(500.0; mat_design=2.0), FilterSelectionObjective(), sim, UmfpackSolver();
        foundry_mode=true, control=filter_selection_control(; kwargs...), root)
filter_selection_eigen_problem(sim, root; kwargs...) =
    EigenOptimizationProblem(EigenProblem(env=MaxwellProblem(500.0; mat_design=2.0).env), EigenObjective(), sim,
        UmfpackSolver(); foundry_mode=true, control=filter_selection_control(; kwargs...), root)

# A custom problem with only a control and its own objective, as the robust sidewall calls accept: no `foundry_mode`
# field. Its evaluation records the sidewall fields it sees.
struct SidewallRecordingProblem
    control::Control
    seen::Vector{Tuple{Bool,Float64,Float64}}
end
SidewallRecordingProblem(control::Control) = SidewallRecordingProblem(control, Tuple{Bool,Float64,Float64}[])
function DistributedEmitterOpt.objective_and_gradient!(g::Vector{Float64}, p::Vector{Float64},
                                                       prob::SidewallRecordingProblem)
    c = prob.control
    push!(prob.seen, (c.use_sidewall, c.sidewall_slope, c.sidewall_z0))
    isempty(g) || fill!(g, 1.0)
    return 1.3
end

@testset "2D filter selection" begin
    @testset "2D filter selection (use_dct)" begin
        D = DistributedEmitterOpt

        # Only `sim.grid` is read by the 2D filter and the 2D constraints: a unit-pitch node grid.
        unit_grid(nx, ny) = (grid=(x=collect(range(0.0, nx - 1.0; length=nx)), y=collect(range(0.0, ny - 1.0; length=ny))),)
        caught(f) = try
            f()
            nothing
        catch e
            e
        end

        rng = MersenneTwister(34)
        R = 3.0

        # The default is unchanged, and use_dct=false still constructs (3D designs set it by convention).
        @test Control().use_dct
        @test !Control(use_dct=false).use_dct

        # Odd and even node counts.
        for (nx, ny) in ((21, 15), (20, 16))
            sim = unit_grid(nx, ny)
            n = nx * ny
            p = 0.5 .+ 0.4 .* (rand(rng, n) .- 0.5)
            good = Control(R_filter=(R, R, R))
            bad = Control(R_filter=(R, R, R), use_dct=false)

            @testset "use_dct=false is refused ($nx × $ny)" begin
                calls = (
                    ("filter_grid", c -> D.filter_grid(p, sim, c)),
                    ("filter_grid_adjoint", c -> D.filter_grid_adjoint(p, sim, c)),
                    ("glc_solid", c -> D.glc_solid(p, Float64[]; sim, control=c)),
                    ("glc_solid", c -> D.glc_solid(p, zeros(n); sim, control=c)),
                    ("glc_void", c -> D.glc_void(p, Float64[]; sim, control=c)),
                    ("glc_void", c -> D.glc_void(p, zeros(n); sim, control=c)),
                )
                for (name, call) in calls
                    err = caught(() -> call(bad))
                    @test err isa ArgumentError
                    @test err isa ArgumentError && occursin("use_dct=true", err.msg)
                    @test err isa ArgumentError && occursin(name, err.msg)
                end

                # The constraint gradient buffer is left alone when the control is refused.
                g = fill(7.0, n)
                @test caught(() -> D.glc_solid(p, g; sim, control=bad)) isa ArgumentError
                @test caught(() -> D.glc_void(p, g; sim, control=bad)) isa ArgumentError
                @test all(==(7.0), g)

                # With filtering off nothing is convolved, so use_dct is not read: identity, as before.
                off = Control(use_filter=false, use_dct=false)
                @test D.filter_grid(p, sim, off) == p
                @test D.filter_grid_adjoint(p, sim, off) == p
            end

            @testset "default DCT-I path ($nx × $ny)" begin
                x = reshape(p, (nx, ny))
                h = D.conic_filter_dct(R, nx - 1.0, ny - 1.0, 1.0)
                @test D.filter_grid(p, sim, good) ≈ vec(D.convolvedcti(x; h)) atol = 1e-14
                @test D.filter_grid_adjoint(p, sim, good) ≈ vec(D.convolvedcti_adjoint(x; h)) atol = 1e-14

                # Forward/transpose dot-product identity and constant preservation.
                q = rand(rng, n)
                @test dot(D.filter_grid(p, sim, good), q) ≈ dot(p, D.filter_grid_adjoint(q, sim, good)) rtol = 1e-12
                @test D.filter_grid(fill(0.37, n), sim, good) ≈ fill(0.37, n) atol = 1e-13

                # Both linewidth constraints run on the default path, with and without a gradient.
                for glc in (D.glc_solid, D.glc_void)
                    g = zeros(n)
                    v = glc(p, g; sim, control=good)
                    @test isfinite(v)
                    @test all(isfinite, g)
                    @test glc(p, Float64[]; sim, control=good) ≈ v rtol = 1e-12
                end
            end
        end
    end

    # regression: the entry points refuse use_dct=false before they change the caller's problem.
    @testset "entry points refuse use_dct=false before changing the problem" begin
        D = DistributedEmitterOpt
        nx, ny = 11, 9
        sim = filter_selection_sim(nx, ny)
        bad = Control(R_filter=(3.0, 3.0, 3.0), use_dct=false)
        mktempdir() do root
            prob = OptimizationProblem(MaxwellProblem(500.0; mat_design=2.0), FilterSelectionObjective(), sim,
                UmfpackSolver(); foundry_mode=true, control=bad, root)
            # A problem that has run: a design, an objective, a history, a best record, an iteration count.
            prob.p .= range(0.1, 0.9; length=nx * ny)
            prob.g = 7.5
            prob.iteration = 9
            append!(prob.g_history, [1.0, 2.0, 3.0])
            prob.g_best = 3.0
            prob.p_best .= 0.25
            prob.control.β = 3.0
            pde0, pool0, R0 = prob.pde, prob.pool, prob.control.R_filter
            snap() = (copy(prob.p), prob.g, prob.iteration, copy(prob.g_history), prob.g_best, copy(prob.p_best),
                prob.control.β, prob.control.R_filter)
            before = snap()
            caught(f) = try
                f()
                nothing
            catch e
                e
            end
            p_design = copy(prob.p)

            # init_random!: refused before the random number generator is seeded or drawn from.
            Random.seed!(123)
            expected_draw = rand()
            Random.seed!(123)
            err = caught(() -> D.init_random!(prob; seed=5))
            @test err isa ArgumentError && occursin("init_random!", err.msg) && occursin("use_dct=true", err.msg)
            @test rand() == expected_draw
            @test snap() == before

            # fabrication_sweep: refused before it prints its announcement and before prob.control.R_filter is swept.
            out = joinpath(root, "fabrication_stdout.txt")
            err = open(out, "w") do io
                redirect_stdout(io) do
                    caught(() -> D.Analysis.fabrication_sweep(prob, p_design; root, save_plot=false, points=3))
                end
            end
            @test err isa ArgumentError && occursin("fabrication_sweep", err.msg)
            @test filesize(out) == 0
            @test snap() == before
            @test !isfile(joinpath(root, "fabrication.jld2"))

            # spectral_sweep: refused before prob.pde is replaced and the solver pool emptied. The pool holds
            # entries that emptying it would release.
            filter_factor, eigen_factor = Ref(:filter_factor), Ref(:eigen_factor)
            pool0.filter_cache.F_factor = filter_factor
            pool0.eigen_cache.A_factor = eigen_factor
            pool0.eigen_shift = 0.75
            err = caught(() -> D.Analysis.spectral_sweep(prob, p_design; root, save_plot=false, points=3))
            @test err isa ArgumentError && occursin("spectral_sweep", err.msg)
            @test prob.pde === pde0
            @test prob.pool === pool0
            @test pool0.filter_cache.F_factor === filter_factor
            @test pool0.eigen_cache.A_factor === eigen_factor
            @test pool0.eigen_shift == 0.75
            @test snap() == before
            @test !isfile(joinpath(root, "spectral.jld2"))

            # With filtering off nothing reaches the 2D filter, so the guard lets the call through.
            prob.control.use_filter = false
            @test D.require_dct_filter_problem(prob, "x"; use_constraints=false) === nothing
            @test caught(() -> D.require_dct_filter_problem(prob, "x"; use_constraints=true)) isa ArgumentError
            prob.control.use_filter = true

            # The eigen optimizer: refused before its first epoch sets prob.control.β.
            eprob = EigenOptimizationProblem(EigenProblem(env=prob.pde.env), EigenObjective(), sim, UmfpackSolver();
                foundry_mode=true, control=bad, root)
            eprob.control.β = 3.0
            eprob.p .= 0.5
            eprob.g = 4.5
            err = caught(() -> optimize!(eprob; β_schedule=[16.0], max_iter=2))
            @test err isa ArgumentError && occursin("optimize!", err.msg)
            @test eprob.control.β == 3.0
            @test eprob.p == fill(0.5, nx * ny)
            @test eprob.g == 4.5
            @test eprob.iteration == 0
        end
    end

    # regression: the eigen gradient refuses use_dct=false before it releases the cached eigen factor and shift.
    @testset "the eigen gradient refuses use_dct=false before releasing the eigen factor" begin
        D = DistributedEmitterOpt
        nx, ny = 11, 9
        mktempdir() do root
            eprob = filter_selection_eigen_problem(filter_selection_sim(nx, ny), root)
            factor = Ref(:eigen_factor)
            eprob.pool.eigen_cache.A_factor = factor
            eprob.pool.eigen_shift = 0.75
            eprob.p .= range(0.1, 0.9; length=nx * ny)
            eprob.g = 4.5
            p0 = copy(eprob.p)
            grad = fill(7.0, nx * ny)
            err = caught_error(() -> D.objective_and_gradient!(grad, copy(p0), eprob))
            @test err isa ArgumentError && occursin("objective_and_gradient!", err.msg) &&
                  occursin("use_dct=true", err.msg)
            @test eprob.pool.eigen_cache.A_factor === factor
            @test eprob.pool.eigen_shift == 0.75
            @test all(==(7.0), grad)
            @test eprob.p == p0
            @test eprob.g == 4.5
        end
    end

    # regression: both robust sidewall calls refuse use_dct=false on the package's problems before they set the
    # caller's sidewall fields.
    @testset "robust sidewall calls refuse use_dct=false before setting the sidewall fields" begin
        D = DistributedEmitterOpt
        nx, ny = 11, 9
        n = nx * ny
        p = fill(0.5, n)
        samples = [(0.3, 5.0), (-0.2, 1.0)]
        fields(c) = (c.use_sidewall, c.sidewall_slope, c.sidewall_z0)

        mktempdir() do root
            sim = filter_selection_sim(nx, ny)
            # Eigen problem: its gradient would release the cached factor; the refusal names the sidewall call,
            # so it came before any sample was set or evaluated.
            eprob = filter_selection_eigen_problem(sim, root)
            factor = Ref(:eigen_factor)
            eprob.pool.eigen_cache.A_factor = factor
            eprob.pool.eigen_shift = 0.75
            err = caught_error(() -> D.robust_sidewall_samples!(nothing, p, eprob, samples))
            @test err isa ArgumentError && occursin("robust_sidewall_samples!", err.msg) && occursin("use_dct=true", err.msg)
            grad = fill(7.0, n)
            err = caught_error(() -> D.robust_sidewall_objective!(grad, p, eprob, samples))
            @test err isa ArgumentError && occursin("robust_sidewall_samples!", err.msg)
            @test all(==(7.0), grad)
            @test eprob.pool.eigen_cache.A_factor === factor
            @test eprob.pool.eigen_shift == 0.75
            @test fields(eprob.control) == (false, 0.0, 0.0)
            # Maxwell problem: its evaluation would reach filter_grid after the fields were set; the refusal names
            # the sidewall call instead.
            prob = filter_selection_problem(sim, root)
            for f in (() -> D.robust_sidewall_samples!(nothing, p, prob, samples),
                      () -> D.robust_sidewall_objective!(zeros(n), p, prob, samples))
                err = caught_error(f)
                @test err isa ArgumentError && occursin("robust_sidewall_samples!", err.msg)
                @test fields(prob.control) == (false, 0.0, 0.0)
            end
        end
    end

    # regression: a custom problem with only a control and its own objective (no foundry_mode field) is evaluated by
    # both robust sidewall calls, whatever its control says; only the package's problem types are checked.
    @testset "robust sidewall calls evaluate a custom control-only problem" begin
        D = DistributedEmitterOpt
        n = 4
        p = fill(0.5, n)
        @test !hasfield(SidewallRecordingProblem, :foundry_mode)
        for control in (Control(), filter_selection_control(), filter_selection_control(use_filter=false))
            prob = SidewallRecordingProblem(control)
            gs = D.robust_sidewall_samples!(nothing, p, prob, [(0.3, 5.0), (-0.2, 1.0)])
            @test gs == [1.3, 1.3]
            @test prob.seen == [(true, 0.3, 5.0), (true, -0.2, 1.0)]
            @test (control.use_sidewall, control.sidewall_slope, control.sidewall_z0) == (false, 0.0, 0.0)

            prob = SidewallRecordingProblem(control)
            grad = zeros(n)
            g, gs, w = D.robust_sidewall_objective!(grad, p, prob, [(0.3, 5.0)])
            @test gs == [1.3] && w == [1.0]
            @test g == 1.3
            @test grad == ones(n)
            @test prob.seen == [(true, 0.3, 5.0)]
            @test (control.use_sidewall, control.sidewall_slope, control.sidewall_z0) == (false, 0.0, 0.0)
        end
    end

    # regression: calls that evaluate nothing still return with use_dct=false, as they did before the refusal.
    @testset "calls that evaluate nothing still return with use_dct=false" begin
        D = DistributedEmitterOpt
        nx, ny = 11, 9
        mktempdir() do root
            sim = filter_selection_sim(nx, ny)
            prob = filter_selection_problem(sim, root)
            p_design = fill(0.5, nx * ny)
            R0 = prob.control.R_filter
            λs, gs = D.Analysis.spectral_sweep(prob, p_design; root, save_plot=false, points=0)
            @test isempty(λs) && isempty(gs)
            Rs, gs = D.Analysis.fabrication_sweep(prob, p_design; root, save_plot=false, points=0)
            @test isempty(Rs) && isempty(gs)
            @test prob.control.R_filter == R0

            eprob = filter_selection_eigen_problem(sim, root)
            eprob.p .= 0.5
            eprob.control.β = 3.0
            g, p_out = optimize!(eprob; β_schedule=Float64[], max_iter=2)
            @test g == 0.0
            @test p_out == eprob.p && p_out !== eprob.p
            @test eprob.control.β == 3.0

            # One point is work: refused.
            @test caught_error(() -> D.Analysis.fabrication_sweep(prob, p_design; root, save_plot=false,
                points=1, range_R=0.0)) isa ArgumentError
        end
    end

    # regression: visualize_results refuses use_dct=false before it prints its announcement or writes a file.
    @testset "visualize_results refuses use_dct=false before printing or writing" begin
        D = DistributedEmitterOpt
        nx, ny = 11, 9
        mktempdir() do root
            prob = filter_selection_problem(filter_selection_sim(nx, ny), root)
            out = joinpath(root, "stdout.txt")
            vis = joinpath(root, "vis")
            mkdir(vis)
            err = open(out, "w") do io
                redirect_stdout(io) do
                    caught_error(() -> D.Analysis.visualize_results(prob, fill(0.5, nx * ny); root=vis))
                end
            end
            @test err isa ArgumentError && occursin("visualize_results", err.msg) && occursin("use_dct=true", err.msg)
            @test filesize(out) == 0
            @test isempty(readdir(vis))
        end
    end
end
