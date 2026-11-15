# Threaded cell loops and split Maxwell assembly (src/Utils/ThreadedAssembly.jl) against the
# serial path with every matrix term assembled from scratch. The two may differ only by the order
# of floating-point additions. THREAD_CHUNKS forces chunking, so the chunked code runs even when
# Julia has one thread; run with `julia -t N` to exercise real concurrency as well.
using DistributedEmitterOpt
using Gridap
using LinearAlgebra
using Random
using SparseArrays
using Test

isdefined(@__MODULE__, :build_test_problem) || include("gradient_common.jl")

const TA = DistributedEmitterOpt

function with_assembly(f; threaded::Bool, split::Bool, chunks::Int=0, min_cells::Int=2000)
    old = (TA.THREADED[], TA.SPLIT_ASSEMBLY[], TA.THREAD_CHUNKS[], TA.THREAD_MIN_CELLS[])
    TA.THREADED[], TA.SPLIT_ASSEMBLY[], TA.THREAD_CHUNKS[], TA.THREAD_MIN_CELLS[] = threaded, split, chunks, min_cells
    TA.clear_assembly_caches!()
    try
        return f()
    finally
        TA.THREADED[], TA.SPLIT_ASSEMBLY[], TA.THREAD_CHUNKS[], TA.THREAD_MIN_CELLS[] = old
        TA.clear_assembly_caches!()
    end
end

"Objective and gradient at each design, on a fresh problem so no solver state is shared."
function evaluate_designs(build, designs)
    prob = build()
    out = Tuple{Float64,Vector{Float64}}[]
    for p in designs
        grad = zeros(length(p))
        g = objective_and_gradient!(grad, p, prob)
        push!(out, (g, copy(grad)))
    end
    return out
end

function designs_for(build; n=3)
    prob = build()
    rng = MersenneTwister(7)
    np = length(prob.p)
    return [clamp.(0.5 .+ 0.4 .* (rand(rng, np) .- 0.5), 0.0, 1.0) for _ in 1:n]
end

@testset "Threaded assembly" begin
    @testset "chunk ranges" begin
        for n in (1, 2, 7, 100, 2001), k in (1, 2, 3, 8, 48)
            r = TA.chunk_ranges(n, k)
            @test length(r) == min(n, k)
            @test first(first(r)) == 1 && last(last(r)) == n
            @test all(!isempty, r)
            @test all(first(r[i+1]) == last(r[i]) + 1 for i in 1:length(r)-1)
        end
    end

    cases = [
        ("3D DOF, volume, elastic", () -> build_test_problem()),
        ("3D DOF, inelastic, anisotropic", () -> build_test_problem(λ_emission=560.0, isotropic=false)),
        ("foundry, surface shell", () -> build_test_problem(foundry_mode=true, flag_volume=false,
            flag_surface=true, surface_mode=:tv_void)),
        ("foundry, mixed outputs", () -> build_test_problem(foundry_mode=true, complex_config=true, λ_emission=560.0)),
        ("foundry, legacy projection (design field is a plain function)",
            () -> build_test_problem(foundry_mode=true, foundry_projection_mode=:legacy)),
    ]

    @testset "objective and gradient: $name" for (name, build) in cases
        designs = designs_for(build)
        ref = with_assembly(() -> evaluate_designs(build, designs); threaded=false, split=false)
        # Chunked loops only, split assembly only, and both: three designs each, so a plan is built
        # and checked against the ordinary assembly, and then used unchecked.
        for (threaded, split) in ((true, false), (false, true), (true, true))
            got = with_assembly(() -> evaluate_designs(build, designs);
                threaded, split, chunks=5, min_cells=1)
            for ((g0, d0), (g1, d1)) in zip(ref, got)
                @test isapprox(g1, g0; rtol=1e-10)
                @test norm(d1 - d0) <= 1e-8 * norm(d0)
            end
        end
    end

    @testset "split matrix equals the ordinary matrix" begin
        prob = build_test_problem(λ_emission=560.0)
        sim0 = default_sim(prob.sim)
        rng = MersenneTwister(11)
        with_assembly(threaded=true, split=true, chunks=4, min_cells=1) do
            for fc in all_configs(prob.pde)
                sim_fc = sim_for(prob.sim, fc)
                phys = TA.build_phys_params(fc, prob.pde.env, sim_fc; α=prob.pde.α_loss)
                for k in 1:3
                    pf = FEFunction(sim0.Pf, rand(rng, num_free_dofs(sim0.Pf)))
                    pt = TA.project_ssp(pf, prob.control, sim0.Ω)
                    A = TA.assemble_maxwell(pt, sim_fc, phys)
                    B = TA.assemble_maxwell_direct(pt, sim_fc, phys)
                    @test A.colptr == B.colptr && A.rowval == B.rowval
                    @test norm(A.nzval - B.nzval) <= 1e-13 * norm(B.nzval)
                end
            end
            @test length(TA._PLANS) == length(all_configs(prob.pde))
            @test all(p -> p.usable && p.verified && p.ncalls == 3, TA._PLANS)
            @test TA.SPLIT_ASSEMBLY[]
        end
    end

    @testset "concurrent calls share a plan safely" begin
        prob = build_test_problem()
        sim0 = default_sim(prob.sim)
        fc = first(all_configs(prob.pde))
        sim_fc = sim_for(prob.sim, fc)
        phys = TA.build_phys_params(fc, prob.pde.env, sim_fc; α=prob.pde.α_loss)
        rng = MersenneTwister(5)
        pts = [TA.project_ssp(FEFunction(sim0.Pf, rand(rng, num_free_dofs(sim0.Pf))), prob.control, sim0.Ω) for _ in 1:6]
        refs = with_assembly(threaded=false, split=false) do
            [TA.assemble_maxwell_direct(pt, sim_fc, phys) for pt in pts]
        end
        with_assembly(threaded=true, split=true, chunks=3, min_cells=1) do
            TA.assemble_maxwell(pts[1], sim_fc, phys)          # builds and checks the plan
            got = Vector{Any}(undef, length(pts))
            for round in 1:5
                @sync for k in eachindex(pts)
                    Threads.@spawn (got[k] = TA.assemble_maxwell(pts[k], sim_fc, phys))
                end
                @test all(norm(got[k].nzval - refs[k].nzval) <= 1e-13 * norm(refs[k].nzval) for k in eachindex(pts))
            end
        end
    end

    @testset "no split result escapes a failed check" begin
        prob = build_test_problem()
        sim0 = default_sim(prob.sim)
        fc = first(all_configs(prob.pde))
        sim_fc = sim_for(prob.sim, fc)
        phys = TA.build_phys_params(fc, prob.pde.env, sim_fc; α=prob.pde.α_loss)
        rng = MersenneTwister(21)
        pts = [TA.project_ssp(FEFunction(sim0.Pf, rand(rng, num_free_dofs(sim0.Pf))), prob.control, sim0.Ω) for _ in 1:6]
        # An "ordinary assembly" that disagrees with the split one away from the zero design, and is
        # slow enough for the other tasks to arrive while the first is still checking.
        wrong = function (pt, sim, phys)
            A = TA.assemble_maxwell_direct(pt, sim, phys)
            pt isa Gridap.CellData.GenericCellField && return A        # the zero design of the plan
            sleep(0.2)
            return 2 * A
        end
        with_assembly(threaded=true, split=true, chunks=3, min_cells=1) do
            got = Vector{Any}(undef, length(pts))
            @sync for k in eachindex(pts)                              # cold: nothing built or checked yet
                Threads.@spawn (got[k] = TA.assemble_maxwell_split(pts[k], sim_fc, phys, wrong))
            end
            refs = [2 * TA.assemble_maxwell_direct(pt, sim_fc, phys) for pt in pts]
            # Each caller gets either the rejected check's reference or `nothing` (assemble directly);
            # never the split matrix that the check rejected.
            @test all(got[k] === nothing || norm(got[k].nzval - refs[k].nzval) <= 1e-13 * norm(refs[k].nzval) for k in eachindex(pts))
            @test count(!isnothing, got) == 1
            @test !only(TA._PLANS).usable
            @test TA.assemble_maxwell_split(pts[1], sim_fc, phys, wrong) === nothing
        end
    end

    @testset "a modified simulation gets a new plan" begin
        prob = build_test_problem()
        sim0 = default_sim(prob.sim)
        fc = first(all_configs(prob.pde))
        sim_fc = sim_for(prob.sim, fc)
        phys = TA.build_phys_params(fc, prob.pde.env, sim_fc; α=prob.pde.α_loss)
        pt = TA.project_ssp(FEFunction(sim0.Pf, rand(MersenneTwister(9), num_free_dofs(sim0.Pf))), prob.control, sim0.Ω)
        old_measure = sim_fc.dΩ_design
        with_assembly(threaded=true, split=true, chunks=3, min_cells=1) do
            try
                TA.assemble_maxwell(pt, sim_fc, phys)
                TA.assemble_maxwell(pt, sim_fc, phys)
                first_plan = only(TA._PLANS)
                sim_fc.dΩ_design = Measure(sim_fc.Ω_design, 2)      # a different quadrature: a different matrix
                B = TA.assemble_maxwell_direct(pt, sim_fc, phys)
                for _ in 1:2                                          # the checked first use, then an unchecked one
                    A = TA.assemble_maxwell(pt, sim_fc, phys)
                    @test norm(A.nzval - B.nzval) <= 1e-13 * norm(B.nzval)
                end
                @test only(TA._PLANS) !== first_plan
                @test only(TA._PLANS).usable && only(TA._PLANS).verified && only(TA._PLANS).ncalls == 2
                TA.invalidate_maxwell_cache!(default_pool(prob.pool))  # the hook for in-place changes
                @test isempty(TA._PLANS)
            finally
                sim_fc.dΩ_design = old_measure
            end
        end
    end

    @testset "an exception in a chunk arrives unwrapped" begin
        @test_throws TA.SplitUnsupported TA.foreach_chunk(TA.chunk_ranges(8, 4)) do i, _
            i == 3 && throw(TA.SplitUnsupported("probe"))
        end
        E = Matrix{ComplexF64}(undef, 4, 6)
        ragged = Gridap.Arrays.lazy_map(i -> ones(ComplexF64, isodd(i) ? 2 : 3, 2), collect(1:6))
        with_assembly(threaded=true, split=true, chunks=3, min_cells=1) do
            @test_throws TA.SplitUnsupported TA._element_matrices!(E, ragged)
        end
    end

    @testset "other element types keep Gridap's semantics" begin
        with_assembly(threaded=true, split=true, chunks=4, min_cells=1) do
            ones8 = Gridap.Arrays.lazy_map(identity, fill(Int8(1), 200))
            @test sum(ones8) === Int8(-56)                          # Gridap accumulates in the element type
            trues = Gridap.Arrays.lazy_map(identity, fill(true, 200))
            @test sum(trues) == sum(Gridap.Arrays.lazy_sum(Gridap.Arrays.array_cache(trues), trues))
            xs = Gridap.Arrays.lazy_map(x -> 2x, collect(range(0.0, 1.0; length=1001)))
            @test sum(xs) ≈ 1001.0
        end
    end

    @testset "dpt/dpf field equals the per-test-function form" begin
        prob = build_test_problem()
        sim0 = default_sim(prob.sim)
        control = prob.control
        rng = MersenneTwister(3)
        n = num_free_dofs(sim0.Pf)
        v = Gridap.FESpaces.get_fe_basis(sim0.Pf)
        pts = Gridap.CellData.get_cell_points(sim0.dΩ_design)
        # Random, exactly flat, exactly zero, nearly flat (around the 1e-8 gradient threshold) and
        # steep (|d| > 1 almost everywhere) filtered fields: every branch of DSP_dpf and DSP_d∇pf.
        fields = ["random" => rand(rng, n), "flat" => fill(0.5, n), "zero" => zeros(n),
            "nearly flat" => 0.5 .+ 1e-9 .* rand(rng, n), "steep" => 40.0 .* (rand(rng, n) .- 0.5)]
        for (name, vals) in fields
            pf = FEFunction(sim0.Pf, vals)
            ∇pf = ∇(pf)
            old = ((p, ∇p, q, ∇q) -> TA.dpt_dpf(p, ∇p, q, ∇q; control)) ∘ (v, ∇(v), pf, ∇pf)
            new = TA.dpt_dpf_field(v, pf, control, sim0.Ω_design)
            a, b = old(pts), new(pts)
            worst, scale = 0.0, 0.0
            for i in 1:length(a)
                worst = max(worst, maximum(abs, a[i] .- b[i]))
                scale = max(scale, maximum(abs, a[i]))
            end
            @test worst <= 1e-12 * max(scale, 1.0)
            name in ("random", "flat") && @test scale > 0
        end
        # A plain function of position (legacy foundry mode) keeps the per-test-function composition.
        pf_fun = x -> 0.5 + 0.3 * sin(x[1] / 30) * cos(x[2] / 40)
        old = ((p, ∇p, q, ∇q) -> TA.dpt_dpf(p, ∇p, q, ∇q; control)) ∘ (v, ∇(v), pf_fun, ∇(pf_fun))
        new = TA.dpt_dpf_field(v, pf_fun, control, sim0.Ω_design)
        a, b = old(pts), new(pts)
        @test maximum(maximum(abs, a[i] .- b[i]) for i in 1:length(a)) == 0
        @test maximum(maximum(abs, a[i]) for i in 1:length(a)) > 0
    end

    @testset "switches off give Gridap's own path" begin
        prob = build_test_problem()
        designs = designs_for(() -> prob; n=1)
        with_assembly(threaded=false, split=false) do
            objective_and_gradient!(zeros(length(designs[1])), designs[1], prob)
            @test isempty(TA._PLANS) && isempty(TA._BUFFERS)
        end
    end
end
