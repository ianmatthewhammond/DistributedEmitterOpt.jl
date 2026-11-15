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
    ]

    @testset "objective and gradient: $name" for (name, build) in cases
        designs = designs_for(build)
        ref = with_assembly(() -> evaluate_designs(build, designs); threaded=false, split=false)
        # Chunked loops only, split assembly only, and both: three designs each, so a plan is built,
        # checked against the ordinary assembly, and then used unchecked.
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
            @test all(p -> p.checked && p.ncalls == 3, TA._PLANS)
            @test TA.SPLIT_ASSEMBLY[]
        end
    end

    @testset "dpt/dpf field equals the per-test-function form" begin
        prob = build_test_problem()
        sim0 = default_sim(prob.sim)
        control = prob.control
        rng = MersenneTwister(3)
        pf = FEFunction(sim0.Pf, rand(rng, num_free_dofs(sim0.Pf)))
        v = Gridap.FESpaces.get_fe_basis(sim0.Pf)
        ∇pf = ∇(pf)
        old = ((p, ∇p, q, ∇q) -> TA.dpt_dpf(p, ∇p, q, ∇q; control)) ∘ (v, ∇(v), pf, ∇pf)
        new = TA.dpt_dpf_field(v, pf, control, sim0.Ω_design)
        pts = Gridap.CellData.get_cell_points(sim0.dΩ_design)
        a, b = old(pts), new(pts)
        worst, scale = 0.0, 0.0
        for i in 1:length(a)
            worst = max(worst, maximum(abs, a[i] .- b[i]))
            scale = max(scale, maximum(abs, a[i]))
        end
        @test scale > 0
        @test worst <= 1e-12 * scale
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
