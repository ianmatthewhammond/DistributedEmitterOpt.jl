# Checkpoint save/load round trip through the production save_checkpoint,
# resume_from_checkpoint! and metadata functions, on a stand-in problem. No FEM, filter or
# optimizer library is needed.
# Usage: julia test/checkpoint_state_tests.jl [path/to/Optimizer.jl]
using Test

module CheckpointFixture
import JLD2
mutable struct OptimizationProblem
    p::Vector{Float64}
    g::Float64
    iteration::Int
    g_history::Vector{Float64}
    g_best::Float64
    p_best::Vector{Float64}
    root::String
end

const NAMES = (:save_checkpoint, :resume_from_checkpoint!, :_checkpoint_metadata,
    :_checkpoint_optional_float, :_checkpoint_epigraph_config, :_checkpoint_alpha_schedule,
    :_robust_samples_matrix, :_validate_checkpoint_metadata!, :_same_robust_config,
    :_same_optional_schedule, :_robust_sharpness_record)

# Evaluate the production definitions above (and the checkpoint version constant) unchanged.
function load_optimizer(path)
    loaded = Set{Symbol}()
    function visit(ex)
        ex isa Expr || return
        if ex.head == :const && ex.args[1] isa Expr && ex.args[1].args[1] === :OPTIMIZER_CHECKPOINT_VERSION
            Core.eval(@__MODULE__, ex); push!(loaded, :OPTIMIZER_CHECKPOINT_VERSION); return
        end
        if ex.head in (:function, :(=)) && ex.args[1] isa Expr
            sig = ex.args[1]
            while sig isa Expr && sig.head in (:where, :(::))   # f(x)::T where T
                sig = sig.args[1]
            end
            if sig isa Expr && sig.head == :call && sig.args[1] in NAMES
                Core.eval(@__MODULE__, ex); push!(loaded, sig.args[1])
            end
            ex.head == :function && return
        end
        foreach(visit, ex.args)
    end
    visit(Meta.parseall(read(path, String)))
    want = Set((NAMES..., :OPTIMIZER_CHECKPOINT_VERSION))
    loaded == want || error("missing production definitions: $(setdiff(want, loaded))")
end
end

const C = CheckpointFixture
C.load_optimizer(isempty(ARGS) ? joinpath(@__DIR__, "../src/Optimization/Optimizer.jl") : ARGS[1])

meta(; kw...) = (; current_epoch=2, epoch_evals=5, max_iter=5, beta_schedule=[8.0, Inf],
    alpha_schedule=nothing, backend=:standalone_ccsaq, use_constraints=true, kw...)

@testset "checkpoint keeps the accepted objective, normalization and constraint scale" begin
    mktempdir() do dir
        path = joinpath(dir, "ck.jld2")
        # The accepted design's objective is 0.01; the last history entry is a rejected
        # trial's 0.81.
        src = C.OptimizationProblem([0.1, 0.1], 0.01, 2, [0.01, 0.81], 0.81, [0.9, 0.9], dir)
        C.save_checkpoint(src, path; meta(g_norm=7.0, constraint_scale=1e8)...)

        dst = C.OptimizationProblem(zeros(2), 0.0, 0, Float64[], -Inf, zeros(2), dir)
        md = C.resume_from_checkpoint!(dst, path; return_metadata=true)
        @test dst.p == [0.1, 0.1] && dst.g == 0.01
        @test dst.g_history == [0.01, 0.81] && dst.g_best == 0.81
        @test md.exact_resume && md.g_norm == 7.0 && md.constraint_scale == 1e8

        v(cs) = C._validate_checkpoint_metadata!(md; filepath=path, max_iter=5,
            beta_schedule=[8.0, Inf], alpha_schedule=nothing, backend=:standalone_ccsaq,
            use_constraints=true, constraint_scale=cs)
        @test v(1e8) === nothing
        @test_throws ArgumentError v(1.0)
        @test v(nothing) === nothing        # caller records no scale (NLopt / unconstrained)

        # Written without the new keys (as before this change): objective from history,
        # normalization recomputed, scale unchecked.
        C.save_checkpoint(src, path; meta()...)
        data = JLD2.load(path); delete!(data, "g"); JLD2.save(path, data)
        dst2 = C.OptimizationProblem(zeros(2), 0.0, 0, Float64[], -Inf, zeros(2), dir)
        md2 = C.resume_from_checkpoint!(dst2, path; return_metadata=true)
        @test dst2.g == 0.81
        @test md2.g_norm === nothing && md2.constraint_scale === nothing
        @test C._validate_checkpoint_metadata!(md2; filepath=path, max_iter=5,
            beta_schedule=[8.0, Inf], alpha_schedule=nothing, backend=:standalone_ccsaq,
            use_constraints=true, constraint_scale=1.0) === nothing
    end
end
