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
    filter_revision::Int
end
# A new problem holds no restored values: the current filter revision, as the package's constructors set it.
OptimizationProblem(p, g, iteration, g_history, g_best, p_best, root) =
    OptimizationProblem(p, g, iteration, g_history, g_best, p_best, root, 2)

const NAMES = (:save_checkpoint, :resume_from_checkpoint!, :_checkpoint_metadata,
    :_checkpoint_optional_float, :_checkpoint_epigraph_config, :_checkpoint_alpha_schedule,
    :_robust_samples_matrix, :_validate_checkpoint_metadata!, :_same_robust_config,
    :_same_optional_schedule, :_robust_sharpness_record, :_checkpoint_filter_revision,
    :_require_filter_revision, :load_checkpoint_design!, :_inherit_filter_revision!)
const CONSTS = (:OPTIMIZER_CHECKPOINT_VERSION,)

# Whether the DCT-I filter revisions differ for the problem: production asks the problem's filter; a test sets it.
const filter_matters = Ref(false)
const asked_use_constraints = Ref{Any}(nothing)
_filter_revision_matters(prob; use_constraints::Bool=true) = (asked_use_constraints[] = use_constraints; filter_matters[])

# Evaluate the production definitions above (and the checkpoint version constant) unchanged.
function load_optimizer(path)
    loaded = Set{Symbol}()
    function visit(ex)
        ex isa Expr || return
        if ex.head == :const && ex.args[1] isa Expr && ex.args[1].args[1] in CONSTS
            Core.eval(@__MODULE__, ex); push!(loaded, ex.args[1].args[1]); return
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
    want = Set((NAMES..., CONSTS...))
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
        data = C.JLD2.load(path); delete!(data, "g"); C.JLD2.save(path, data)
        dst2 = C.OptimizationProblem(zeros(2), 0.0, 0, Float64[], -Inf, zeros(2), dir)
        md2 = C.resume_from_checkpoint!(dst2, path; return_metadata=true)
        @test dst2.g == 0.81
        @test md2.g_norm === nothing && md2.constraint_scale === nothing
        @test C._validate_checkpoint_metadata!(md2; filepath=path, max_iter=5,
            beta_schedule=[8.0, Inf], alpha_schedule=nothing, backend=:standalone_ccsaq,
            use_constraints=true, constraint_scale=1.0) === nothing
    end
end

@testset "checkpoint records the DCT-I filter revision; an old one is refused where the filter changed" begin
    mktempdir() do dir
        path = joinpath(dir, "ck.jld2")
        src = C.OptimizationProblem([0.1, 0.1], 0.01, 2, [0.01, 0.81], 0.81, [0.9, 0.9], dir)
        v(md, matters) = C._validate_checkpoint_metadata!(md; filepath=path, max_iter=5,
            beta_schedule=[8.0, Inf], alpha_schedule=nothing, backend=:standalone_ccsaq,
            use_constraints=true, filter_revision_matters=matters)
        load() = C.resume_from_checkpoint!(C.OptimizationProblem(zeros(2), 0.0, 0, Float64[], -Inf, zeros(2), dir), path;
            return_metadata=true)

        # Written now: the current revision, resumable whether or not the revisions differ for the problem.
        C.save_checkpoint(src, path; meta()...)
        @test C.JLD2.load(path)["dct_filter_revision"] == C._checkpoint_filter_revision() == 2
        md = load()
        @test md.exact_resume && md.filter_revision == 2
        @test v(md, true) === nothing && v(md, false) === nothing

        # Written before the revision was recorded (no key): revision 1. Resumable where the kernel fits the cell
        # (the two revisions agree), refused where the old filter tiled the cell.
        data = C.JLD2.load(path); delete!(data, "dct_filter_revision"); C.JLD2.save(path, data)
        md1 = load()
        @test md1.exact_resume && md1.filter_revision == 1
        @test v(md1, false) === nothing
        @test_throws ArgumentError v(md1, true)
        msg = try v(md1, true); "" catch err; sprint(showerror, err) end
        @test occursin("revision 1", msg) && occursin("fresh optimization", msg)

        # A legacy checkpoint without optimizer metadata is refused in the same case, although nothing else in it
        # is validated; one written now without metadata carries the current revision.
        C.save_checkpoint(src, path)
        mdn = load()
        @test !mdn.exact_resume && mdn.filter_revision == 2 && v(mdn, true) === nothing
        data = C.JLD2.load(path); delete!(data, "dct_filter_revision"); C.JLD2.save(path, data)
        mdl = load()
        @test !mdl.exact_resume && mdl.filter_revision == 1
        @test v(mdl, false) === nothing
        @test_throws ArgumentError v(mdl, true)
    end
end

@testset "a direct load refuses an old filter revision before restoring anything; the design alone can be imported" begin
    mktempdir() do dir
        path = joinpath(dir, "ck.jld2")
        src = C.OptimizationProblem([0.1, 0.2], 0.01, 2, [0.01, 0.81], 0.81, [0.9, 0.8], dir)
        C.save_checkpoint(src, path; meta()...)
        data = C.JLD2.load(path); delete!(data, "dct_filter_revision"); C.JLD2.save(path, data)   # as written by revision 1
        fresh() = C.OptimizationProblem([0.5, 0.5], 7.0, 3, [1.0, 2.0, 3.0], 3.0, [0.6, 0.6], dir)
        untouched(q) = q.p == [0.5, 0.5] && q.g == 7.0 && q.iteration == 3 && q.g_history == [1.0, 2.0, 3.0] &&
                       q.g_best == 3.0 && q.p_best == [0.6, 0.6]

        # Where the revisions differ, resume_from_checkpoint! throws and leaves every field of the problem as it was;
        # nothing of the old filter's can then be saved again under the current revision.
        C.filter_matters[] = true
        dst = fresh()
        @test_throws ArgumentError C.resume_from_checkpoint!(dst, path)
        @test untouched(dst)
        @test C.asked_use_constraints[] == true            # the default is the cautious one
        dst = fresh()
        @test_throws ArgumentError C.resume_from_checkpoint!(dst, path; return_metadata=true, use_constraints=false)
        @test untouched(dst) && C.asked_use_constraints[] == false   # the caller's answer reaches the test

        # The design alone can be taken; cached values are discarded, not carried over.
        dst = fresh()
        @test C.load_checkpoint_design!(dst, path) === dst
        @test dst.p == [0.1, 0.2]
        @test dst.g == 0.0 && dst.iteration == 0 && isempty(dst.g_history) && dst.g_best == -Inf && dst.p_best == [0.0, 0.0]
        C.save_checkpoint(dst, joinpath(dir, "new.jld2"); meta()...)
        again = C.JLD2.load(joinpath(dir, "new.jld2"))
        @test again["dct_filter_revision"] == 2 && isempty(again["g_history"]) && again["g_best"] == -Inf
        @test_throws ArgumentError C.load_checkpoint_design!(C.OptimizationProblem(zeros(3), 0.0, 0, Float64[], -Inf, zeros(3), dir), path)

        # Where the revisions agree, the same old checkpoint loads in full, as before.
        C.filter_matters[] = false
        dst = fresh()
        md = C.resume_from_checkpoint!(dst, path; return_metadata=true)
        @test md.filter_revision == 1 && dst.p == [0.1, 0.2] && dst.g == 0.01 && dst.g_history == [0.01, 0.81]

        # The refusal itself: unrecorded and old revisions, only where it matters.
        C.filter_matters[] = false
        @test C._require_filter_revision(nothing, "Optimizer state s", false) === nothing
        @test C._require_filter_revision(1, "Optimizer state s", false) === nothing
        @test C._require_filter_revision(2, "Optimizer state s", true) === nothing
        @test_throws ArgumentError C._require_filter_revision(1, "Optimizer state s", true)
        @test_throws ArgumentError C._require_filter_revision(nothing, "Optimizer state s", true)
        msg = try C._require_filter_revision(nothing, "Optimizer state s", true); "" catch err; sprint(showerror, err) end
        @test occursin("Optimizer state s", msg) && occursin("without a recorded revision", msg) && occursin("load_checkpoint_design!", msg)
    end
end

@testset "a checkpoint keeps the filter revision its cached values came with, however it is saved again" begin
    mktempdir() do dir
        path, copy1, copy2 = joinpath(dir, "old.jld2"), joinpath(dir, "copy1.jld2"), joinpath(dir, "copy2.jld2")
        src = C.OptimizationProblem([0.1, 0.2], 0.01, 2, [0.01, 0.81], 0.81, [0.9, 0.8], dir)
        @test src.filter_revision == 2                  # a problem that restored nothing: the running code
        C.save_checkpoint(src, path; meta()...)
        data = C.JLD2.load(path); delete!(data, "dct_filter_revision"); C.JLD2.save(path, data)   # as written by main
        blank() = C.OptimizationProblem(zeros(2), 0.0, 0, Float64[], -Inf, zeros(2), dir)
        revision(file) = C.JLD2.load(file)["dct_filter_revision"]

        # Loaded where the revisions agree for the problem (say filter and constraints off, or a kernel that fits):
        # accepted in full, and the values remember where they came from.
        C.filter_matters[] = false
        a = blank()
        C.resume_from_checkpoint!(a, path; use_constraints=false)
        @test a.g_history == [0.01, 0.81] && a.filter_revision == 1
        # Saved again, with or without optimizer metadata, after more evaluations too: still revision 1.
        C.save_checkpoint(a, copy1; meta()...)
        @test revision(copy1) == 1
        push!(a.g_history, 0.5); a.g = 0.5; a.iteration += 1
        C.save_checkpoint(a, copy2)
        @test revision(copy2) == 1
        # Such a copy still resumes wherever the revisions agree (a run in flight is not stopped by its label) ...
        b = blank()
        md = C.resume_from_checkpoint!(b, copy1; return_metadata=true, use_constraints=false)
        @test md.filter_revision == 1 && b.g_history == [0.01, 0.81] && b.filter_revision == 1
        # ... and is refused, untouched, once the problem is one where they differ (the filter switched on, a longer
        # radius): the label was not laundered by the save.
        C.filter_matters[] = true
        for file in (copy1, copy2)
            c = blank()
            @test_throws ArgumentError C.resume_from_checkpoint!(c, file)
            @test isempty(c.g_history) && c.g == 0.0 && c.filter_revision == 2
        end

        # Promotion happens only where the inherited caches are discarded: the design-only import.
        d = blank()
        C.load_checkpoint_design!(d, copy1)
        @test d.filter_revision == 2
        C.save_checkpoint(d, joinpath(dir, "fresh.jld2"); meta()...)
        @test revision(joinpath(dir, "fresh.jld2")) == 2
        e = blank()
        C.resume_from_checkpoint!(e, joinpath(dir, "fresh.jld2"))            # accepted where the revisions differ
        @test e.p == [0.1, 0.2] && isempty(e.g_history) && e.filter_revision == 2
        # The label travels with a copy of the problem: a deep copy of a problem holding restored values still saves
        # them as revision 1, and that file is refused where the revisions differ.
        C.filter_matters[] = false
        h = blank()
        C.resume_from_checkpoint!(h, path)
        hc = deepcopy(h)
        @test hc !== h && hc.filter_revision == 1 && hc.g_history == [0.01, 0.81]
        C.save_checkpoint(hc, joinpath(dir, "copy3.jld2"); meta()...)
        @test revision(joinpath(dir, "copy3.jld2")) == 1
        C.filter_matters[] = true
        k = blank()
        @test_throws ArgumentError C.resume_from_checkpoint!(k, joinpath(dir, "copy3.jld2"))
        @test isempty(k.g_history) && k.filter_revision == 2
        # An import never raises the provenance: a problem that holds revision-1 values and then restores a
        # revision-2 checkpoint in full stays at 1 (the cautious side; a design-only import or a fresh run promotes).
        C.filter_matters[] = false
        m = blank()
        C.resume_from_checkpoint!(m, path)
        C.resume_from_checkpoint!(m, joinpath(dir, "fresh.jld2"))
        @test m.filter_revision == 1
        @test C._inherit_filter_revision!(blank(), nothing).filter_revision == 1    # unrecorded counts as 1
        @test C._inherit_filter_revision!(blank(), 2).filter_revision == 2
        # A problem that had restored old values and then imports a design alone is promoted as well.
        C.filter_matters[] = false
        f = blank()
        C.resume_from_checkpoint!(f, path)
        @test f.filter_revision == 1
        C.load_checkpoint_design!(f, path)
        @test f.filter_revision == 2 && isempty(f.g_history)
    end
end
