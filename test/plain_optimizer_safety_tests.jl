# Execute the production optimizer orchestration with an injected backend and
# arithmetic objective. No FEM, filtering, library build, or package load is needed.
# Usage: julia test/plain_optimizer_safety_tests.jl [path/to/Optimizer.jl]
using Test, LinearAlgebra

module PlainOptimizerFixture
using LinearAlgebra
mutable struct FakeControl
    β::Float64
    b1::Float64
    use_dct::Bool
    use_filter::Bool
end
FakeControl(β, b1) = FakeControl(β, b1, true, true)
# The 2D filter's refusal (production: Filtering2D.jl); the optimizer's own guard is what these tests exercise.
require_dct_filter_2d(control, caller::AbstractString) =
    control.use_dct ? nothing : throw(ArgumentError("$caller: use_dct=true required"))
mutable struct OptimizationProblem
    p::Vector{Float64}
    g::Float64
    ∇g::Vector{Float64}
    iteration::Int
    g_history::Vector{Float64}
    g_best::Float64
    p_best::Vector{Float64}
    control::FakeControl
    pde::NamedTuple
    root::String
    foundry_mode::Bool
    sim::Nothing
    filter_revision::Int
end
problem() = OptimizationProblem([0.4,0.6], 0.0, zeros(2), 0, Float64[], -Inf,
    zeros(2), FakeControl(8.0,1e-6), (;env=(;logger_cfg=nothing), inputs=nothing, outputs=nothing, α_loss=0.0),
    "unused", false, nothing, 2)
# The loss schedule rebuilds prob.pde; only α_loss matters here.
MaxwellProblem(; env, inputs, outputs, α_loss) = (; env, inputs, outputs, α_loss)
const checkpoints = NamedTuple[]
const freed = Ref(0)
const return_code = Ref(-1)
struct MmaState end
free_state!(::MmaState) = (freed[] += 1; nothing)
# The legacy-named state file of an epoch (two arguments); nothing is written under the per-evaluation name here.
const legacy_state = Ref{Any}(nothing)
_state_path(backup_path, epoch) = legacy_state[]
_state_path(backup_path, epoch, epoch_evals) = nothing
_iteration_checkpoint_metadata(args...; kwargs...) = nothing
_epigraph_config(args...) = nothing
_robust_sharpness_record(args...) = nothing
with_run_logger(f; kwargs...) = f()
with_run_logger(f, config; kwargs...) = f()
const norm_calls = Ref(0)   # counts flat evaluations, the first thing optimize! does after its checks
flat_substrate_norm(prob) = (norm_calls[] += 1; prob === nothing || (prob.g = 999.0); 2.0)   # production's flat evaluation overwrites prob.g
log_info(args...; kwargs...) = nothing
log_debug(args...; kwargs...) = nothing
const warnings = Any[]
log_warn(args...; kwargs...) = (push!(warnings, (args, (;kwargs...))); nothing)
next_iteration!(prob) = (prob.iteration += 1)
# What the problem held as its objective when the run's first evaluation began.
const g_at_first_eval = Ref{Any}(nothing)
function objective_and_gradient!(g, p, prob)
    g_at_first_eval[] === nothing && (g_at_first_eval[] = prob.g)
    g .= 2p
    return sum(abs2,p)
end
function log_iteration!(prob, g, p; kwargs...)
    push!(prob.g_history, g)
    if g > prob.g_best   # best-record update, as in production
        prob.g_best = g
        prob.p_best .= p
    end
end
save_checkpoint(prob, path; kwargs...) = push!(checkpoints, (;kwargs..., problem_filter_revision=prob.filter_revision))
# Resume: the loaded design/objective and metadata come from `resume_state`; the
# production `_checkpoint_resume_progress` turns the metadata into a start epoch.
const resume_state = Ref{Any}(nothing)
const validated = Ref{Any}(nothing)
const resume_kwargs = Ref{Any}(nothing)
function resume_from_checkpoint!(prob, path; return_metadata=false, kwargs...)
    resume_kwargs[] = (; kwargs...)
    st = resume_state[]
    prob.p .= st.p; prob.g = st.g; prob.iteration = st.iteration
    return st.metadata
end
_validate_checkpoint_metadata!(md; kwargs...) = (validated[] = (;kwargs...); nothing)
default_sim(sim) = sim
# Whether the DCT-I filter revisions differ for the problem: production asks the filter; here a test sets it.
const filter_matters = Ref(false)
const matters_kwargs = Ref{Any}(nothing)
_filter_revision_matters(prob; kwargs...) = (matters_kwargs[] = (; kwargs...); filter_matters[])
# Reaching the state loader is all a test here needs to see.
struct StateLoaded <: Exception end
const state_loads = Ref(:throw)     # :ok lets a run go on with the loaded state
load_state(path) = state_loads[] === :ok ? MmaState() : throw(StateLoaded())
# The epigraph epoch builds its cache before it looks at a state file; nothing here reaches further.
EpigraphCache(np, ns) = nothing
# Constraint values: fixed by default; `final_values` overrides them for gradient-free
# calls (the final feasibility check) so a test can set the returned design's values.
const final_values = Ref{Any}(nothing)
const gradfree_calls = Ref(0)
function glc_solid(p,g; kwargs...)
    isempty(g) || (g .= 2.0; return 0.2)
    gradfree_calls[] += 1
    final_values[] isa Function && return final_values[](kwargs[:control].β)[1]
    return final_values[] === nothing ? 0.2 : final_values[][1]
end
function glc_void(p,g; kwargs...)
    isempty(g) || (g .= -3.0; return -0.3)
    final_values[] isa Function && return final_values[](kwargs[:control].β)[2]
    return final_values[] === nothing ? -0.3 : final_values[][2]
end

module MmaccsaBackend
using ..PlainOptimizerFixture: return_code
const captured_constraints = Ref{Any}(nothing)
# (accepted, trials): evaluate the accepted point, then the trials, so the last
# callback is a rejected trial; leave `p` at the accepted point, as the library does.
const script = Ref{Any}(nothing)
function minimize!(algorithm, objective, p; maxeval, kwargs...)
    captured_constraints[] = kwargs[:constraints]
    if script[] !== nothing
        accepted, trials = script[]
        fmin = objective(copy(accepted), Float64[])
        foreach(t -> objective(copy(t), Float64[]), trials)
        p .= accepted
        return (return_code[], fmin, p)
    end
    n = return_code[] < 0 ? 3 : maxeval
    for _ in 1:n
        objective(p, similar(p))
    end
    return (return_code[], 0.0, n)
end


end

# Evaluate the actual three production functions, with their bodies unchanged.
# Only their external problem/backend dependencies above are injected.
function load_optimizer(path)
    parsed = Meta.parseall(read(path,String))
    loaded = Set{Symbol}()
    function visit(ex)
        ex isa Expr || return
        if ex.head == :function
            sig = ex.args[1]
            sig = sig isa Expr && sig.head == :where ? sig.args[1] : sig
            if sig isa Expr && sig.head == :call
                name = sig.args[1]
                if (name in (:optimize!, :run_epoch!, :_run_epoch_standalone!, :linewidth_feasibility,
                             :_set_epoch_state!, :_run_epoch_standalone_epigraph!) &&
                    occursin("prob::OptimizationProblem", string(sig))) ||
                   name in (:_checkpoint_resume_progress, :_require_filter_revision, :_checkpoint_filter_revision,
                            :_inherit_filter_revision!, :require_dct_filter_problem)
                    Core.eval(@__MODULE__, ex)
                    push!(loaded,name)
                end
            end
            return
        end
        foreach(visit,ex.args)
    end
    visit(parsed)
    # `require_dct_filter_problem` is loaded when the source defines it; a source without it (one from before the
    # 2D filter refusal) still loads, so its optimizer runs every testset here and the refusal tests fail there.
    required = Set([:optimize!,:run_epoch!,:_run_epoch_standalone!,:linewidth_feasibility,
                     :_set_epoch_state!,:_checkpoint_resume_progress,:_require_filter_revision,
                     :_checkpoint_filter_revision,:_run_epoch_standalone_epigraph!,:_inherit_filter_revision!])
    issubset(required, loaded) || error("missing production functions: $(setdiff(required, loaded))")
end
end

const F = PlainOptimizerFixture
F.load_optimizer(isempty(ARGS) ? joinpath(@__DIR__,"../src/Optimization/Optimizer.jl") : ARGS[1])

@testset "plain backend failures never complete an epoch" begin
    for code in (-1,-2,-3,-4,-5)
        F.return_code[] = code
        empty!(F.checkpoints); F.freed[] = 0
        p = F.problem()
        err = try
            F.optimize!(p; backend=:standalone_mma,max_iter=5,β_schedule=[8.0],
                backup=true,backup_every=0,backup_path="injected")
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test err !== nothing && occursin("ret=$code",sprint(showerror,err))
        @test p.iteration == 3
        @test isempty(F.checkpoints)
        @test F.freed[] == 1
    end
end
@testset "a successful backend can write completion metadata" begin
    F.return_code[] = 5
    empty!(F.checkpoints); F.freed[] = 0
    p = F.problem()
    F.optimize!(p; backend=:standalone_mma,max_iter=5,β_schedule=[8.0],
        backup=true,backup_every=0,backup_path="injected")
    @test p.iteration == 5
    @test length(F.checkpoints) == 2
    @test all(c.current_epoch == 1 && c.epoch_evals == 5 for c in F.checkpoints)
    @test F.freed[] == 1
end

@testset "the final accepted design and its objective are returned" begin
    F.return_code[] = 5
    accepted = [0.1, 0.1]
    g_accepted = sum(abs2, accepted) / F.flat_substrate_norm(nothing)
    for trials in ([[0.9, 0.9]], [[0.9, 0.9], [0.05, 0.05]])
        F.MmaccsaBackend.script[] = (accepted, trials)
        p = F.problem()
        g, design = F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0])
        # A trial with a higher objective (here infeasible for sum(p) <= 0.3) is
        # recorded as the best seen but never returned in place of the accepted design.
        @test design == accepted && p.p == accepted
        @test g == g_accepted && p.g == g_accepted
        @test p.p_best == [0.9, 0.9] && p.g_best > g
    end
    F.MmaccsaBackend.script[] = nothing
end

begin
    @testset "raw tolerance reaches both standalone constraints" begin
        F.return_code[] = 5
        for backend in (:standalone_mma,:standalone_ccsaq), raw in (1e-8,6e-14), cs in (1.0,1e8,1e-3)
            p=F.problem(); p.foundry_mode=true
            F.optimize!(p;backend,max_iter=5,β_schedule=[Inf],use_constraints=true,
                constraint_tol_raw=raw,constraint_scale=cs)
            constraints=F.MmaccsaBackend.captured_constraints[]
            @test length(constraints)==2
            @test all(c[2]==raw*cs for c in constraints)
            grad=zeros(2)
            @test constraints[1][1](p.p,grad)==0.2cs
            @test grad==fill(2cs,2)
            @test constraints[2][1](p.p,grad)==-0.3cs
            @test grad==fill(-3cs,2)
        end
        p=F.problem();p.foundry_mode=true
        F.optimize!(p;backend=:standalone_mma,max_iter=5,β_schedule=[Inf],use_constraints=true)
        @test all(c[2]==1e-8 for c in F.MmaccsaBackend.captured_constraints[])
        for bad in (-1.0,NaN,Inf)
            @test_throws ArgumentError F.optimize!(F.problem();constraint_tol_raw=bad)
        end
        @test_throws ArgumentError F.optimize!(F.problem();backend=:standalone_mma,
            robust_mode=:epigraph, constraint_tol_raw=6e-14)
    end
end

@testset "zero, negative and nonfinite constraint scales are refused before any backend runs" begin
    # The standalone foundry constraints multiply value, gradient and tolerance by the
    # scale: zero would erase both linewidth constraints and a negative factor would
    # reverse them. The scale is checked for every backend, with or without constraints.
    # Before the fix, for the standalone backends, the `err isa ArgumentError`,
    # `p.iteration == 0`, `captured_constraints === nothing` and `isempty(checkpoints)`
    # assertions fail: optimize! and run_epoch! accept the scale and run the backend. For
    # :nlopt (not loaded in this fixture) only the ArgumentError assertion fails there.
    F.return_code[] = 5
    refused(f) = try f(); nothing catch err; err end
    for bad in (0.0, -0.0, -1.0, NaN, Inf, -Inf)
        for (backend, cons) in ((:standalone_mma, true), (:standalone_ccsaq, true),
                                (:standalone_mma, false), (:nlopt, true))
            F.MmaccsaBackend.captured_constraints[] = nothing
            empty!(F.checkpoints)
            p = F.problem(); p.foundry_mode = true
            err = refused(() -> F.optimize!(p; backend, max_iter=5, β_schedule=[8.0, Inf],
                use_constraints=cons, constraint_scale=bad, backup=true, backup_every=0,
                backup_path="injected"))
            @test err isa ArgumentError && occursin("constraint_scale", sprint(showerror, err))
            @test p.iteration == 0 && p.p == [0.4, 0.6]
            @test F.MmaccsaBackend.captured_constraints[] === nothing
            @test isempty(F.checkpoints)
        end
        for backend in (:standalone_mma, :standalone_ccsaq)
            F.MmaccsaBackend.captured_constraints[] = nothing
            p = F.problem(); p.foundry_mode = true
            err = refused(() -> F.run_epoch!(p, 5, true, 1e-6; backend, epoch=1,
                constraint_scale=bad))
            @test err isa ArgumentError && occursin("constraint_scale", sprint(showerror, err))
            @test p.iteration == 0
            @test F.MmaccsaBackend.captured_constraints[] === nothing
        end
    end
end

# regression: a refused scale must leave a problem that has already run untouched. Without the guard in
# optimize!, run_epoch! still throws, but only after optimize! has normalized, reset the
# iteration and cleared the history and best record.
@testset "a refused constraint_scale leaves a populated problem and its checkpoint untouched" begin
    F.return_code[] = 5
    for bad in (0.0, -0.0, -1.0, NaN, Inf, -Inf), resume in (false, true)
        p = F.problem(); p.foundry_mode = true
        F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], use_constraints=true)
        @test p.iteration == 5 && !isempty(p.g_history) && p.g_best > -Inf
        # State a refusal must not touch, with a sentinel for every later step.
        before = (iteration=p.iteration, g_history=copy(p.g_history), g_best=p.g_best,
                  p_best=copy(p.p_best), p=copy(p.p), g=p.g, ∇g=copy(p.∇g),
                  filter_revision=p.filter_revision, β=p.control.β)
        ckpt = tempname()
        write(ckpt, "sentinel checkpoint bytes")
        bytes = read(ckpt)
        empty!(F.checkpoints)
        F.norm_calls[] = 0
        F.resume_kwargs[] = :untouched
        F.resume_state[] = (p=[0.9, 0.9], g=7.0, iteration=3, metadata=nothing)
        F.MmaccsaBackend.captured_constraints[] = nothing
        err = try
            F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0, Inf],
                use_constraints=true, constraint_scale=bad, backup=true, backup_every=0,
                backup_path=ckpt, resume_from=(resume ? ckpt : nothing))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("constraint_scale", sprint(showerror, err))
        @test F.norm_calls[] == 0                       # no normalization ran
        @test F.resume_kwargs[] === :untouched          # no checkpoint was loaded
        @test p.iteration == before.iteration
        @test p.g_history == before.g_history
        @test p.g_best == before.g_best && p.p_best == before.p_best
        @test p.p == before.p && p.g == before.g && p.∇g == before.∇g
        @test p.filter_revision == before.filter_revision && p.control.β == before.β
        @test F.MmaccsaBackend.captured_constraints[] === nothing
        @test isempty(F.checkpoints)
        @test read(ckpt) == bytes
        rm(ckpt)
    end
end

# The same for the epoch entry point: a populated problem is not changed by a refused epoch.
@testset "run_epoch! refuses a bad constraint_scale without touching a populated problem" begin
    F.return_code[] = 5
    for bad in (0.0, -0.0, -1.0, NaN, Inf, -Inf), backend in (:standalone_mma, :standalone_ccsaq)
        p = F.problem(); p.foundry_mode = true
        F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], use_constraints=true)
        before = (iteration=p.iteration, g_history=copy(p.g_history), g_best=p.g_best,
                  p_best=copy(p.p_best), p=copy(p.p), g=p.g)
        empty!(F.checkpoints)
        F.MmaccsaBackend.captured_constraints[] = nothing
        err = try
            F.run_epoch!(p, 5, true, 1e-6; backend, epoch=1, constraint_scale=bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("constraint_scale", sprint(showerror, err))
        @test (p.iteration, p.g_history, p.g_best, p.p_best, p.p, p.g) ==
              (before.iteration, before.g_history, before.g_best, before.p_best, before.p, before.g)
        @test F.MmaccsaBackend.captured_constraints[] === nothing
        @test isempty(F.checkpoints)
    end
end

# regression: the public text says exactly what each path does.
@testset "the optimize! docstring states which quantities each path scales" begin
    src = read(isempty(ARGS) ? joinpath(@__DIR__, "../src/Optimization/Optimizer.jl") : ARGS[1], String)
    a = first(findfirst("- `constraint_scale` --", src))
    b = last(findfirst("reciprocal scale.", src))
    doc = replace(src[a:b], r"\s+" => " ")
    @test occursin("scales both linewidth values, both gradients and `constraint_tol_raw`", doc)
    @test occursin("in exact arithmetic the feasible set is unchanged", doc)
    @test occursin("plain/KS with `backend=:nlopt`, non-foundry standalone, or linewidths off: the constraint arithmetic ignores it", doc)
    @test occursin("with linewidths off none are installed", doc)
    @test occursin("epigraph with linewidths off: not used; any positive value is accepted", doc)
    @test occursin("epigraph with foundry linewidths: must satisfy `constraint_scale * b1 == 1`", doc)
    @test occursin("scales the linewidth value and the design gradient", doc)
    @test occursin("`epigraph_linewidth_tol` is passed to the backend unscaled", doc)
    @test occursin("a non-foundry epigraph epoch with linewidths is refused by an existing check, even with the reciprocal scale", doc)
    @test !occursin("requires it to equal `1/b1`", doc)
end

@testset "the returned design's linewidth feasibility is checked and reported" begin
    F.return_code[] = 5
    run(; kwargs...) = begin
        p = F.problem(); p.foundry_mode = true
        empty!(F.warnings); F.gradfree_calls[] = 0
        out = F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
            use_constraints=true, constraint_tol_raw=1e-8, kwargs...)
        (p, out)
    end
    # Feasible, including a value exactly at the tolerance: no warning, design returned.
    for vals in ((-1e-3, -2e-3), (1e-8, -1.0))
        F.final_values[] = vals
        p, (g, design) = run()
        @test F.gradfree_calls[] == 1
        @test isempty(F.warnings)
        @test design == p.p
    end
    # Infeasible: :warn (default) returns the design and logs both values; :error throws
    # and leaves the design in prob.p.
    for vals in ((2e-8, -1.0), (-1.0, 0.05), (NaN, -1.0))
        F.final_values[] = vals
        p, (g, design) = run()
        @test design == p.p
        @test length(F.warnings) == 1
        w = F.warnings[1]
        @test w[1][1] === :constraints && occursin("violates", w[1][2])
        @test isequal(w[2].solid, vals[1]) && isequal(w[2].void, vals[2]) && w[2].tol == 1e-8
        p2 = F.problem(); p2.foundry_mode = true
        err = try
            F.optimize!(p2; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
                use_constraints=true, on_infeasible=:error)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("violates the linewidth constraints at β=Inf", err.msg)
        @test p2.p == [0.4, 0.6]   # the fixture backend leaves the start point accepted
        @test p2.iteration == 10
    end
    # No constraints: nothing is evaluated or reported.
    F.final_values[] = (1.0, 1.0)
    p = F.problem(); p.foundry_mode = true; F.gradfree_calls[] = 0; empty!(F.warnings)
    F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0], on_infeasible=:error)
    @test F.gradfree_calls[] == 0 && isempty(F.warnings)
    @test_throws ArgumentError F.optimize!(F.problem(); on_infeasible=:ignore)
    # The helper itself, at the boundary and with an explicit tolerance.
    F.final_values[] = (0.1, 0.1)
    pf = F.problem(); pf.foundry_mode = true
    @test F.linewidth_feasibility(pf, [0.5, 0.5]; tol=0.1).feasible
    @test !F.linewidth_feasibility(pf, [0.5, 0.5]; tol=0.099).feasible
    F.final_values[] = nothing
end

# Metadata as `_checkpoint_metadata` returns it, for the fields optimize! reads.
resume_md(; current_epoch, epoch_evals, g_norm=nothing) = (; exact_resume=true, current_epoch,
    epoch_evals, mma_state_path=nothing, epigraph_T=nothing, epigraph_tau=nothing, g_norm,
    constraint_scale=nothing, filter_revision=2)

@testset "a completed resume restores the last epoch's state before returning" begin
    F.return_code[] = 5
    F.MmaccsaBackend.script[] = nothing
    # Feasible only at the last epoch's beta: a check at the constructed beta=8 would warn.
    F.final_values[] = β -> β == Inf ? (-1.0, -1.0) : (1.0, 1.0)
    F.resume_state[] = (; p=[0.1, 0.1], g=0.01, iteration=10,
        metadata=resume_md(current_epoch=2, epoch_evals=5))   # no g_norm: the flat evaluation runs
    p = F.problem(); p.foundry_mode = true; empty!(F.warnings)
    @test p.control.β == 8.0 && p.pde.α_loss == 0.0
    g, design = F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
        α_schedule=[0.25, 0.5], use_constraints=true, resume_from="injected", on_infeasible=:error)
    @test p.control.β == Inf
    @test p.pde.α_loss == 0.5
    @test isempty(F.warnings)
    @test design == [0.1, 0.1] && g == 0.01
    @test p.g == 0.01            # not the flat evaluation's 999
    @test p.iteration == 10      # no epoch ran
    # The resume told the checkpoint validation whether the DCT-I filter revisions differ for this problem.
    @test F.validated[].filter_revision_matters == false
    F.filter_matters[] = true
    p = F.problem(); p.foundry_mode = true
    F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
        α_schedule=[0.25, 0.5], use_constraints=true, resume_from="injected", on_infeasible=:error)
    @test F.validated[].filter_revision_matters == true
    # optimize! tells both the loader and the test whether the length-scale constraints will be on.
    @test F.resume_kwargs[].use_constraints == true && F.matters_kwargs[].use_constraints == true
    p = F.problem(); p.foundry_mode = true
    F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
        α_schedule=[0.25, 0.5], use_constraints=false, resume_from="injected")
    @test F.resume_kwargs[].use_constraints == false && F.matters_kwargs[].use_constraints == false
    F.filter_matters[] = false
    # A resume keeps whatever revision the loader recorded for the restored values: optimize! does not promote it ...
    p = F.problem(); p.foundry_mode = true; p.filter_revision = 1      # as the loader leaves it after an old checkpoint
    F.optimize!(p; backend=:standalone_ccsaq, max_iter=5, β_schedule=[8.0, Inf],
        α_schedule=[0.25, 0.5], use_constraints=true, resume_from="injected", on_infeasible=:error)
    @test p.filter_revision == 1
    F.final_values[] = nothing
end

@testset "a fresh optimization labels its cached values with the current filter revision" begin
    F.return_code[] = 5
    # The problem arrives with an objective, a history and a best record of revision 1. A fresh history discards all
    # of them BEFORE the first evaluation: the objective too, so nothing inherited can be published as current.
    old!(p) = (p.filter_revision = 1; p.g = 0.77; push!(p.g_history, 0.77); p.g_best = 0.77; p.p_best .= 0.3; p)
    p = old!(F.problem()); F.g_at_first_eval[] = nothing
    F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0])
    @test F.g_at_first_eval[] == 0.0
    @test p.filter_revision == 2 && !(0.77 in p.g_history)
    # ... and not when the caller keeps the problem's history (empty_history=false): inherited values stay, and so
    # does their label.
    p = old!(F.problem()); F.g_at_first_eval[] = nothing
    F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], empty_history=false)
    @test F.g_at_first_eval[] == 0.77
    @test p.filter_revision == 1 && 0.77 in p.g_history
end

@testset "an optimizer-state file is not loaded under another DCT-I filter revision" begin
    mktemp() do path, io
        close(io)
        p = F.problem(); p.foundry_mode = true
        epoch!(rev; uc=true) = F._run_epoch_standalone!(p, 5, uc, 1e-6; epoch=1,
            checkpoint_mma_state_path=path, checkpoint_filter_revision=rev)
        # Where the revisions differ: a state file named explicitly needs the current revision with it.
        F.filter_matters[] = true
        @test_throws ArgumentError epoch!(nothing)
        @test_throws ArgumentError epoch!(1)
        @test_throws F.StateLoaded epoch!(2)
        @test F.matters_kwargs[].use_constraints == true
        @test_throws ArgumentError epoch!(1; uc=false)
        @test F.matters_kwargs[].use_constraints == false     # the epoch's own answer is passed on
        # Where they agree (kernel fits, or the filter is not used): loaded whatever the record says.
        F.filter_matters[] = false
        @test_throws F.StateLoaded epoch!(nothing)
        @test_throws F.StateLoaded epoch!(1)

        # The same through the dispatcher: run_epoch! forwards the revision to the standalone epoch ...
        disp!(rev) = F.run_epoch!(p, 5, true, 1e-6; backend=:standalone_ccsaq, epoch=1,
            checkpoint_mma_state_path=path, checkpoint_filter_revision=rev)
        F.filter_matters[] = true
        @test_throws ArgumentError disp!(nothing)
        @test_throws ArgumentError disp!(1)
        @test_throws F.StateLoaded disp!(2)
        # ... and to the epigraph epoch, whose check sits before anything is read from the state file. Past the
        # filter check the next check of that loader speaks (this call gives it no epigraph_T), which shows where
        # the filter check stands.
        said(f) = try f(); "" catch err; sprint(showerror, err) end
        epi!(rev) = F.run_epoch!(p, 5, false, 1e-6; backend=:standalone_ccsaq, epoch=1, robust_mode=:epigraph,
            epigraph_config=(tol=1e-6, sigma=1.0, eps=1e-6, linewidth_tol=0.01), robust_samples=[(0.0, 0.0)],
            checkpoint_mma_state_path=path, checkpoint_filter_revision=rev)
        @test occursin("DCT-I design filter", said(() -> epi!(nothing)))
        @test occursin("DCT-I design filter", said(() -> epi!(1)))
        @test occursin("epigraph_T", said(() -> epi!(2))) && !occursin("DCT-I design filter", said(() -> epi!(2)))
        F.filter_matters[] = false
        @test occursin("epigraph_T", said(() -> epi!(nothing)))

        # Completed restores (the loader hands back a state and the epoch runs on), where the revisions agree for the
        # problem so that nothing is refused. Loaded optimizer state is inherited, so the provenance follows it and
        # every checkpoint the run publishes carries it.
        F.state_loads[] = :ok
        F.return_code[] = 5
        # (a) a fresh optimize! finds a state file under the legacy name: written by revision-1 code.
        F.legacy_state[] = path
        empty!(F.checkpoints)
        q = F.problem()
        F.optimize!(q; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], backup=true, backup_every=0, backup_path="injected")
        @test q.filter_revision == 1
        @test !isempty(F.checkpoints) && all(c.problem_filter_revision == 1 for c in F.checkpoints)
        # Without such a file the same run publishes the current revision.
        F.legacy_state[] = nothing
        empty!(F.checkpoints)
        q = F.problem()
        F.optimize!(q; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], backup=true, backup_every=0, backup_path="injected")
        @test q.filter_revision == 2
        @test !isempty(F.checkpoints) && all(c.problem_filter_revision == 2 for c in F.checkpoints)
        # (b) a state file named explicitly brings the revision of the checkpoint that names it; unrecorded counts
        # as 1; an older provenance already in the problem is never raised.
        for (had, rev, now) in ((2, 1, 1), (2, nothing, 1), (2, 2, 2), (1, 2, 1))
            q = F.problem(); q.filter_revision = had
            F._run_epoch_standalone!(q, 5, false, 1e-6; epoch=1, checkpoint_mma_state_path=path, checkpoint_filter_revision=rev)
            @test q.filter_revision == now
        end
        F.state_loads[] = :throw
    end
end

@testset "an exact resume reuses the checkpoint's normalization" begin
    F.return_code[] = 5
    accepted = [0.1, 0.1]
    F.MmaccsaBackend.script[] = (accepted, [])
    for (ck_norm, used) in ((7.0, 7.0), (nothing, 2.0))
        F.resume_state[] = (; p=[0.4, 0.6], g=0.3, iteration=2,
            metadata=resume_md(current_epoch=1, epoch_evals=2, g_norm=ck_norm))
        p = F.problem()
        g, design = F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0],
            resume_from="injected")
        @test g == sum(abs2, accepted) / used
    end
    F.MmaccsaBackend.script[] = nothing
end

@testset "constraint scale is recorded and passed to resume validation" begin
    F.return_code[] = 5
    F.resume_state[] = (; p=[0.4, 0.6], g=0.3, iteration=2,
        metadata=resume_md(current_epoch=1, epoch_evals=2))
    # Recorded only when constraints are on (NLopt, which ignores the scale, records none).
    for (backend, cons, cs, want) in ((:standalone_mma, true, 1e8, 1e8),
                                      (:standalone_ccsaq, true, 1.0, 1.0),
                                      (:standalone_mma, false, 1e8, nothing))
        empty!(F.checkpoints)
        p = F.problem(); p.foundry_mode = true
        F.optimize!(p; backend, max_iter=5, β_schedule=[8.0], use_constraints=cons,
            constraint_scale=cs, resume_from="injected", backup=true, backup_every=0,
            backup_path="injected")
        @test F.validated[].constraint_scale === want
        @test all(c.constraint_scale === want && c.g_norm == 2.0 for c in F.checkpoints)
    end
end

@testset "invalid KS sharpness is refused before any evaluation" begin
    for bad in (0.0, -1.0, NaN, -Inf)
        p = F.problem()
        @test_throws ArgumentError F.optimize!(p; backend=:standalone_mma, max_iter=5,
            β_schedule=[8.0], robust_samples=[(0.0, 0.0)], robust_sharpness=bad)
        @test p.iteration == 0
    end
end

# regression: a 2D problem without the DCT-I filter is refused before the problem changes.
@testset "use_dct=false is refused before optimize! and run_epoch! change the problem" begin
    F.return_code[] = 5
    populated() = begin
        p = F.problem(); p.foundry_mode = true
        p.control.use_dct = false
        p.p .= [0.31, 0.62]; p.g = 7.5; p.iteration = 9
        append!(p.g_history, [1.0, 2.0, 3.0]); p.g_best = 3.0; p.p_best .= [0.2, 0.8]
        p
    end
    snapshot(p) = (copy(p.p), p.g, p.iteration, copy(p.g_history), p.g_best, copy(p.p_best), p.control.β, p.filter_revision)
    # Refused with nothing changed, no checkpoint written, and neither the flat normalization nor the objective run.
    refuses(f, p) = begin
        before = snapshot(p)
        empty!(F.checkpoints); F.g_at_first_eval[] = nothing
        norm_before = F.norm_calls[]
        err = try
            f()
            nothing
        catch e
            e
        end
        err isa ArgumentError && snapshot(p) == before && isempty(F.checkpoints) && F.g_at_first_eval[] === nothing &&
            F.norm_calls[] == norm_before
    end
    # Filter or constraints reach the 2D filter: refused, nothing touched (a fresh history would have been emptied).
    for kwargs in ((; use_constraints=false), (; use_constraints=true))
        for backend in (:nlopt, :standalone_mma, :standalone_ccsaq)
            p = populated()
            @test refuses(() -> F.optimize!(p; backend, max_iter=5, β_schedule=[8.0], backup=true,
                backup_every=0, backup_path="injected", kwargs...), p)
            p = populated()
            @test refuses(() -> F.run_epoch!(p, 5, kwargs.use_constraints, 1e-15; backend), p)
        end
    end
    # Filtering off and no constraints: the 2D filter is never reached, so the run is allowed.
    p = populated(); p.control.use_filter = false
    g, design = F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0])
    @test length(design) == 2
    # Filtering off with constraints: the linewidth constraints filter, so this is refused.
    p = populated(); p.control.use_filter = false
    @test refuses(() -> F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], use_constraints=true), p)
    # regression: with two β epochs and backups the first, unconstrained epoch would write a checkpoint before the
    # constrained last epoch reached the filter; the refusal comes before that checkpoint.
    p = populated(); p.control.use_filter = false
    @test refuses(() -> F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0, 16.0],
        use_constraints=true, backup=true, backup_every=0, backup_path="injected"), p)
    # A resume would load the checkpoint into the problem first: refused before that load.
    F.resume_state[] = (; p=[0.9, 0.9], g=1.0, iteration=3, metadata=nothing)
    p = populated()
    @test refuses(() -> F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0],
        resume_from="injected"), p)
    F.resume_state[] = nothing
    # regression: a resume whose checkpoint has completed every epoch would evaluate nothing, but optimize! checks the
    # requested configuration before it loads any checkpoint, so it is refused as well.
    F.resume_state[] = (; p=[0.9, 0.9], g=1.5, iteration=5,
        metadata=resume_md(current_epoch=1, epoch_evals=5, g_norm=2.0))
    p = populated()
    F.resume_kwargs[] = nothing
    @test refuses(() -> F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0],
        resume_from="injected"), p)
    @test F.resume_kwargs[] === nothing      # the checkpoint was never loaded
    # The same completed resume with use_dct=true returns the saved score and design and evaluates nothing.
    p = populated(); p.control.use_dct = true
    norm_before = F.norm_calls[]; F.g_at_first_eval[] = nothing
    g, design = F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0], resume_from="injected")
    @test g == 1.5 && design == [0.9, 0.9]
    @test F.norm_calls[] == norm_before && F.g_at_first_eval[] === nothing
    F.resume_state[] = nothing
    # 3D problems never read use_dct.
    p = populated(); p.foundry_mode = false
    g, design = F.optimize!(p; backend=:standalone_mma, max_iter=5, β_schedule=[8.0])
    @test length(design) == 2
end
