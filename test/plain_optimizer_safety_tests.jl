# Execute the production optimizer orchestration with an injected backend and
# arithmetic objective. No FEM, filtering, library build, or package load is needed.
# Usage: julia test/plain_optimizer_safety_tests.jl [path/to/Optimizer.jl]
using Test, LinearAlgebra

module PlainOptimizerFixture
using LinearAlgebra
mutable struct FakeControl
    β::Float64
    b1::Float64
end
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
end
problem() = OptimizationProblem([0.4,0.6], 0.0, zeros(2), 0, Float64[], -Inf,
    zeros(2), FakeControl(8.0,1e-6), (;env=(;logger_cfg=nothing), inputs=nothing, outputs=nothing, α_loss=0.0),
    "unused", false, nothing)
# The loss schedule rebuilds prob.pde; only α_loss matters here.
MaxwellProblem(; env, inputs, outputs, α_loss) = (; env, inputs, outputs, α_loss)
const checkpoints = NamedTuple[]
const freed = Ref(0)
const return_code = Ref(-1)
struct MmaState end
free_state!(::MmaState) = (freed[] += 1; nothing)
_state_path(args...) = nothing
_iteration_checkpoint_metadata(args...; kwargs...) = nothing
_epigraph_config(args...) = nothing
_robust_sharpness_record(args...) = nothing
with_run_logger(f; kwargs...) = f()
with_run_logger(f, config; kwargs...) = f()
flat_substrate_norm(prob) = (prob === nothing || (prob.g = 999.0); 2.0)   # production's flat evaluation overwrites prob.g
log_info(args...; kwargs...) = nothing
log_debug(args...; kwargs...) = nothing
const warnings = Any[]
log_warn(args...; kwargs...) = (push!(warnings, (args, (;kwargs...))); nothing)
next_iteration!(prob) = (prob.iteration += 1)
function objective_and_gradient!(g, p, prob)
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
save_checkpoint(prob, path; kwargs...) = push!(checkpoints, (;kwargs...))
# Resume: the loaded design/objective and metadata come from `resume_state`; the
# production `_checkpoint_resume_progress` turns the metadata into a start epoch.
const resume_state = Ref{Any}(nothing)
const validated = Ref{Any}(nothing)
function resume_from_checkpoint!(prob, path; return_metadata=false)
    st = resume_state[]
    prob.p .= st.p; prob.g = st.g; prob.iteration = st.iteration
    return st.metadata
end
_validate_checkpoint_metadata!(md; kwargs...) = (validated[] = (;kwargs...); nothing)
default_sim(sim) = sim
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
                             :_set_epoch_state!) &&
                    occursin("prob::OptimizationProblem", string(sig))) || name === :_checkpoint_resume_progress
                    Core.eval(@__MODULE__, ex)
                    push!(loaded,name)
                end
            end
            return
        end
        foreach(visit,ex.args)
    end
    visit(parsed)
    loaded == Set([:optimize!,:run_epoch!,:_run_epoch_standalone!,:linewidth_feasibility,
                   :_set_epoch_state!,:_checkpoint_resume_progress]) || error("missing production functions: $loaded")
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
        for backend in (:standalone_mma,:standalone_ccsaq), raw in (1e-8,6e-14), cs in (1.0,1e8)
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
    constraint_scale=nothing)

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
    F.final_values[] = nothing
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
