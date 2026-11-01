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
    zeros(2), FakeControl(8.0,1e-6), (;env=(;logger_cfg=nothing)), "unused", false, nothing)
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
flat_substrate_norm(prob) = 2.0
log_info(args...; kwargs...) = nothing
log_debug(args...; kwargs...) = nothing
log_warn(args...; kwargs...) = nothing
next_iteration!(prob) = (prob.iteration += 1)
function objective_and_gradient!(g, p, prob)
    g .= 2p
    return sum(abs2,p)
end
function log_iteration!(prob, g, p; kwargs...)
    push!(prob.g_history, g)
end
save_checkpoint(prob, path; kwargs...) = push!(checkpoints, (;kwargs...))
default_sim(sim) = sim
glc_solid(p,g; kwargs...) = (isempty(g) || (g .= 2.0); 0.2)
glc_void(p,g; kwargs...) = (isempty(g) || (g .= -3.0); -0.3)

module MmaccsaBackend
using ..PlainOptimizerFixture: return_code
const captured_constraints = Ref{Any}(nothing)
function minimize!(algorithm, objective, p; maxeval, kwargs...)
    captured_constraints[] = kwargs[:constraints]
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
                if name in (:optimize!, :run_epoch!, :_run_epoch_standalone!) &&
                   occursin("prob::OptimizationProblem", string(sig))
                    Core.eval(@__MODULE__, ex)
                    push!(loaded,name)
                end
            end
            return
        end
        foreach(visit,ex.args)
    end
    visit(parsed)
    loaded == Set([:optimize!,:run_epoch!,:_run_epoch_standalone!]) || error("missing production functions: $loaded")
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

if get(ENV,"TEST_RAW_TOLERANCE","0") == "1"
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
