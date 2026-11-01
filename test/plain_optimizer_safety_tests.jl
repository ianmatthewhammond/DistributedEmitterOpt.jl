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
end
problem() = OptimizationProblem([0.4,0.6], 0.0, zeros(2), 0, Float64[], -Inf,
    zeros(2), FakeControl(8.0,1e-6), (;env=(;logger_cfg=nothing)), "unused", false)
const checkpoints = NamedTuple[]
const freed = Ref(0)
const return_code = Ref(-1)
struct MmaState end
free_state!(::MmaState) = (freed[] += 1; nothing)
_state_path(args...) = nothing
_iteration_checkpoint_metadata(args...) = nothing
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

module MmaccsaBackend
using ..PlainOptimizerFixture: return_code
function minimize!(algorithm, objective, p; maxeval, kwargs...)
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
