# Helper script for test_backend_parity.jl. Run as:
#   julia parity_helper.jl <backend> <case> <algorithm> <mode> <result_path>
#
# backend   : "nlopt"   — NLopt.jl
#           : "stdlib"  — DEO's MmaccsaBackend (libmmaccsa)
# case      : "rosenbrock" | "mccormick" | "quadratic_constrained"
# algorithm : "mma" | "ccsaq"
# mode      : "full"      — single uninterrupted run
#           : "checkpoint" — split (40% evals → save → load → 60% more);
#                            only valid for backend=stdlib
# result_path : where to write the Serialization-format result file
#
# Both libraries export nlopt_* symbols, so each backend MUST run in its
# own process. This script is the per-process worker.
#
# Both backends are configured with explicit sigma_init equal to NLopt's
# default initial-step heuristic, so MMA's asymptote initialization is
# bit-identical between the two backends. With the same dual solver and
# the same C source for mma.c / ccsa_quadratic.c, results then match to
# numerical precision.

using Serialization

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

struct ParityCase
    name::String
    f::Function
    lb::Vector{Float64}
    ub::Vector{Float64}
    x0::Vector{Float64}
    constraints::Vector  # Vector of (g, tol) tuples
    maxeval::Int
end

function rosenbrock(x, grad)
    a = x[2] - x[1]^2
    b = 1.0 - x[1]
    if !isempty(grad)
        grad[1] = -400.0 * a * x[1] - 2.0 * b
        grad[2] = 200.0 * a
    end
    return 100.0 * a^2 + b^2
end

function mccormick(x, grad)
    a = x[1] + x[2]
    b = x[1] - x[2]
    if !isempty(grad)
        grad[1] = cos(a) + 2*b - 1.5
        grad[2] = cos(a) - 2*b + 2.5
    end
    return sin(a) + b^2 - 1.5*x[1] + 2.5*x[2] + 1.0
end

function quad(x, grad)
    if !isempty(grad)
        grad[1] = 2 * (x[1] - 0.25)
        grad[2] = 2 * (x[2] + 0.5)
    end
    return (x[1] - 0.25)^2 + (x[2] + 0.5)^2
end

function lin_constraint(x, grad)
    if !isempty(grad)
        grad[1] = 1.0
        grad[2] = 1.0
    end
    return x[1] + x[2] - 0.25
end

const CASES = Dict(
    "rosenbrock" => ParityCase("Rosenbrock", rosenbrock,
        [-2.0, -2.0], [2.0, 2.0], [0.0, 0.0],
        Tuple{Function,Float64}[], 60),
    "mccormick" => ParityCase("McCormick", mccormick,
        [-1.5, -3.0], [4.0, 4.0], [-0.5, -1.5],
        Tuple{Function,Float64}[], 60),
    "quadratic_constrained" => ParityCase("Quadratic+Linear", quad,
        [-2.0, -2.0], [2.0, 2.0], [0.8, 0.8],
        [(lin_constraint, 0.0)], 60),
)

# NLopt's default initial-step heuristic, replicated exactly so we can pass
# it explicitly to both backends. Source: NLopt src/api/options.c
# (nlopt_set_default_initial_step).
function nlopt_default_step(lb::Vector{Float64}, ub::Vector{Float64}, x::Vector{Float64})
    step = similar(x)
    for i in eachindex(x)
        s = Inf
        if isfinite(ub[i]) && isfinite(lb[i]) && (ub[i] - lb[i]) * 0.25 < s && ub[i] > lb[i]
            s = (ub[i] - lb[i]) * 0.25
        end
        if isfinite(ub[i]) && (ub[i] - x[i]) < s && ub[i] > x[i]
            s = (ub[i] - x[i]) * 0.75
        end
        if isfinite(lb[i]) && (x[i] - lb[i]) < s && x[i] > lb[i]
            s = (x[i] - lb[i]) * 0.75
        end
        if !isfinite(s)
            if isfinite(ub[i])
                s = abs(ub[i] - x[i])
            elseif isfinite(lb[i])
                s = abs(lb[i] - x[i])
            else
                s = max(abs(x[i]), 1.0)
            end
        end
        step[i] = s
    end
    return step
end

# ---------------------------------------------------------------------------
# Top-level conditional load (avoid world-age trap by keeping include/import
# at top level, not inside any function).
# ---------------------------------------------------------------------------

const BACKEND = ARGS[1]
const CASE_S  = ARGS[2]
const ALG_S   = ARGS[3]
const MODE    = ARGS[4]
const OUT     = ARGS[5]

if BACKEND == "nlopt"
    import NLopt
elseif BACKEND == "stdlib"
    include(joinpath(@__DIR__, "..", "src", "Optimization", "MmaccsaBackend.jl"))
    using .MmaccsaBackend
else
    error("unknown backend: $BACKEND")
end

# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

function run_nlopt(case::ParityCase, algorithm::Symbol)
    n = length(case.x0)
    alg = algorithm === :mma ? :LD_MMA : :LD_CCSAQ
    opt = NLopt.Opt(alg, n)
    opt.lower_bounds = case.lb
    opt.upper_bounds = case.ub
    opt.maxeval = case.maxeval
    opt.initial_step = nlopt_default_step(case.lb, case.ub, case.x0)

    # NLopt's MMA/CCSAQ default to NLOPT_LD_MMA for the dual subproblem
    # (recursive MMA-on-MMA), but the standalone hard-codes NLOPT_LD_LBFGS
    # (Luksan PLIS). Force NLopt to use LBFGS for the dual so the two
    # backends are running the same algorithm end-to-end.
    opt.params["dual_algorithm"] = Int(NLopt.LD_LBFGS)

    opt.min_objective = case.f
    for (g, tol) in case.constraints
        NLopt.inequality_constraint!(opt, g, tol)
    end
    fmin, p_opt, ret = NLopt.optimize(opt, copy(case.x0))
    return (; backend="nlopt", case=case.name, algorithm=String(algorithm), mode="full",
            ret=String(ret), fmin=fmin, p=p_opt, nevals=opt.numevals)
end

function run_stdlib_full(case::ParityCase, algorithm::Symbol)
    state = MmaState()
    p = copy(case.x0)
    sigma = nlopt_default_step(case.lb, case.ub, case.x0)
    ret, fmin, p_opt = minimize!(algorithm, case.f, p;
        lb=case.lb, ub=case.ub, maxeval=case.maxeval,
        constraints=case.constraints, sigma_init=sigma, state=state)
    free_state!(state)
    return (; backend="stdlib", case=case.name, algorithm=String(algorithm), mode="full",
            ret=Int(ret), fmin=fmin, p=p_opt, nevals=case.maxeval)
end

function run_stdlib_checkpoint(case::ParityCase, algorithm::Symbol)
    target_split = max(1, div(2 * case.maxeval, 5))
    sigma = nlopt_default_step(case.lb, case.ub, case.x0)

    # Phase 1: run until target_split evals. The algorithm finishes the
    # current outer iteration, so actual evals may slightly exceed target.
    evals_phase1 = Ref(0)
    f_phase1 = function(x, grad)
        evals_phase1[] += 1
        return case.f(x, grad)
    end

    state = MmaState()
    p = copy(case.x0)
    minimize!(algorithm, f_phase1, p;
        lb=case.lb, ub=case.ub, maxeval=target_split,
        constraints=case.constraints, sigma_init=sigma, state=state)

    path = tempname() * ".mma_state"
    save_state(path, state)
    free_state!(state)

    # Phase 2: remaining budget = total - actual phase 1 evals.
    remaining = case.maxeval - evals_phase1[]
    loaded = load_state(path)
    ret, fmin, p_opt = minimize!(algorithm, case.f, p;
        lb=case.lb, ub=case.ub, maxeval=remaining,
        constraints=case.constraints, sigma_init=sigma, state=loaded)
    free_state!(loaded)
    rm(path; force=true)

    return (; backend="stdlib", case=case.name, algorithm=String(algorithm), mode="checkpoint",
            ret=Int(ret), fmin=fmin, p=p_opt, nevals=evals_phase1[] + remaining)
end

# ---------------------------------------------------------------------------
# Dispatch and emit result
# ---------------------------------------------------------------------------

algorithm = Symbol(ALG_S)
case = CASES[CASE_S]

result = if BACKEND == "nlopt"
    MODE == "full" || error("nlopt backend only supports mode=full (got $MODE)")
    run_nlopt(case, algorithm)
elseif BACKEND == "stdlib"
    if MODE == "full"
        run_stdlib_full(case, algorithm)
    elseif MODE == "checkpoint"
        run_stdlib_checkpoint(case, algorithm)
    else
        error("unknown mode: $MODE")
    end
end

open(io -> serialize(io, result), OUT, "w")
println("OK ", result.backend, "/", result.case, "/", result.algorithm, "/", result.mode,
        " ret=", result.ret, " fmin=", result.fmin,
        " p=", result.p, " nevals=", result.nevals)
