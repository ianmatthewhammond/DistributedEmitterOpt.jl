# =============================================================================
# Shared manual NLopt LD_CCSAQ epoch — used IDENTICALLY by the OLD and the
# NEW-:nlopt arms.  The optimizer configuration is bit-identical between arms;
# only the `objfun!` closure differs (OLD e3.stepobjective vs NEW
# objective_and_gradient!).  Any divergence is therefore in the assembly, not
# the optimizer.  Saves per-eval history.csv and design snapshots at iters
# 1..10 and every 10th.  Requires config.jl already included.
# =============================================================================

using NLopt
using Printf
using LinearAlgebra: norm
import JLD2

# objfun!(p, grad) -> g_raw ; fills `grad` with the RAW gradient when !isempty.
function run_manual_epoch(objfun!, p0::Vector{Float64}, g_norm::Float64;
        snapdir::String, historypath::String,
        max_iter::Int, tol::Float64, label::String)
    n = length(p0)
    mkpath(snapdir)
    history = NamedTuple[]

    opt = NLopt.Opt(:LD_CCSAQ, n)
    opt.lower_bounds = 0.0
    opt.upper_bounds = 1.0
    opt.ftol_rel = tol
    opt.maxeval = max_iter

    evalc = Ref(0)
    opt.max_objective = function (p, grad)
        g_raw = objfun!(p, grad)
        if !isempty(grad)
            grad ./= g_norm
        end
        g = g_raw / g_norm
        evalc[] += 1
        it = evalc[]
        gnorm = isempty(grad) ? 0.0 : norm(grad)
        push!(history, (iter=it, g_raw=g_raw, g=g, grad_norm=gnorm,
                        p_min=minimum(p), p_max=maximum(p)))
        if should_snapshot(it)
            JLD2.save(joinpath(snapdir, "snap_" * lpad(it, 4, '0') * ".jld2"),
                Dict("iter" => it, "g_raw" => g_raw, "g" => g, "p" => collect(Float64, p)))
        end
        @printf("[%s] iter=%d g_raw=% .8e g=% .8e ||grad||=% .6e p=[%.4f,%.4f]\n",
            label, it, g_raw, g, gnorm, minimum(p), maximum(p))
        flush(stdout)
        return g
    end

    t0 = time()
    (g_opt, p_opt, ret) = NLopt.optimize(opt, copy(p0))
    elapsed = time() - t0

    open(historypath, "w") do io
        println(io, "iter,g_raw,g,grad_norm,p_min,p_max")
        for r in history
            println(io, string(r.iter, ",", r.g_raw, ",", r.g, ",",
                              r.grad_norm, ",", r.p_min, ",", r.p_max))
        end
    end
    JLD2.save(joinpath(snapdir, "final.jld2"),
        Dict("g_opt" => g_opt, "p_opt" => collect(Float64, p_opt),
             "ret" => string(ret), "numevals" => opt.numevals, "g_norm" => g_norm))
    @printf("[%s] DONE ret=%s evals=%d elapsed=%.1fs g_opt(norm)=% .8e\n",
        label, string(ret), opt.numevals, elapsed, g_opt)
    flush(stdout)
    return g_opt, p_opt, ret, history
end
