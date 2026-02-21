"""
    Optimizer

NLopt-based optimization with beta-continuation scheduling (CCSAQ algorithm).
"""

import NLopt
import NLopt: Opt, optimize
import JLD2

"""
    flat_substrate_norm(prob::OptimizationProblem) -> Float64

Compute objective baseline at the flat-substrate design (`p = 0`), matching
legacy normalization behavior (`g / g_base`).
"""
function flat_substrate_norm(prob::OptimizationProblem)
    p_flat = zeros(Float64, length(prob.p))
    g_norm = objective_and_gradient!(Float64[], p_flat, prob)
    if !isfinite(g_norm) || g_norm <= 0.0
        @warn "Flat-substrate normalization baseline is invalid; falling back to 1.0" g_norm
        return 1.0
    end
    return g_norm
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    optimize!(prob; kwargs...) -> (g_opt, p_opt)

Run topology optimization with beta-continuation.

Keyword arguments:
- `max_iter` -- iterations per beta value (default 40)
- `β_schedule` -- projection steepness values to sweep
- `α_schedule` -- optional loss schedule (same length as beta_schedule)
- `use_constraints` -- enable linewidth constraints on the final beta epoch only
- `tol` -- relative tolerance for convergence
- `backup` -- enable autosaving `(p, g_history)` checkpoints
- `backup_every` -- autosave interval (iterations)
- `backup_path` -- optional checkpoint path (default `joinpath(prob.root, "results_backup.jld2")`)
- `resume_from` -- optional checkpoint path to resume from
"""
function optimize!(prob::OptimizationProblem;
    max_iter::Int=40,
    β_schedule::Vector{Float64}=[8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    α_schedule::Union{Vector{Float64},Nothing}=nothing,
    use_constraints::Bool=false,
    tol::Float64=1e-15,
    empty_history::Bool=true,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    resume_from::Union{Nothing,String}=nothing)

    p_opt = copy(prob.p)
    g_opt = 0.0
    ckpt_path = isnothing(backup_path) ? joinpath(prob.root, "results_backup.jld2") : backup_path

    if !isnothing(resume_from)
        resume_from_checkpoint!(prob, resume_from)
        p_opt .= prob.p
        g_opt = prob.g
    end

    g_norm = flat_substrate_norm(prob)
    @info "Flat-substrate normalization baseline: g_norm = $g_norm"

    # Initialize history
    if empty_history && isnothing(resume_from)
        empty!(prob.g_history)
        prob.iteration = 0
    elseif empty_history && !isnothing(resume_from)
        @info "resume_from provided: preserving loaded g_history"
    end

    for (epoch, β) in enumerate(β_schedule)
        @info "Epoch $epoch: β = $β"
        @show Sys.total_memory() / 2^20
        @show Sys.free_memory() / 2^20
        @show sizeof(prob) * 1e-6
        flush(stdout)
        Libc.flush_cstdio()

        prob.control.β = β

        if !isnothing(α_schedule) && epoch <= length(α_schedule)
            prob.pde = MaxwellProblem(
                env=prob.pde.env,
                inputs=prob.pde.inputs,
                outputs=prob.pde.outputs,
                α_loss=α_schedule[epoch]
            )
        end

        epoch_use_constraints = use_constraints && (epoch == length(β_schedule))
        g_opt, p_opt, _ = run_epoch!(prob, max_iter, epoch_use_constraints, tol;
            g_norm, backup, backup_every, backup_path=ckpt_path)

        prob.p .= p_opt
        prob.g = g_opt

        @info "  Epoch $epoch done: g = $g_opt"
        @show Sys.total_memory() / 2^20
        @show Sys.free_memory() / 2^20
        @show sizeof(prob) * 1e-6
        flush(stdout)
        Libc.flush_cstdio()
    end

    if backup
        save_checkpoint(prob, ckpt_path)
    end

    return g_opt, p_opt
end

# ---------------------------------------------------------------------------
# Eigen optimization entry point (gradient TODO)
# ---------------------------------------------------------------------------

"""
    optimize!(prob::EigenOptimizationProblem; kwargs...) -> (g_opt, p_opt)

Run eigenvalue-based optimization with beta-continuation.
Note: eigen sensitivities are TODO and will error during gradient evaluation.
"""
function optimize!(prob::EigenOptimizationProblem;
    max_iter::Int=40,
    β_schedule::Vector{Float64}=[8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    use_constraints::Bool=false,
    tol::Float64=1e-15)

    p_opt = copy(prob.p)
    g_opt = 0.0

    for (epoch, β) in enumerate(β_schedule)
        @info "Epoch $epoch: β = $β"

        prob.control.β = β

        g_opt, p_opt, _ = run_epoch_eigen!(prob, max_iter, use_constraints, tol)

        prob.p .= p_opt
        prob.g = g_opt

        @info "  Epoch $epoch done: g = $g_opt"
    end

    return g_opt, p_opt
end

# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

"""Run one epoch of optimization at fixed beta."""
function run_epoch!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing)
    np = length(prob.p)
    ret_grad = zeros(Float64, np)

    opt = Opt(:LD_CCSAQ, np)
    opt.lower_bounds = 0.0
    opt.upper_bounds = 1.0
    opt.ftol_rel = tol
    opt.maxeval = max_iter

    opt.max_objective = function (p, grad)
        free_mem_before = Sys.free_memory() / 2^20
        @show free_mem_before
        @show sizeof(prob) * 1e-6
        flush(stdout)
        Libc.flush_cstdio()

        g_raw = objective_and_gradient!(grad, p, prob)
        free_mem_after = Sys.free_memory() / 2^20
        @show free_mem_after
        g = g_raw / g_norm
        grad ./= g_norm
        ret_grad .= grad

        # Keep cached state consistent with returned normalized objective/gradient.
        prob.g = g
        if !isempty(prob.∇g)
            prob.∇g .= grad
        end

        @show g
        flush(stdout)
        Libc.flush_cstdio()

        next_iteration!(prob)
        log_iteration!(prob, g, p; backup, backup_every, backup_path)

        return g
    end

    if use_constraints
        if prob.foundry_mode
            sim0 = default_sim(prob.sim)
            NLopt.inequality_constraint!(opt,
                (p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), 1e-8)
            NLopt.inequality_constraint!(opt,
                (p, g) -> glc_void(p, g; sim=sim0, control=prob.control), 1e-8)
        else
            sim0 = default_sim(prob.sim)
            obj = (; sim=sim0, control=prob.control, cache_pump=default_pool(prob.pool).filter_cache)
            NLopt.inequality_constraint!(opt,
                (p, g) -> glc_solid_fe(p, g, obj), 1e-8)
            NLopt.inequality_constraint!(opt,
                (p, g) -> glc_void_fe(p, g, obj), 1e-8)
        end
    end

    (g_opt, p_opt, ret) = optimize(opt, prob.p)

    @info "  NLopt: $ret after $(opt.numevals) evaluations"

    return g_opt, p_opt, ret_grad
end

"""Run one epoch of eigen optimization at fixed beta."""
function run_epoch_eigen!(prob::EigenOptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64)
    if use_constraints
        error("TODO: constraints for eigen optimization are not implemented")
    end

    np = length(prob.p)
    ret_grad = zeros(Float64, np)

    opt = Opt(:LD_CCSAQ, np)
    opt.lower_bounds = 0.0
    opt.upper_bounds = 1.0
    opt.ftol_rel = tol
    opt.maxeval = max_iter

    opt.max_objective = function (p, grad)
        g = objective_and_gradient!(grad, p, prob)
        ret_grad .= grad

        next_iteration!(prob)
        log_iteration!(prob, g, p)

        return g
    end

    (g_opt, p_opt, ret) = optimize(opt, prob.p)

    @info "  NLopt: $ret after $(opt.numevals) evaluations"

    return g_opt, p_opt, ret_grad
end

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

"""Initialize history tracking."""
function init_history!(prob::OptimizationProblem)
    prob.g_history = Float64[]
end

"""Log iteration to console and optionally save checkpoint."""
function log_iteration!(prob::OptimizationProblem, g, p;
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing)
    iter = prob.iteration

    # Store history
    push!(prob.g_history, g)

    if iter % 5 == 0
        @info "Iter $iter: g = $(round(g, sigdigits=4))"
    end

    if backup && backup_every > 0 && (iter % backup_every == 0)
        path = isnothing(backup_path) ? joinpath(prob.root, "results_backup.jld2") : backup_path
        save_checkpoint(prob, path)
    end
end

"""Save optimization checkpoint."""
function save_checkpoint(prob::OptimizationProblem, filepath::String)
    mkpath(prob.root)
    JLD2.save(filepath, Dict(
        "p" => copy(prob.p),
        "g_history" => copy(prob.g_history)
    ))
end

"""Resume optimization state from a checkpoint containing `(p, g_history)`."""
function resume_from_checkpoint!(prob::OptimizationProblem, filepath::String)
    if !isfile(filepath)
        throw(ArgumentError("Checkpoint file not found: $filepath"))
    end

    data = JLD2.load(filepath)
    if !haskey(data, "p") || !haskey(data, "g_history")
        throw(ArgumentError("Checkpoint must contain keys 'p' and 'g_history': $filepath"))
    end

    p_loaded = vec(data["p"])
    if length(p_loaded) != length(prob.p)
        throw(ArgumentError("Checkpoint p length $(length(p_loaded)) != problem DOFs $(length(prob.p))"))
    end

    prob.p .= p_loaded
    prob.g_history = Float64.(vec(data["g_history"]))
    prob.iteration = length(prob.g_history)
    prob.g = isempty(prob.g_history) ? 0.0 : prob.g_history[end]
    return prob
end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

"""Single evaluation (no optimization loop)."""
function evaluate(prob::OptimizationProblem, p::Vector{Float64})
    ∇g = zeros(Float64, length(p))
    g = objective_and_gradient!(∇g, p, prob)
    return g, ∇g
end

"""Finite-difference gradient check."""
function test_gradient(prob::OptimizationProblem, p::Vector{Float64}; δ::Float64=1e-6)
    g0, ∇g = evaluate(prob, p)

    ∇g_fd = zeros(length(p))
    for i in 1:length(p)
        p_plus = copy(p)
        p_plus[i] += δ
        g_plus, _ = evaluate(prob, p_plus)
        ∇g_fd[i] = (g_plus - g0) / δ
    end

    rel_error = norm(∇g - ∇g_fd) / (norm(∇g) + 1e-12)
    @info "Gradient test: relative error = $rel_error"

    return ∇g, ∇g_fd, rel_error
end
