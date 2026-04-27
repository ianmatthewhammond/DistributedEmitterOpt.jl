"""
    Optimizer

Beta-continuation topology optimization. Two backends:

- `:nlopt` (default) — `NLopt.jl`, algorithm `LD_CCSAQ`. Maximization.
- `:standalone_mma`, `:standalone_ccsaq` — the standalone NLopt-MMA-CCSA
  shared library (see `MmaccsaBackend`). Minimization (objective is
  negated internally), with process-portable checkpoints written next to
  the JLD2 backup at `<backup_path>.mma_state`.
"""

import NLopt
import NLopt: Opt, optimize
import JLD2

using .MmaccsaBackend: MmaState, save_state, load_state, free_state!

"""
    flat_substrate_norm(prob::OptimizationProblem) -> Float64

Compute objective baseline at the flat-substrate design (`p = 0`), matching
legacy normalization behavior (`g / g_base`).
"""
function flat_substrate_norm(prob::OptimizationProblem)
    p_flat = zeros(Float64, length(prob.p))
    g_norm = objective_and_gradient!(Float64[], p_flat, prob)
    if !isfinite(g_norm) || g_norm <= 0.0
        log_warn(:objective, "Flat-substrate normalization baseline is invalid; using 1.0"; g_norm)
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
    resume_from::Union{Nothing,String}=nothing,
    backend::Symbol=:nlopt)
    backend in (:nlopt, :standalone_mma, :standalone_ccsaq) ||
        throw(ArgumentError("backend must be :nlopt, :standalone_mma, or :standalone_ccsaq (got :$backend)"))
    with_run_logger(prob.pde.env.logger_cfg; root=prob.root) do
        p_opt = copy(prob.p)
        g_opt = 0.0
        ckpt_path = isnothing(backup_path) ? joinpath(prob.root, "results_backup.jld2") : backup_path

        if !isnothing(resume_from)
            resume_from_checkpoint!(prob, resume_from)
            p_opt .= prob.p
            g_opt = prob.g
        end

        g_norm = flat_substrate_norm(prob)
        log_info(:objective, "Flat-substrate normalization baseline"; g_norm)

        # Initialize history
        if empty_history && isnothing(resume_from)
            empty!(prob.g_history)
            prob.iteration = 0
        elseif empty_history && !isnothing(resume_from)
            log_info(:objective, "resume_from provided: preserving loaded g_history")
        end

        start_epoch = if !isnothing(resume_from) && prob.iteration > 0
            div(prob.iteration, max_iter) + 1
        else
            1
        end
        if start_epoch > 1
            log_info(:epoch, "Resuming: skipping $(start_epoch - 1) completed epoch(s)";
                start_epoch, total_iterations=prob.iteration)
        end

        for (epoch, β) in enumerate(β_schedule)
            epoch < start_epoch && continue
            log_info(:epoch, "Epoch start"; epoch, β)
            log_debug(:memory, "epoch memory snapshot (start)";
                epoch,
                total_mem_mb=Sys.total_memory() / 2^20,
                free_mem_mb=Sys.free_memory() / 2^20,
                prob_size_mb=sizeof(prob) * 1e-6)

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
                g_norm, backup, backup_every, backup_path=ckpt_path, backend, epoch)

            prob.p .= p_opt
            prob.g = g_opt

            log_info(:epoch, "Epoch done"; epoch, g_opt_norm=g_opt)
            log_debug(:memory, "epoch memory snapshot (end)";
                epoch,
                total_mem_mb=Sys.total_memory() / 2^20,
                free_mem_mb=Sys.free_memory() / 2^20,
                prob_size_mb=sizeof(prob) * 1e-6)
        end

        if backup
            save_checkpoint(prob, ckpt_path)
        end

        return g_opt, p_opt
    end
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

    with_run_logger(prob.pde.env.logger_cfg; root=prob.root) do
        p_opt = copy(prob.p)
        g_opt = 0.0

        for (epoch, β) in enumerate(β_schedule)
            log_info(:epoch, "Eigen epoch start"; epoch, β)

            prob.control.β = β

            g_opt, p_opt, _ = run_epoch_eigen!(prob, max_iter, use_constraints, tol)

            prob.p .= p_opt
            prob.g = g_opt

            log_info(:epoch, "Eigen epoch done"; epoch, g_opt)
        end

        return g_opt, p_opt
    end
end

# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

"""Run one epoch of optimization at fixed beta."""
function run_epoch!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    backend::Symbol=:nlopt,
    epoch::Int=0)

    if backend === :nlopt
        return _run_epoch_nlopt!(prob, max_iter, use_constraints, tol;
            g_norm, backup, backup_every, backup_path)
    else
        algorithm = backend === :standalone_mma ? :mma : :ccsaq
        return _run_epoch_standalone!(prob, max_iter, use_constraints, tol;
            g_norm, backup, backup_every, backup_path, algorithm, epoch)
    end
end

# ---------------------------------------------------------------------------
# Backend: NLopt.jl (original path)
# ---------------------------------------------------------------------------

function _run_epoch_nlopt!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
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
        iter = prob.iteration + 1
        free_mem_before = Sys.free_memory() / 2^20
        log_debug(:memory, "objective callback (before)";
            iter,
            free_mem_mb=free_mem_before,
            prob_size_mb=sizeof(prob) * 1e-6)

        g_raw = objective_and_gradient!(grad, p, prob)
        free_mem_after = Sys.free_memory() / 2^20
        g = g_raw / g_norm
        grad ./= g_norm
        ret_grad .= grad

        # Keep cached state consistent with returned normalized objective/gradient.
        prob.g = g
        if !isempty(prob.∇g)
            prob.∇g .= grad
        end

        log_debug(:memory, "objective callback (after)";
            iter,
            free_mem_mb=free_mem_after,
            delta_free_mem_mb=(free_mem_after - free_mem_before))
        log_info(:objective, "normalized objective";
            iter,
            g_norm=g,
            g_raw,
            grad_norm=norm(grad))

        next_iteration!(prob)
        log_iteration!(prob, g, p; backup, backup_every, backup_path)

        return g
    end

    if use_constraints
        log_info(:constraints, "enabling constraints for this epoch")
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

    log_info(:epoch, "NLopt completed"; ret, evals=opt.numevals)

    return g_opt, p_opt, ret_grad
end

# ---------------------------------------------------------------------------
# Backend: standalone NLopt-MMA-CCSA (minimization, with checkpoint state)
# ---------------------------------------------------------------------------

# State path layout: <backup_path>.epoch<N>.mma_state. Each epoch runs in a
# fresh state because beta-continuation changes the objective shape — the
# MMA asymptote/sigma history from epoch k-1 isn't meaningful for epoch k.
# The on-disk state lets a job killed mid-epoch resume the *same* epoch
# with identical algorithm state.
_state_path(backup_path::Nothing, epoch::Int) = nothing
_state_path(backup_path::String, epoch::Int) = backup_path * ".epoch$(epoch).mma_state"

function _run_epoch_standalone!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    algorithm::Symbol=:mma,
    epoch::Int=0)
    np = length(prob.p)
    ret_grad = zeros(Float64, np)

    state_path = _state_path(backup_path, epoch)
    state = if state_path !== nothing && isfile(state_path)
        log_info(:epoch, "standalone backend: resuming from on-disk state"; state_path)
        load_state(state_path)
    else
        MmaState()
    end

    # Standalone minimizes; DEO maximizes. Negate objective + gradient on the
    # boundary, but keep `prob.g` and `prob.g_history` in DEO's max convention.
    objective = function (p, grad)
        iter = prob.iteration + 1
        free_mem_before = Sys.free_memory() / 2^20
        log_debug(:memory, "objective callback (before)";
            iter,
            free_mem_mb=free_mem_before,
            prob_size_mb=sizeof(prob) * 1e-6)

        # `grad` is empty when the C side passes NULL (gradient-free probe).
        # NLopt-MMA never does this for LD algorithms, but defend anyway.
        scratch = isempty(grad) ? zeros(Float64, length(p)) : grad
        g_raw = objective_and_gradient!(scratch, p, prob)
        free_mem_after = Sys.free_memory() / 2^20
        g = g_raw / g_norm
        scratch ./= g_norm

        ret_grad .= scratch
        prob.g = g
        if !isempty(prob.∇g)
            prob.∇g .= scratch
        end

        if !isempty(grad)
            grad .*= -1.0   # min -f  (after prob.∇g is saved)
        end

        log_debug(:memory, "objective callback (after)";
            iter,
            free_mem_mb=free_mem_after,
            delta_free_mem_mb=(free_mem_after - free_mem_before))
        log_info(:objective, "normalized objective";
            iter,
            g_norm=g,
            g_raw,
            grad_norm=norm(scratch))

        next_iteration!(prob)
        log_iteration!(prob, g, p; backup, backup_every, backup_path)

        # Save standalone checkpoint on the same cadence as the JLD2 backup.
        # NB: this fires only between MMA outer iterations (the C side calls
        # the objective at well-defined points), so the state snapshot is
        # consistent.
        if backup && backup_every > 0 && state_path !== nothing &&
            (prob.iteration % backup_every == 0) && state.ref[] != C_NULL
            try
                save_state(state_path, state)
            catch err
                log_warn(:objective, "failed to save standalone checkpoint"; err=string(err))
            end
        end

        return -g   # standalone minimizes
    end

    constraints = Tuple{Function,Float64}[]
    if use_constraints
        log_info(:constraints, "enabling constraints for this epoch (standalone backend)")
        if prob.foundry_mode
            sim0 = default_sim(prob.sim)
            push!(constraints, ((p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), 1e-8))
            push!(constraints, ((p, g) -> glc_void(p, g; sim=sim0, control=prob.control), 1e-8))
        else
            sim0 = default_sim(prob.sim)
            obj = (; sim=sim0, control=prob.control, cache_pump=default_pool(prob.pool).filter_cache)
            push!(constraints, ((p, g) -> glc_solid_fe(p, g, obj), 1e-8))
            push!(constraints, ((p, g) -> glc_void_fe(p, g, obj), 1e-8))
        end
    end

    p_opt = copy(prob.p)
    lb = zeros(Float64, np)
    ub = ones(Float64, np)

    ret, _, _ = MmaccsaBackend.minimize!(algorithm, objective, p_opt;
        lb=lb, ub=ub, maxeval=max_iter, ftol_rel=tol, state=state,
        constraints=constraints)

    g_opt = prob.g   # in DEO's max convention (set inside the callback)

    if backup && state_path !== nothing && state.ref[] != C_NULL
        try
            save_state(state_path, state)
        catch err
            log_warn(:epoch, "failed to save standalone checkpoint at epoch end"; err=string(err))
        end
    end

    free_state!(state)

    log_info(:epoch, "standalone $(algorithm) completed"; ret, evals=prob.iteration, g_opt)

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

    log_info(:epoch, "NLopt (eigen) completed"; ret, evals=opt.numevals)

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
        log_info(:objective, "iteration checkpoint"; iter, g=round(g, sigdigits=4))
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
    log_info(:objective, "gradient test"; rel_error)

    return ∇g, ∇g_fd, rel_error
end
