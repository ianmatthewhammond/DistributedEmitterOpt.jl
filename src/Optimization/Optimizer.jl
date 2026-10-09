"""
    Optimizer

Beta-continuation topology optimization. Two backends:

- `:nlopt` (default) — `NLopt.jl`, algorithm `LD_CCSAQ`. Maximization.
- `:standalone_mma`, `:standalone_ccsaq` — the standalone NLopt-MMA-CCSA
  shared library (see `MmaccsaBackend`). Minimization (objective is
  negated internally), with committed process-portable checkpoints written
  next to the JLD2 backup.
"""

import NLopt
import NLopt: Opt, optimize
import JLD2

using .MmaccsaBackend: MmaState, save_state, load_state, free_state!

const OPTIMIZER_CHECKPOINT_VERSION = 2

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

function _checkpoint_alpha_schedule(value)
    value === nothing && return nothing
    return Float64.(vec(value))
end

# Robust (worst-case sidewall) configuration is part of the OBJECTIVE definition, so it is
# recorded in the checkpoint beside beta_schedule/backend/use_constraints and a resume with a
# different configuration is refused.
_robust_samples_matrix(samples) = samples === nothing ? nothing :
    Float64[samples[i][j] for i in eachindex(samples), j in 1:2]
# The aggregation (`robust_mode`: :ks softmin or :epigraph) and, for the epigraph, its
# tolerances and the linewidth eps/tolerance are part of the objective/constraint
# definition too. A checkpoint without `robust_mode` predates
# the epigraph and was written by the KS path. The KS comparison is unchanged: with
# ck_mode = mode = :ks and no epigraph config, the result is the one it always was.
function _same_robust_config(ck_samples, ck_sharpness, samples, sharpness;
                             ck_mode::Symbol=:ks, mode::Symbol=:ks,
                             ck_epigraph=nothing, epigraph=nothing)::Bool
    ck_mode == mode || return false
    isequal(ck_epigraph, epigraph) || return false
    a = ck_samples === nothing ? nothing : Float64.(ck_samples)
    b = _robust_samples_matrix(samples)
    a === nothing && b === nothing && return true
    (a === nothing) != (b === nothing) && return false
    size(a) == size(b) || return false
    a == b || return false
    return ck_sharpness === nothing ? sharpness === nothing : (sharpness !== nothing && Float64(ck_sharpness) == sharpness)
end

# The sharpness recorded/validated for a configuration: meaningless (nothing) when robust
# is off or the aggregation is the epigraph.
_robust_sharpness_record(samples, mode::Symbol, sharpness) =
    (samples === nothing || mode === :epigraph) ? nothing : sharpness

"""Epigraph configuration recorded in checkpoints and compared on resume (`nothing` for :ks)."""
_epigraph_config(mode::Symbol, tol, sigma, eps, linewidth_tol) = mode === :epigraph ?
    (tol=Float64(tol), sigma=Float64(sigma), eps=Float64(eps), linewidth_tol=Float64(linewidth_tol)) :
    nothing

function _checkpoint_epigraph_config(data)
    keys_ = ("epigraph_tol", "epigraph_sigma", "linewidth_eps", "linewidth_tol")
    all(k -> haskey(data, k) && data[k] !== nothing, keys_) || return nothing
    return (tol=Float64(data["epigraph_tol"]), sigma=Float64(data["epigraph_sigma"]),
            eps=Float64(data["linewidth_eps"]), linewidth_tol=Float64(data["linewidth_tol"]))
end
_checkpoint_optional_float(data, key) =
    (haskey(data, key) && data[key] !== nothing) ? Float64(data[key]) : nothing

function _same_optional_schedule(a, b)::Bool
    aa = _checkpoint_alpha_schedule(a)
    bb = _checkpoint_alpha_schedule(b)
    aa === nothing && return bb === nothing
    bb === nothing && return false
    return aa == bb
end

function _checkpoint_metadata(data)
    version = haskey(data, "checkpoint_version") ? Int(data["checkpoint_version"]) : 1
    if version < OPTIMIZER_CHECKPOINT_VERSION
        return (;
            version,
            exact_resume=false,
            current_epoch=nothing,
            epoch_evals=nothing,
            max_iter=nothing,
            beta_schedule=nothing,
            alpha_schedule=nothing,
            backend=nothing,
            use_constraints=nothing,
            mma_state_path=nothing,
            robust_samples=nothing,
            robust_sharpness=nothing,
            robust_mode=:ks,
            epigraph_config=nothing,
            epigraph_tau=nothing,
            epigraph_T=nothing,
        )
    end

    required = (
        "current_epoch", "epoch_evals", "max_iter", "beta_schedule",
        "alpha_schedule", "backend", "use_constraints",
    )
    missing = [key for key in required if !haskey(data, key)]
    if !isempty(missing)
        throw(ArgumentError("v$version checkpoint is missing metadata keys: $(join(missing, ", "))"))
    end

    return (;
        version,
        exact_resume=true,
        current_epoch=Int(data["current_epoch"]),
        epoch_evals=Int(data["epoch_evals"]),
        max_iter=Int(data["max_iter"]),
        beta_schedule=Float64.(vec(data["beta_schedule"])),
        alpha_schedule=_checkpoint_alpha_schedule(data["alpha_schedule"]),
        backend=Symbol(String(data["backend"])),
        use_constraints=Bool(data["use_constraints"]),
        mma_state_path=haskey(data, "mma_state_path") ? String(data["mma_state_path"]) : nothing,
        robust_samples=(haskey(data, "robust_samples") && data["robust_samples"] !== nothing) ? Matrix{Float64}(data["robust_samples"]) : nothing,
        robust_sharpness=(haskey(data, "robust_sharpness") && data["robust_sharpness"] !== nothing) ? Float64(data["robust_sharpness"]) : nothing,
        robust_mode=(haskey(data, "robust_mode") && data["robust_mode"] !== nothing) ? Symbol(String(data["robust_mode"])) : :ks,
        epigraph_config=_checkpoint_epigraph_config(data),
        epigraph_tau=_checkpoint_optional_float(data, "epigraph_tau"),
        epigraph_T=_checkpoint_optional_float(data, "epigraph_T"),
    )
end

function _validate_checkpoint_metadata!(metadata;
    filepath::AbstractString,
    max_iter::Int,
    beta_schedule::Vector{Float64},
    alpha_schedule::Union{Vector{Float64},Nothing},
    backend::Symbol,
    use_constraints::Bool,
    robust_samples=nothing,
    robust_sharpness=nothing,
    robust_mode::Symbol=:ks,
    epigraph_config=nothing)

    metadata.exact_resume || return nothing

    metadata.max_iter == max_iter ||
        throw(ArgumentError("Checkpoint $filepath was written with max_iter=$(metadata.max_iter), got max_iter=$max_iter"))
    metadata.beta_schedule == beta_schedule ||
        throw(ArgumentError("Checkpoint $filepath beta_schedule does not match this resume call"))
    _same_optional_schedule(metadata.alpha_schedule, alpha_schedule) ||
        throw(ArgumentError("Checkpoint $filepath alpha_schedule does not match this resume call"))
    metadata.backend == backend ||
        throw(ArgumentError("Checkpoint $filepath was written with backend=$(metadata.backend), got backend=$backend"))
    metadata.use_constraints == use_constraints ||
        throw(ArgumentError("Checkpoint $filepath was written with use_constraints=$(metadata.use_constraints), got use_constraints=$use_constraints"))
    sharp = _robust_sharpness_record(robust_samples, robust_mode, robust_sharpness)
    _same_robust_config(metadata.robust_samples, metadata.robust_sharpness, robust_samples, sharp;
                        ck_mode=metadata.robust_mode, mode=robust_mode,
                        ck_epigraph=metadata.epigraph_config, epigraph=epigraph_config) ||
        throw(ArgumentError("Checkpoint $filepath robust sidewall configuration (mode=$(metadata.robust_mode), samples=$(metadata.robust_samples), sharpness=$(metadata.robust_sharpness), epigraph=$(metadata.epigraph_config)) does not match this resume call (mode=$robust_mode, samples=$(_robust_samples_matrix(robust_samples)), sharpness=$sharp, epigraph=$epigraph_config)"))
    return nothing
end

function _checkpoint_resume_progress(metadata, total_iterations::Int, max_iter::Int, n_epochs::Int)
    if metadata.exact_resume
        current_epoch = metadata.current_epoch
        epoch_evals = metadata.epoch_evals
        if current_epoch < 1 || current_epoch > n_epochs
            throw(ArgumentError("Checkpoint current_epoch=$current_epoch is outside beta_schedule length $n_epochs"))
        end
        epoch_evals >= 0 ||
            throw(ArgumentError("Checkpoint epoch_evals=$epoch_evals must be non-negative"))

        if epoch_evals >= max_iter
            return (;
                exact_resume=true,
                start_epoch=current_epoch + 1,
                epoch_evals_start=0,
                completed_checkpoint_epoch=current_epoch,
            )
        end

        return (;
            exact_resume=true,
            start_epoch=current_epoch,
            epoch_evals_start=epoch_evals,
            completed_checkpoint_epoch=current_epoch - 1,
        )
    end

    legacy_start_epoch = total_iterations > 0 ? div(total_iterations, max_iter) + 1 : 1
    return (;
        exact_resume=false,
        start_epoch=legacy_start_epoch,
        epoch_evals_start=0,
        completed_checkpoint_epoch=legacy_start_epoch - 1,
    )
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
- `constraint_scale` -- multiply constraint values, gradients, and tolerances by this
  factor (default 1.0, no-op). A pure reparameterization: the feasible set is unchanged.
  Set large (e.g. 1e8) for constrained β=Inf epochs, where the raw constraint terms
  (~1e-6) are otherwise below one ULP of the objective inside the CCSA dual and the
  dual line search fails (ret=-1). Known issues at constrained β=Inf, independent of
  this fix: (1) `p_best` is objective-only selection, so under an active length-scale
  constraint it systematically picks the most constraint-violating design in the
  trajectory — prefer the final iterate; (2) each epoch resets CCSA ρ and σ to
  defaults, discarding the previous epoch's learned trust-region state.
- `ccsa_verbose` -- verbosity level passed through to the standalone CCSA backend
- `dual_algorithm` -- solver for the standalone backends' CCSA dual subproblem: `:mma`
  (default, as in upstream NLopt) or `:lbfgs` (Luksan PLIS). Ignored by `backend=:nlopt`.
  The choice is not stored in the checkpoint; a run may resume under either solver.
- `robust_samples` -- `nothing`, or a vector of (slope, z0) sidewall samples to take the
  worst case over
- `robust_mode` -- how the worst case is aggregated: `:ks` (default; KS softmin with
  `robust_sharpness`, fixed scale g_norm) or `:epigraph` (standalone backends only; one
  constraint per sample, see `EpigraphSidewall.jl` and `_run_epoch_standalone_epigraph!`)
- `epigraph_sigma` -- initial CCSA σ of the epigraph variable τ (default 1.0)
- `epigraph_tol` -- CCSA tolerance of the epigraph constraints τ − gᵢ/(g_norm T) ≤ 0
- `epigraph_linewidth_tol` -- tolerance of the NORMALIZED linewidth constraints
  M/ε − 1 ≤ 0 in epigraph mode (requires `constraint_scale == 1/b1`)
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
    backend::Symbol=:nlopt,
    rho_init::Float64=1.0,
    ccsa_verbose::Integer=0,
    dual_algorithm::Symbol=:mma,
    constraint_scale::Float64=1.0,
    sigma_init::Union{Nothing,Vector{Float64}}=nothing,
    # Worst-case design over sidewall profiles. `nothing` keeps the
    # ordinary single evaluation; otherwise a vector of (slope, z0) samples, each a full
    # forward + adjoint, aggregated by a KS soft-min with FIXED scale g_norm.
    robust_samples::Union{Nothing,Vector{Tuple{Float64,Float64}}}=nothing,
    robust_sharpness::Float64=200.0,
    robust_mode::Symbol=:ks,
    epigraph_sigma::Float64=1.0,
    epigraph_tol::Float64=1e-8,
    epigraph_linewidth_tol::Float64=1e-3)
    backend in (:nlopt, :standalone_mma, :standalone_ccsaq) ||
        throw(ArgumentError("backend must be :nlopt, :standalone_mma, or :standalone_ccsaq (got :$backend)"))
    robust_mode in (:ks, :epigraph) ||
        throw(ArgumentError("robust_mode must be :ks or :epigraph (got :$robust_mode)"))
    dual_algorithm in (:mma, :lbfgs) ||
        throw(ArgumentError("dual_algorithm must be :mma or :lbfgs (got :$dual_algorithm)"))
    if robust_mode === :epigraph
        backend === :nlopt &&
            throw(ArgumentError("robust_mode=:epigraph needs a standalone backend (:standalone_ccsaq or :standalone_mma), got :nlopt"))
        robust_samples === nothing &&
            throw(ArgumentError("robust_mode=:epigraph requires robust_samples"))
        (isfinite(epigraph_sigma) && epigraph_sigma > 0) ||
            throw(ArgumentError("epigraph_sigma must be finite and > 0 (got $epigraph_sigma)"))
    end
    epigraph_config = _epigraph_config(robust_mode, epigraph_tol, epigraph_sigma,
                                       prob.control.b1, epigraph_linewidth_tol)
    with_run_logger(prob.pde.env.logger_cfg; root=prob.root) do
        p_opt = copy(prob.p)
        g_opt = 0.0
        ckpt_path = isnothing(backup_path) ? joinpath(prob.root, "results_backup.jld2") : backup_path
        resume_metadata = nothing

        if !isnothing(resume_from)
            resume_metadata = resume_from_checkpoint!(prob, resume_from; return_metadata=true)
            p_opt .= prob.p
            g_opt = prob.g
        end

        g_norm = flat_substrate_norm(prob)
        log_info(:objective, "Flat-substrate normalization baseline"; g_norm)

        # Initialize history
        if empty_history && isnothing(resume_from)
            empty!(prob.g_history)
            prob.iteration = 0
            prob.g_best = -Inf
            fill!(prob.p_best, 0.0)
        elseif empty_history && !isnothing(resume_from)
            log_info(:objective, "resume_from provided: preserving loaded g_history")
        end

        start_epoch = 1
        resume_epoch_evals = 0
        completed_checkpoint_epoch = 0
        if !isnothing(resume_from)
            _validate_checkpoint_metadata!(resume_metadata;
                filepath=resume_from,
                max_iter,
                beta_schedule=β_schedule,
                alpha_schedule=α_schedule,
                backend,
                use_constraints,
                robust_samples,
                robust_sharpness,
                robust_mode,
                epigraph_config)
            progress = _checkpoint_resume_progress(resume_metadata, prob.iteration, max_iter, length(β_schedule))
            start_epoch = progress.start_epoch
            resume_epoch_evals = progress.epoch_evals_start
            completed_checkpoint_epoch = progress.completed_checkpoint_epoch

            if progress.exact_resume
                if start_epoch <= length(β_schedule)
                    log_info(:epoch, "Resuming epoch $start_epoch with $resume_epoch_evals/$max_iter evals done";
                        start_epoch, epoch_evals=resume_epoch_evals, remaining=max_iter - resume_epoch_evals)
                else
                    log_info(:epoch, "Resume checkpoint already completed all epochs";
                        total_iterations=prob.iteration)
                end
            else
                log_warn(:epoch, "Legacy checkpoint lacks epoch metadata; using global-iteration resume heuristic";
                    start_epoch, total_iterations=prob.iteration)
            end
        end
        if start_epoch > 1 && start_epoch <= length(β_schedule)
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
            epoch_evals_start = epoch == start_epoch ? resume_epoch_evals : 0
            checkpoint_mma_state_path =
                (!isnothing(resume_metadata) &&
                 resume_metadata.exact_resume &&
                 resume_metadata.epoch_evals < max_iter &&
                 epoch == resume_metadata.current_epoch) ?
                resume_metadata.mma_state_path : nothing
            # Epigraph: a mid-epoch resume rebuilds x = [p; τ] from the checkpoint's τ and
            # keeps the stage's T; a fresh epoch recomputes both from its starting design.
            epigraph_resume = checkpoint_mma_state_path === nothing ? nothing :
                (T=resume_metadata.epigraph_T, tau=resume_metadata.epigraph_tau)
            remaining_evals = max(max_iter - epoch_evals_start, 0)
            remaining_evals == 0 && continue
            g_opt, p_opt, _ = run_epoch!(prob, remaining_evals, epoch_use_constraints, tol;
                g_norm, backup, backup_every, backup_path=ckpt_path, backend, epoch,
                epoch_evals_start,
                epoch_max_iter=max_iter,
                checkpoint_beta_schedule=β_schedule,
                checkpoint_alpha_schedule=α_schedule,
                checkpoint_backend=backend,
                checkpoint_use_constraints=use_constraints,
                checkpoint_mma_state_path, rho_init, sigma_init, ccsa_verbose, dual_algorithm,
                constraint_scale, robust_samples, robust_sharpness,
                robust_mode, epigraph_config, epigraph_resume)

            prob.p .= p_opt
            prob.g = g_opt
            completed_checkpoint_epoch = epoch

            log_info(:epoch, "Epoch done"; epoch, g_opt_norm=g_opt)
            log_debug(:memory, "epoch memory snapshot (end)";
                epoch,
                total_mem_mb=Sys.total_memory() / 2^20,
                free_mem_mb=Sys.free_memory() / 2^20,
                prob_size_mb=sizeof(prob) * 1e-6)

            if backup
                save_checkpoint(prob, ckpt_path;
                    current_epoch=epoch,
                    epoch_evals=max_iter,
                    max_iter,
                    beta_schedule=β_schedule,
                    alpha_schedule=α_schedule,
                    backend,
                    use_constraints,
                    robust_samples,
                    robust_sharpness=_robust_sharpness_record(robust_samples, robust_mode, robust_sharpness),
                    robust_mode,
                    epigraph_config)
            end
        end

        if backup
            final_epoch = clamp(completed_checkpoint_epoch, 1, length(β_schedule))
            save_checkpoint(prob, ckpt_path;
                current_epoch=final_epoch,
                epoch_evals=max_iter,
                max_iter,
                beta_schedule=β_schedule,
                alpha_schedule=α_schedule,
                backend,
                use_constraints,
                robust_samples,
                robust_sharpness=_robust_sharpness_record(robust_samples, robust_mode, robust_sharpness),
                robust_mode,
                epigraph_config)
        end

        # Return the best design seen across all epochs, not the last iterate
        if prob.g_best > g_opt
            log_info(:objective, "Returning best design (g_best > g_final)";
                g_best=prob.g_best, g_final=g_opt)
            return prob.g_best, copy(prob.p_best)
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
    epoch::Int=0,
    epoch_evals_start::Int=0,
    epoch_max_iter::Int=max_iter,
    checkpoint_beta_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_alpha_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_backend::Union{Symbol,Nothing}=nothing,
    checkpoint_use_constraints::Union{Bool,Nothing}=nothing,
    checkpoint_mma_state_path::Union{String,Nothing}=nothing,
    rho_init::Float64=1.0,
    ccsa_verbose::Integer=0,
    dual_algorithm::Symbol=:mma,
    constraint_scale::Float64=1.0,
    sigma_init::Union{Nothing,Vector{Float64}}=nothing,
    robust_samples::Union{Nothing,Vector{Tuple{Float64,Float64}}}=nothing,
    robust_sharpness::Float64=200.0,
    robust_mode::Symbol=:ks,
    epigraph_config=nothing,
    epigraph_resume=nothing)

    backend === :nlopt || log_info(:epoch, "standalone CCSA dual solver"; epoch, dual_algorithm)
    if robust_mode === :epigraph
        backend === :nlopt &&
            throw(ArgumentError("robust_mode=:epigraph needs a standalone backend, got :nlopt"))
        epigraph_config === nothing &&
            throw(ArgumentError("robust_mode=:epigraph requires an epigraph_config"))
        algorithm = backend === :standalone_mma ? :mma : :ccsaq
        return _run_epoch_standalone_epigraph!(prob, max_iter, use_constraints;
            g_norm, backup, backup_every, backup_path, algorithm, epoch,
            epoch_evals_start, epoch_max_iter, checkpoint_beta_schedule,
            checkpoint_alpha_schedule, checkpoint_backend,
            checkpoint_use_constraints, checkpoint_mma_state_path,
            rho_init, sigma_init, ccsa_verbose, dual_algorithm, constraint_scale,
            robust_samples, epigraph_config, epigraph_resume)
    end
    robust_mode === :ks ||
        throw(ArgumentError("robust_mode must be :ks or :epigraph (got :$robust_mode)"))

    if backend === :nlopt
        return _run_epoch_nlopt!(prob, max_iter, use_constraints, tol;
            g_norm, backup, backup_every, backup_path, epoch, epoch_evals_start,
            epoch_max_iter, checkpoint_beta_schedule, checkpoint_alpha_schedule,
            checkpoint_backend, checkpoint_use_constraints, robust_samples, robust_sharpness)
    else
        algorithm = backend === :standalone_mma ? :mma : :ccsaq
        return _run_epoch_standalone!(prob, max_iter, use_constraints, tol;
            g_norm, backup, backup_every, backup_path, algorithm, epoch,
            epoch_evals_start, epoch_max_iter, checkpoint_beta_schedule,
            checkpoint_alpha_schedule, checkpoint_backend,
            checkpoint_use_constraints, checkpoint_mma_state_path,
            rho_init, sigma_init, ccsa_verbose, dual_algorithm, constraint_scale,
            robust_samples, robust_sharpness)
    end
end

# ---------------------------------------------------------------------------
# Backend: NLopt.jl (original path)
# ---------------------------------------------------------------------------

function _run_epoch_nlopt!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    epoch::Int=0,
    epoch_evals_start::Int=0,
    epoch_max_iter::Int=max_iter,
    checkpoint_beta_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_alpha_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_backend::Union{Symbol,Nothing}=nothing,
    checkpoint_use_constraints::Union{Bool,Nothing}=nothing,
    robust_samples::Union{Nothing,Vector{Tuple{Float64,Float64}}}=nothing,
    robust_sharpness::Float64=200.0)
    np = length(prob.p)
    ret_grad = zeros(Float64, np)
    epoch_evals = Ref(epoch_evals_start)

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

        # `scale = g_norm` is the run's fixed reference, so the KS sharpness is a
        # constant of the run and the aggregate is a stationary objective. Never
        # derive it from the sample values — see RobustSidewall.jl.
        g_raw = if robust_samples === nothing
            objective_and_gradient!(grad, p, prob)
        else
            g_agg, gs, w = robust_sidewall_objective!(grad, p, prob, robust_samples;
                sharpness=robust_sharpness, scale=g_norm)
            log_info(:objective, "robust sidewall worst case";
                iter,
                samples=robust_samples,
                g_samples=gs,
                weights=w,
                worst_raw=minimum(gs),
                ks_softness=g_agg - minimum(gs),
                sharpness=robust_sharpness)
            g_agg
        end
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
        epoch_evals[] += 1
        checkpoint_metadata = _iteration_checkpoint_metadata(epoch, epoch_evals[],
            epoch_max_iter, checkpoint_beta_schedule, checkpoint_alpha_schedule,
            checkpoint_backend, checkpoint_use_constraints, robust_samples, robust_sharpness)
        log_iteration!(prob, g, p; backup, backup_every, backup_path, checkpoint_metadata)

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
_state_path(backup_path::Nothing, epoch::Int, epoch_evals::Int) = nothing
_state_path(backup_path::String, epoch::Int, epoch_evals::Int) =
    backup_path * ".epoch$(epoch).eval$(epoch_evals).mma_state"

function _run_epoch_standalone!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    algorithm::Symbol=:mma,
    epoch::Int=0,
    epoch_evals_start::Int=0,
    epoch_max_iter::Int=max_iter,
    checkpoint_beta_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_alpha_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_backend::Union{Symbol,Nothing}=nothing,
    checkpoint_use_constraints::Union{Bool,Nothing}=nothing,
    checkpoint_mma_state_path::Union{String,Nothing}=nothing,
    rho_init::Float64=1.0,
    ccsa_verbose::Integer=0,
    dual_algorithm::Symbol=:mma,
    constraint_scale::Float64=1.0,
    sigma_init::Union{Nothing,Vector{Float64}}=nothing,
    robust_samples::Union{Nothing,Vector{Tuple{Float64,Float64}}}=nothing,
    robust_sharpness::Float64=200.0)
    np = length(prob.p)
    ret_grad = zeros(Float64, np)
    epoch_evals = Ref(epoch_evals_start)

    legacy_state_path = _state_path(backup_path, epoch)
    resume_state_path = checkpoint_mma_state_path === nothing ? legacy_state_path : checkpoint_mma_state_path
    state = if resume_state_path !== nothing && isfile(resume_state_path)
        log_info(:epoch, "standalone backend: resuming from on-disk state"; state_path=resume_state_path)
        load_state(resume_state_path)
    elseif checkpoint_mma_state_path !== nothing
        throw(ArgumentError("Checkpoint metadata references missing MMA state file: $checkpoint_mma_state_path"))
    else
        MmaState()
    end

    next_checkpoint_evals = Ref(backup_every > 0 ?
        backup_every * (div(epoch_evals_start, backup_every) + 1) : typemax(Int))

    safe_checkpoint = function (x_committed, minf, local_nevals, k)
        backup || return nothing
        backup_every > 0 || return nothing
        backup_path === nothing && return nothing

        total_epoch_evals = epoch_evals_start + local_nevals
        total_epoch_evals >= next_checkpoint_evals[] || return nothing

        try
            prob.p .= x_committed
            prob.g = -minf

            state_path = _state_path(backup_path, epoch, total_epoch_evals)
            save_state(state_path, state)

            checkpoint_metadata = _iteration_checkpoint_metadata(epoch, total_epoch_evals,
                epoch_max_iter, checkpoint_beta_schedule, checkpoint_alpha_schedule,
                checkpoint_backend, checkpoint_use_constraints, robust_samples, robust_sharpness)
            if checkpoint_metadata === nothing
                save_checkpoint(prob, backup_path)
            else
                save_checkpoint(prob, backup_path; checkpoint_metadata..., mma_state_path=state_path)
            end
            log_info(:objective, "safe standalone checkpoint";
                epoch, epoch_evals=total_epoch_evals, k, state_path)

            while next_checkpoint_evals[] <= total_epoch_evals
                next_checkpoint_evals[] += backup_every
            end
        catch err
            log_warn(:objective, "failed to save safe standalone checkpoint"; err=string(err))
        end

        return nothing
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
        # Worst case over sidewall profiles: `scale = g_norm` is the run's FIXED reference,
        # so the KS sharpness is a constant of the run and the aggregate is stationary.
        # KS is taken in RAW units; the g/g_norm normalization below is unchanged.
        g_raw = if robust_samples === nothing
            objective_and_gradient!(scratch, p, prob)
        else
            g_agg, gs, w = robust_sidewall_objective!(scratch, p, prob, robust_samples;
                sharpness=robust_sharpness, scale=g_norm)
            log_info(:objective, "robust sidewall worst case";
                iter,
                samples=robust_samples,
                g_samples=gs,
                weights=w,
                worst_raw=minimum(gs),
                ks_softness=g_agg - minimum(gs),
                sharpness=robust_sharpness)
            g_agg
        end
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
        epoch_evals[] += 1
        checkpoint_metadata = _iteration_checkpoint_metadata(epoch, epoch_evals[],
            epoch_max_iter, checkpoint_beta_schedule, checkpoint_alpha_schedule,
            checkpoint_backend, checkpoint_use_constraints, robust_samples, robust_sharpness)
        log_iteration!(prob, g, p; backup=false, backup_every, backup_path, checkpoint_metadata)

        return -g   # standalone minimizes
    end

    constraints = Tuple{Function,Float64}[]
    if use_constraints
        log_info(:constraints, "enabling constraints for this epoch (standalone backend)")
        if prob.foundry_mode
            sim0 = default_sim(prob.sim)
            # constraint_scale rescales value, gradient, AND tolerance together — a pure
            # reparameterization (the feasible set is unchanged). Needed at constrained
            # β=∞: the geometric constraints (~1e-6) contribute ~1e-15 to the CCSA dual
            # next to an objective of O(1e2), below one ULP of f — the dual line search
            # then cannot represent any decrease and exits with iterm=-6 → ret=-1.
            cs = constraint_scale
            scaled(f) = (p, g) -> begin
                v = f(p, g)
                cs == 1.0 && return v
                isempty(g) || (g .*= cs)
                return cs * v
            end
            push!(constraints, (scaled((p, g) -> glc_solid(p, g; sim=sim0, control=prob.control)), 1e-8 * cs))
            push!(constraints, (scaled((p, g) -> glc_void(p, g; sim=sim0, control=prob.control)), 1e-8 * cs))
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
        constraints=constraints,
        checkpoint_callback=safe_checkpoint,
        rho_init=rho_init, sigma_init=sigma_init, verbose=ccsa_verbose, dual_algorithm)

    g_opt = prob.g   # in DEO's max convention (set inside the callback)

    free_state!(state)

    log_info(:epoch, "standalone $(algorithm) completed"; ret, evals=prob.iteration, g_opt)

    return g_opt, p_opt, ret_grad
end

# ---------------------------------------------------------------------------
# Backend: standalone CCSA, EPIGRAPH worst case over sidewall samples
# ---------------------------------------------------------------------------

"""
    _run_epoch_standalone_epigraph!(prob, max_iter, use_constraints; ...) -> (g_opt, p_opt, grad)

One β-epoch of the epigraph worst case (formulation and evaluation-order facts in
`EpigraphSidewall.jl`). Variables x = [p; τ], minimize −τ subject to
τ − gᵢ(p)/(g_norm T) ≤ 0 per sample, plus optional normalized linewidth
constraints on p.

Conventions that differ from the KS path, each on purpose:
- `prob.g_history` records the TRUE worst case min_i gᵢ/g_norm at every evaluated
  point (preserving the normalized improvement criterion g/g_norm > 1); `prob.g` and the
  returned `g_opt` are T·τ at the committed point, a certified lower bound on it.
- `ftol_rel = 0`: τ can sit still for an iteration while p moves, and a relative
  f-tolerance would then stop the run with FTOL_REACHED.
- `rho_init` is read in the KS path's units (for example, 1/g_norm) and
  divided by T, i.e. expressed in τ units: ρ = rho_init / T.
- Every place x meets `prob.p` slices `x[1:np]` (checkpoint, returned design,
  `log_iteration!`). With x of length np+1, `prob.p .= x` throws.
- A negative CCSA return code throws (the backend does not), and a resumed state's
  (n, m) is checked against (np+1, nsamples [+2]) before C sees it.
- A failed checkpoint write is logged at the time and turned into an error when the
  epoch returns, instead of being swallowed.
"""
function _run_epoch_standalone_epigraph!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool;
    g_norm::Float64=1.0,
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    algorithm::Symbol=:ccsaq,
    epoch::Int=0,
    epoch_evals_start::Int=0,
    epoch_max_iter::Int=max_iter,
    checkpoint_beta_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_alpha_schedule::Union{Vector{Float64},Nothing}=nothing,
    checkpoint_backend::Union{Symbol,Nothing}=nothing,
    checkpoint_use_constraints::Union{Bool,Nothing}=nothing,
    checkpoint_mma_state_path::Union{String,Nothing}=nothing,
    rho_init::Float64=1.0,
    sigma_init::Union{Nothing,Vector{Float64}}=nothing,
    ccsa_verbose::Integer=0,
    dual_algorithm::Symbol=:mma,
    constraint_scale::Float64=1.0,
    robust_samples::Vector{Tuple{Float64,Float64}},
    epigraph_config,
    epigraph_resume=nothing)

    np = length(prob.p)
    n = np + 1
    ns = length(robust_samples)
    ns >= 1 || throw(ArgumentError("robust_mode=:epigraph: empty sample set"))
    m = ns + (use_constraints ? 2 : 0)
    ret_grad = zeros(Float64, np)
    epoch_evals = Ref(epoch_evals_start)
    cache = EpigraphCache(np, ns)

    if use_constraints
        prob.foundry_mode ||
            throw(ArgumentError("robust_mode=:epigraph: linewidth constraints are implemented for foundry (2D-DOF) mode only"))
        # The normalized tolerance only means M <= eps*(1+tol) if the constraint reads
        # M/eps - 1, i.e. constraint_scale * b1 == 1. Anything else silently changes the
        # admitted linewidth violation, so refuse it.
        isapprox(constraint_scale * prob.control.b1, 1.0; rtol=1e-12) ||
            throw(ArgumentError("robust_mode=:epigraph with linewidth constraints needs constraint_scale == 1/b1 " *
                "(constraint M/eps - 1 <= 0, tolerance $(epigraph_config.linewidth_tol)); got constraint_scale=$constraint_scale, b1=$(prob.control.b1)"))
    end

    # ── state, T, τ₀ ─────────────────────────────────────────────────────────────
    local state, T, τ0
    if checkpoint_mma_state_path !== nothing
        isfile(checkpoint_mma_state_path) ||
            throw(ArgumentError("Checkpoint metadata references missing MMA state file: $checkpoint_mma_state_path"))
        (epigraph_resume === nothing || epigraph_resume.T === nothing || epigraph_resume.tau === nothing) &&
            throw(ArgumentError("epigraph resume: checkpoint lacks epigraph_T/epigraph_tau; cannot rebuild x = [p; tau] exactly"))
        sn, sm = MmaccsaBackend.state_dims(checkpoint_mma_state_path)
        (sn, sm) == (n, m) ||
            throw(ArgumentError("epigraph resume: MMA state $checkpoint_mma_state_path was written for n=$sn, m=$sm " *
                "but this epoch has n=$n, m=$m (np=$np + tau; $ns sample constraints$(use_constraints ? " + 2 linewidth" : ""))"))
        hdr = MmaccsaBackend.state_header(checkpoint_mma_state_path)
        -hdr.minf == epigraph_resume.tau ||
            throw(ArgumentError("epigraph resume: checkpoint tau=$(epigraph_resume.tau) is not the MMA state's committed -minf=$(-hdr.minf); " *
                "the JLD2 and the state file are not from the same commit"))
        T = Float64(epigraph_resume.T)
        τ0 = Float64(epigraph_resume.tau)
        state = load_state(checkpoint_mma_state_path)
        log_info(:epoch, "epigraph: resuming from on-disk state";
            epoch, state_path=checkpoint_mma_state_path, T, tau=τ0, k=Int(hdr.k))
    else
        # A fresh epoch starts from a fresh state (the legacy <backup>.epochN.mma_state
        # path is a KS convention and is not consulted here).
        state = MmaState()
        T, τ0 = epigraph_seed!(cache, copy(prob.p), prob, robust_samples, g_norm)
        log_info(:epoch, "epigraph: stage start";
            epoch, T, tau0=τ0, g_samples=copy(cache.gs), g_norm, rho=rho_init / T)
    end

    stats = (tau=Float64[], worst=Float64[], gs=Vector{Float64}[], slacks=Vector{Float64}[])
    checkpoint_failures = Ref(0)

    next_checkpoint_evals = Ref(backup_every > 0 ?
        backup_every * (div(epoch_evals_start, backup_every) + 1) : typemax(Int))

    epigraph_md(evals, τ) = _iteration_checkpoint_metadata(epoch, evals,
        epoch_max_iter, checkpoint_beta_schedule, checkpoint_alpha_schedule,
        checkpoint_backend, checkpoint_use_constraints, robust_samples, nothing;
        robust_mode=:epigraph, epigraph_config, epigraph_tau=τ, epigraph_T=T)

    safe_checkpoint = function (x_committed, minf, local_nevals, k)
        backup || return nothing
        backup_every > 0 || return nothing
        backup_path === nothing && return nothing

        total_epoch_evals = epoch_evals_start + local_nevals
        total_epoch_evals >= next_checkpoint_evals[] || return nothing

        try
            τc = x_committed[np+1]
            τc == -minf || log_warn(:objective, "epigraph checkpoint: committed tau != -minf"; tau=τc, minf)
            prob.p .= view(x_committed, 1:np)
            prob.g = T * τc

            state_path = _state_path(backup_path, epoch, total_epoch_evals)
            save_state(state_path, state)

            md = epigraph_md(total_epoch_evals, τc)
            if md === nothing
                save_checkpoint(prob, backup_path)
            else
                save_checkpoint(prob, backup_path; md..., mma_state_path=state_path)
            end
            log_info(:objective, "safe standalone checkpoint (epigraph)";
                epoch, epoch_evals=total_epoch_evals, k, state_path, tau=τc, T)

            while next_checkpoint_evals[] <= total_epoch_evals
                next_checkpoint_evals[] += backup_every
            end
        catch err
            checkpoint_failures[] += 1
            log_error(:objective, "failed to save epigraph checkpoint"; err=string(err))
        end
        return nothing
    end

    # Objective: −τ. The sample solves happen here (the objective is evaluated first at
    # every point), and the constraints below read the same cache.
    objective = function (x, grad)
        iter = prob.iteration + 1
        gs, grads, hit = epigraph_samples!(cache, x, prob, robust_samples)
        τ = x[np+1]
        k = argmin(gs)
        worst = gs[k] / g_norm
        ret_grad .= grads[k] ./ g_norm

        prob.g = T * τ
        if !isempty(prob.∇g)
            prob.∇g .= ret_grad
        end
        if !isempty(grad)
            fill!(grad, 0.0)
            grad[np+1] = -1.0
        end

        slacks = gs ./ (g_norm * T) .- τ          # = −cᵢ ≥ 0 when feasible
        push!(stats.tau, τ); push!(stats.worst, worst)
        push!(stats.gs, copy(gs)); push!(stats.slacks, slacks)
        log_info(:objective, "epigraph worst case";
            iter,
            samples=robust_samples,
            g_samples=gs,
            worst_raw=gs[k],
            argmin=k,
            T, tau=τ, t=T * τ,
            slacks,
            cache_hit=hit)
        log_info(:objective, "normalized objective";
            iter,
            g_norm=worst,
            g_raw=gs[k],
            grad_norm=norm(ret_grad))

        next_iteration!(prob)
        epoch_evals[] += 1
        log_iteration!(prob, worst, Vector{Float64}(x[1:np]); backup=false, backup_every, backup_path,
            checkpoint_metadata=epigraph_md(epoch_evals[], τ))

        (epoch_evals[] - epoch_evals_start) >= _EPIGRAPH_ABORT_AFTER[] &&
            throw(EpigraphTestAbort(epoch_evals[]))
        return -τ
    end

    constraints = Tuple{Function,Float64}[]
    for i in 1:ns
        push!(constraints, (epigraph_constraint(cache, i, prob, robust_samples, g_norm, T), epigraph_config.tol))
    end
    if use_constraints
        log_info(:constraints, "enabling linewidth constraints for this epoch (epigraph, normalized)";
            eps=prob.control.b1, tol=epigraph_config.linewidth_tol)
        sim0 = default_sim(prob.sim)
        cs = constraint_scale
        push!(constraints, (epigraph_linewidth((p, g) -> glc_solid(p, g; sim=sim0, control=prob.control), np, cs),
                            epigraph_config.linewidth_tol))
        push!(constraints, (epigraph_linewidth((p, g) -> glc_void(p, g; sim=sim0, control=prob.control), np, cs),
                            epigraph_config.linewidth_tol))
    end

    σp = if sigma_init === nothing
        fill(0.5, np)      # the C default 0.5·(ub − lb) for p ∈ [0, 1]
    elseif length(sigma_init) == np
        sigma_init
    else
        throw(ArgumentError("sigma_init has length $(length(sigma_init)); epigraph mode expects np = $np (τ's σ is epigraph_sigma)"))
    end
    σ = [σp; epigraph_config.sigma]

    x = [copy(prob.p); τ0]
    lb = [zeros(Float64, np); 0.0]
    ub = [ones(Float64, np); Inf]

    ret = try
        r, _, _ = MmaccsaBackend.minimize!(algorithm, objective, x;
            lb=lb, ub=ub, maxeval=max_iter, ftol_rel=0.0, state=state,
            constraints=constraints,
            checkpoint_callback=safe_checkpoint,
            rho_init=rho_init / T, sigma_init=σ, verbose=ccsa_verbose, dual_algorithm)
        r
    finally
        _EPIGRAPH_LAST_STATS[] = (; epoch, T, tau0=τ0, nevals=epoch_evals[] - epoch_evals_start,
            solves=cache.solves, hits=cache.hits, misses=cache.misses, nsamples=ns, m,
            tau=stats.tau, worst=stats.worst, gs=stats.gs, slacks=stats.slacks,
            x_final=copy(x), checkpoint_failures=checkpoint_failures[])
        free_state!(state)
    end

    ret < 0 && throw(ErrorException("epigraph $(algorithm) returned ret=$ret (NLopt failure code) at epoch $epoch " *
        "after $(epoch_evals[] - epoch_evals_start) evals; refusing to report a design from a failed run"))
    (backup && checkpoint_failures[] > 0) &&
        throw(ErrorException("epigraph epoch $epoch: $(checkpoint_failures[]) checkpoint write(s) failed (see log)"))

    p_opt = x[1:np]
    g_opt = T * x[np+1]
    prob.g = g_opt

    log_info(:epoch, "standalone $(algorithm) (epigraph) completed";
        ret, evals=prob.iteration, g_opt, T, tau=x[np+1],
        solves=cache.solves, hits=cache.hits, misses=cache.misses)

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

function _iteration_checkpoint_metadata(epoch::Int, epoch_evals::Int, max_iter::Int,
    beta_schedule::Union{Vector{Float64},Nothing},
    alpha_schedule::Union{Vector{Float64},Nothing},
    backend::Union{Symbol,Nothing},
    use_constraints::Union{Bool,Nothing},
    robust_samples=nothing,
    robust_sharpness=nothing;
    robust_mode::Symbol=:ks,
    epigraph_config=nothing,
    epigraph_tau=nothing,
    epigraph_T=nothing)

    if epoch <= 0 || beta_schedule === nothing || backend === nothing || use_constraints === nothing
        return nothing
    end

    return (;
        current_epoch=epoch,
        epoch_evals,
        max_iter,
        beta_schedule,
        alpha_schedule,
        backend,
        robust_samples,
        robust_sharpness,
        use_constraints,
        robust_mode,
        epigraph_config,
        epigraph_tau,
        epigraph_T,
    )
end

"""Log iteration to console and optionally save checkpoint."""
function log_iteration!(prob::OptimizationProblem, g, p;
    backup::Bool=false,
    backup_every::Int=20,
    backup_path::Union{Nothing,String}=nothing,
    checkpoint_metadata::Union{NamedTuple,Nothing}=nothing)
    iter = prob.iteration

    # Store history
    push!(prob.g_history, g)

    # Track best design (maximization)
    if g > prob.g_best
        prob.g_best = g
        prob.p_best .= p
    end

    # Keep prob.p in sync with the current MMA iterate so mid-epoch
    # checkpoints capture the actual design, not the stale epoch-start value.
    prob.p .= p

    println(stdout, "[iter=$iter] g=$(round(g, sigdigits=6))  g_best=$(round(prob.g_best, sigdigits=6))")
    flush(stdout)

    log_info(:objective, "iteration checkpoint"; iter, g=round(g, sigdigits=4), g_best=round(prob.g_best, sigdigits=4))

    if backup && backup_every > 0 && (iter % backup_every == 0)
        path = isnothing(backup_path) ? joinpath(prob.root, "results_backup.jld2") : backup_path
        if checkpoint_metadata === nothing
            save_checkpoint(prob, path)
        else
            save_checkpoint(prob, path; checkpoint_metadata...)
        end
    end
end

"""Save optimization checkpoint."""
function save_checkpoint(prob::OptimizationProblem, filepath::String;
    current_epoch::Union{Int,Nothing}=nothing,
    epoch_evals::Union{Int,Nothing}=nothing,
    max_iter::Union{Int,Nothing}=nothing,
    beta_schedule::Union{Vector{Float64},Nothing}=nothing,
    alpha_schedule::Union{Vector{Float64},Nothing}=nothing,
    backend::Union{Symbol,Nothing}=nothing,
    use_constraints::Union{Bool,Nothing}=nothing,
    mma_state_path::Union{String,Nothing}=nothing,
    robust_samples=nothing,
    robust_sharpness=nothing,
    robust_mode::Symbol=:ks,
    epigraph_config=nothing,
    epigraph_tau=nothing,
    epigraph_T=nothing)

    metadata = (current_epoch, epoch_evals, max_iter, beta_schedule, backend, use_constraints)
    has_metadata = all(x -> x !== nothing, metadata)
    has_partial_metadata = any(x -> x !== nothing, metadata)
    if has_partial_metadata && !has_metadata
        throw(ArgumentError("checkpoint optimizer metadata must be provided as a complete set"))
    end
    if mma_state_path !== nothing && !has_metadata
        throw(ArgumentError("mma_state_path requires complete checkpoint optimizer metadata"))
    end

    mkpath(prob.root)
    data = Dict{String,Any}(
        "p" => copy(prob.p),
        "p_best" => copy(prob.p_best),
        "g_best" => prob.g_best,
        "g_history" => copy(prob.g_history),
    )

    if has_metadata
        data["checkpoint_version"] = OPTIMIZER_CHECKPOINT_VERSION
        data["current_epoch"] = current_epoch
        data["epoch_evals"] = epoch_evals
        data["max_iter"] = max_iter
        data["beta_schedule"] = copy(beta_schedule)
        data["alpha_schedule"] = alpha_schedule === nothing ? nothing : copy(alpha_schedule)
        data["backend"] = string(backend)
        data["use_constraints"] = use_constraints
        data["robust_samples"] = _robust_samples_matrix(robust_samples)
        data["robust_sharpness"] = (robust_samples === nothing || robust_sharpness === nothing) ? nothing : Float64(robust_sharpness)
        data["robust_mode"] = string(robust_mode)
        if epigraph_config !== nothing
            data["epigraph_tol"] = epigraph_config.tol
            data["epigraph_sigma"] = epigraph_config.sigma
            data["linewidth_eps"] = epigraph_config.eps
            data["linewidth_tol"] = epigraph_config.linewidth_tol
        end
        # Stage state, not configuration: x = [p; τ] at the committed point and the stage's
        # T. Needed to rebuild x bit-identically on a mid-epoch resume.
        epigraph_tau === nothing || (data["epigraph_tau"] = Float64(epigraph_tau))
        epigraph_T === nothing || (data["epigraph_T"] = Float64(epigraph_T))
        if mma_state_path !== nothing
            data["mma_state_path"] = mma_state_path
        end
    end

    JLD2.save(filepath, data)
end

"""Resume optimization state from a checkpoint containing `(p, g_history)` and optionally `(p_best, g_best)`."""
function resume_from_checkpoint!(prob::OptimizationProblem, filepath::String; return_metadata::Bool=false)
    if !isfile(filepath)
        throw(ArgumentError("Checkpoint file not found: $filepath"))
    end

    data = JLD2.load(filepath)
    metadata = _checkpoint_metadata(data)
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

    # Restore best-design tracking (backwards-compatible with old checkpoints)
    if haskey(data, "p_best") && haskey(data, "g_best")
        prob.p_best .= vec(data["p_best"])
        prob.g_best = Float64(data["g_best"])
    else
        prob.p_best .= p_loaded
        prob.g_best = prob.g
    end
    return return_metadata ? metadata : prob
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
