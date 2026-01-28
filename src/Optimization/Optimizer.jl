"""
    Optimizer

NLopt-based optimization with β-continuation scheduling.
Supports CCSAQ (conservative convex separable approximation with quadratic).
"""

import NLopt
import NLopt: Opt, optimize

# ═══════════════════════════════════════════════════════════════════════════════
# Main optimization entry point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    optimize!(prob::OptimizationProblem; kwargs...) → (g_opt, p_opt)

Run topology optimization with β-continuation for OptimizationProblem.

## Keyword Arguments
- `max_iter` — Maximum iterations per β value (default: 40)
- `β_schedule` — Sequence of projection steepness values
- `α_schedule` — Optional loss schedule (same length as β_schedule)
- `use_constraints` — Enable linewidth constraints
- `tol` — Relative tolerance for convergence
"""
function optimize!(prob::OptimizationProblem;
    max_iter::Int=40,
    β_schedule::Vector{Float64}=[8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    α_schedule::Union{Vector{Float64},Nothing}=nothing,
    use_constraints::Bool=false,
    tol::Float64=1e-8)

    p_opt = copy(prob.p)
    g_opt = 0.0

    # β-continuation loop
    for (epoch, β) in enumerate(β_schedule)
        @info "Epoch $epoch: β = $β"

        # Update control
        prob.control.β = β

        # Update loss if scheduled
        if !isnothing(α_schedule) && epoch <= length(α_schedule)
            prob.pde = MaxwellProblem(
                env=prob.pde.env,
                inputs=prob.pde.inputs,
                outputs=prob.pde.outputs,
                α_loss=α_schedule[epoch]
            )
        end

        # Run epoch
        g_opt, p_opt, _ = run_epoch!(prob, max_iter, use_constraints, tol)

        # Update state
        prob.p .= p_opt
        prob.g = g_opt

        @info "  Epoch $epoch complete: g = $g_opt"
    end

    return g_opt, p_opt
end

# ═══════════════════════════════════════════════════════════════════════════════
# Single epoch
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_epoch!(prob, max_iter, use_constraints, tol) → (g_opt, p_opt, grad)

Run one epoch of optimization at fixed β.
"""
function run_epoch!(prob::OptimizationProblem, max_iter::Int, use_constraints::Bool, tol::Float64)
    np = length(prob.p)
    ret_grad = zeros(Float64, np)

    # Setup NLopt
    opt = Opt(:LD_CCSAQ, np)
    opt.lower_bounds = 0.0
    opt.upper_bounds = 1.0
    opt.ftol_rel = tol
    opt.maxeval = max_iter

    # Objective callback
    opt.max_objective = function (p, grad)
        g = objective_and_gradient!(grad, p, prob)
        ret_grad .= grad

        # Log iteration
        next_iteration!(prob)
        log_iteration!(prob, g, p)

        return g
    end

    # Constraints (if enabled)
    if use_constraints && prob.foundry_mode
        NLopt.inequality_constraint!(opt,
            (p, g) -> glc_solid(p, g; sim=prob.sim, control=prob.control), 1e-8)
        NLopt.inequality_constraint!(opt,
            (p, g) -> glc_void(p, g; sim=prob.sim, control=prob.control), 1e-8)
    end

    # Optimize
    (g_opt, p_opt, ret) = optimize(opt, prob.p)

    @info "  NLopt: $ret after $(opt.numevals) evaluations"

    return g_opt, p_opt, ret_grad
end

# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

"""Log optimization iteration."""
function log_iteration!(prob::OptimizationProblem, g, p)
    iter = prob.iteration

    # Console output every 5 iterations
    if iter % 5 == 0
        @info "Iter $iter: g = $(round(g, sigdigits=4))"
    end

    # Save checkpoint every 20 iterations
    if iter % 20 == 0
        save_checkpoint(prob, "iter_$(iter)")
    end
end

"""Save optimization checkpoint."""
function save_checkpoint(prob::OptimizationProblem, name::String)
    filepath = joinpath(prob.root, "$(name).jld2")

    mkpath(prob.root)

    data = Dict(
        "p" => prob.p,
        "g" => prob.g,
        "iteration" => prob.iteration,
        "β" => prob.control.β
    )

    # JLD2.save(filepath, data)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════════

"""Run single evaluation (no optimization)."""
function evaluate(prob::OptimizationProblem, p::Vector{Float64})
    ∇g = zeros(Float64, length(p))
    g = objective_and_gradient!(∇g, p, prob)
    return g, ∇g
end

"""Test gradient with finite differences."""
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
