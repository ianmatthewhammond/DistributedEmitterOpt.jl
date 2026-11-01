"""
    RobustSidewall

Worst-case design over a range of sidewall slopes.

Maximize the worst-case objective over a caller-supplied set of linear wall profiles:

    g_robust(p) = min_{(s, z₀) ∈ S} g(p; s, z₀)

This is the same idea as the erosion/dilation robust formulation DEO already
carries, one step further: erosion/dilation models a z-*uniform* linewidth bias,
a sidewall models a z-*graded* one, and the unknown-angle case is exactly the
case robust design exists for.

`min` is not differentiable where two samples tie, so the aggregate is a
Kreisselmeier–Steinhauser softmin

    g_KS = m − (1/μ̃) log Σᵢ exp(−μ̃ (gᵢ − m)),      m = min gᵢ

which is smooth for finite μ̃, is a lower bound on the true min (so the reported
worst case is never optimistic), and → min as μ̃ → ∞. Its gradient is a convex
combination of the sample gradients, ∇g_KS = Σᵢ wᵢ ∇gᵢ with
wᵢ = exp(−μ̃(gᵢ−m)) / Σⱼ exp(−μ̃(gⱼ−m)).

`sharpness` is divided by an explicit, caller-supplied `scale` to make it
dimensionless: μ̃ = sharpness / scale. **`scale` must not be computed from `gs`.**
An earlier version set μ̃ = sharpness / |min(gs)|, which is wrong twice over:

  1. It breaks the gradient. With μ̃ held fixed, ∂g_KS/∂gᵢ = wᵢ exactly and the
     `m` offset cancels. If μ̃ depends on min(gs) then perturbing the argmin also
     perturbs μ̃, adding an unaccounted term. Measured against central FD: the
     argmin's weight came out 0.444 analytic against 0.204 numerical — a factor
     of 2.2, not a roundoff.
  2. It makes the objective non-stationary. A sharpness that drifts with the
     objective value means the optimizer is maximizing a function whose
     definition changes as the design improves.

In production pass `scale = g_norm` (the flat-substrate baseline DEO already
computes), which is fixed for the whole run.

`sharpness = Inf` selects the exact min (the argmin's gradient), which is what
Sigmund's robust formulation does; it is correct but non-smooth at ties, so
finite-difference checks on it are only valid where the argmin does not switch
across the step.

COST: one full forward + adjoint solve per sample. `length(samples)` × a normal
evaluation. Keep the sample set small — 3 to 5 is the usual choice.
"""

"""
    sidewall_samples(; angles_deg, z0s) -> Vector{Tuple{Float64,Float64}}

Build the sample set from wall angles (degrees from horizontal) and reference
heights. `90°` gives slope 0 — the vertical extrusion — so the nominal design is
always in the range for free. Include both `z₀` conventions when it is unknown
whether the mask defines the top or the bottom of the feature (they differ by
`s·hd` per wall over layer height `hd`).
"""
function sidewall_samples(; angles_deg::AbstractVector{<:Real}=[80.0, 85.0, 90.0],
                          z0s::AbstractVector{<:Real}=[0.0])
    out = Tuple{Float64,Float64}[]
    for θ in angles_deg, z0 in z0s
        s = θ == 90 ? 0.0 : 1 / tan(deg2rad(θ))
        push!(out, (Float64(s), Float64(z0)))
        # slope 0 is the same design regardless of z₀; do not solve it twice
        s == 0.0 && break
    end
    return unique(out)
end

"""
    ks_softmin(gs, sharpness; scale=1.0) -> (g_agg, weights)

KS softmin and the weights its gradient is a convex combination with:
`∂g_KS/∂gᵢ = wᵢ`, exactly, because μ̃ = sharpness/`scale` does not depend on `gs`.

`m = minimum(gs)` appears only as a numerical offset and cancels in exact
arithmetic — it is not part of the mathematical definition and must not be used
to set μ̃ (see the module docstring).
"""
function ks_softmin(gs::AbstractVector{Float64}, sharpness::Float64;
                    scale::Float64=1.0)
    if !isfinite(sharpness)
        w = zeros(length(gs))
        w[argmin(gs)] = 1.0
        return minimum(gs), w
    end
    (isfinite(scale) && scale > 0) ||
        throw(ArgumentError("ks_softmin: scale must be finite and > 0, got $scale"))
    μ̃ = sharpness / scale
    m = minimum(gs)                      # offset for conditioning only
    e = exp.(-μ̃ .* (gs .- m))
    S = sum(e)
    return m - log(S) / μ̃, e ./ S
end

"""
    robust_sidewall_samples!(grads, p, prob, samples; verbose) -> gs

Evaluate the objective at every sample in `samples` — one full forward + adjoint
solve each — and return the per-sample values `gs`. When `grads` is a vector of
`length(samples)` gradient buffers, sample `i`'s gradient is written into
`grads[i]`; pass `nothing` for values only.

This is the building block both worst-case aggregations share: the KS softmin
below and the epigraph form (`EpigraphSidewall.jl`), which needs the samples
separately because each one is its own constraint.

Mutates `prob.control`'s sidewall fields per sample and restores them on exit
(including on error).

Downstream caches must refresh when the projected material field changes between samples.
"""
function robust_sidewall_samples!(grads::Union{Nothing,AbstractVector{Vector{Float64}}},
                                  p::Vector{Float64}, prob,
                                  samples::AbstractVector{<:Tuple};
                                  verbose::Bool=false)
    isempty(samples) && error("robust_sidewall_samples!: empty sample set")
    n = length(samples)
    grads === nothing || length(grads) == n ||
        throw(ArgumentError("robust_sidewall_samples!: $(length(grads)) gradient buffers for $n samples"))
    control = prob.control
    saved = (control.use_sidewall, control.sidewall_slope, control.sidewall_z0)
    gs = zeros(Float64, n)

    try
        for (i, (s, z0)) in enumerate(samples)
            control.use_sidewall = true
            control.sidewall_slope = s
            control.sidewall_z0 = z0
            gs[i] = objective_and_gradient!(grads === nothing ? Float64[] : grads[i], p, prob)
            verbose && println("    sidewall sample $i: slope=$s z0=$z0  g=$(gs[i])")
        end
    finally
        control.use_sidewall, control.sidewall_slope, control.sidewall_z0 = saved
    end
    return gs
end

"""
    robust_sidewall_objective!(∇g, p, prob, samples; sharpness, scale, verbose)
        -> (g_robust, gs, weights)

Evaluate the worst-case objective over `samples` and, when `∇g` is non-empty,
its gradient: the KS softmin of `robust_sidewall_samples!`. Mutates
`prob.control`'s sidewall fields per sample and restores them on exit
(including on error).
"""
function robust_sidewall_objective!(∇g::Vector{Float64}, p::Vector{Float64},
                                    prob, samples::AbstractVector{<:Tuple};
                                    sharpness::Float64=50.0, scale::Float64=1.0,
                                    verbose::Bool=false)
    isempty(samples) && error("robust_sidewall_objective!: empty sample set")
    want_grad = !isempty(∇g)

    n = length(samples)
    grads = want_grad ? [zeros(Float64, length(p)) for _ in 1:n] : nothing
    gs = robust_sidewall_samples!(grads, p, prob, samples; verbose)

    g_agg, w = ks_softmin(gs, sharpness; scale)

    if want_grad
        fill!(∇g, 0.0)
        for i in 1:n
            w[i] == 0.0 && continue
            @. ∇g += w[i] * grads[i]
        end
    end

    return g_agg, gs, w
end
