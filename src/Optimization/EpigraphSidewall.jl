"""
    EpigraphSidewall

Worst case over sidewall samples in EPIGRAPH form, for the standalone CCSA backend.

A sharply weighted KS softmin can approach a hard minimum whose gradient jumps
when the worst-case sample changes. The epigraph form instead gives each sample
its own constraint and keeps the objective linear:

    variables  x = [p; τ],   τ = t / T
    minimize   −τ
    s.t.       cᵢ(x) = τ − gᵢ(p) / (g_norm · T) ≤ 0,     i = 1 … length(samples)
               (optionally) linewidth solid/void on p, normalized M/ε − 1 ≤ 0
    bounds     p ∈ [0, 1],   τ ∈ [0, +∞)

`T = min_i g_i(p₀)/g_norm` is fixed for a stage (and saved in the checkpoint), so τ
is O(1) and every constraint is O(1).

EVALUATION ORDER, which the cache relies on (NLopt-MMA-CCSA `ccsa_quadratic.c`
@ 7372741): at every point CCSA evaluates the objective first and then every
constraint at the SAME x (start `:383,:390-395`; every inner trial `:500,:507-513`;
a resume evaluates nothing, `:355-366`). So the expensive solves (one forward +
adjoint per sample) run in the objective callback, and the constraints read a cache.
`nevals` counts objective calls only, so eval counting, `maxeval`, the NEVAL gate and
checkpoints keep their meaning.

THE CACHE IS KEYED ON THE VALUES OF p, never on object identity. The C side reuses
one x buffer for every trial point, so the key is a COPY of `x[1:np]` compared with
exact `==`. An identity- or pointer-keyed cache here would return the previous
point's samples for every new trial point — an identity-based cache would silently reuse stale sample values.
"""

# Test hooks. They exist so the tests can build real negative controls; production
# never touches them.
#   _EPIGRAPH_CACHE_ENABLED[] = false  → every call recomputes (the uncached count)
#   _EPIGRAPH_ABORT_AFTER[] = k        → throw EpigraphTestAbort after k evals of an
#                                        epoch (a simulated job kill)
#   _EPIGRAPH_LAST_STATS[]             → per-epoch counters and trajectories
const _EPIGRAPH_CACHE_ENABLED = Ref(true)
const _EPIGRAPH_ABORT_AFTER = Ref(typemax(Int))
const _EPIGRAPH_LAST_STATS = Ref{Any}(nothing)

struct EpigraphTestAbort <: Exception
    evals::Int
end

"""
    EpigraphCache(np, nsamples)

Per-sample values and gradients at one design `p`, keyed on a copy of `p`'s values.
`solves` counts sample solves (one forward + adjoint each); `hits`/`misses` count
lookups. A new cache is created for every epoch (β changes the objective).
"""
mutable struct EpigraphCache
    np::Int
    key::Vector{Float64}
    valid::Bool
    gs::Vector{Float64}
    grads::Vector{Vector{Float64}}
    solves::Int
    hits::Int
    misses::Int
end
EpigraphCache(np::Int, ns::Int) =
    EpigraphCache(np, Float64[], false, zeros(ns), [zeros(np) for _ in 1:ns], 0, 0, 0)

function _epigraph_key_matches(c::EpigraphCache, x::AbstractVector{Float64})
    c.valid || return false
    @inbounds for i in 1:c.np
        x[i] == c.key[i] || return false
    end
    return true
end

"""
    epigraph_samples!(cache, x, prob, samples) -> (gs, grads, hit)

Per-sample values and gradients at `p = x[1:np]` (`x` may carry τ in its last
entry; it is ignored). A hit returns the cached arrays; a miss solves every sample
and re-keys the cache on a COPY of `x[1:np]`.
"""
function epigraph_samples!(cache::EpigraphCache, x::AbstractVector{Float64}, prob,
                           samples::AbstractVector{<:Tuple})
    length(x) >= cache.np ||
        throw(ArgumentError("epigraph_samples!: x has length $(length(x)) < np = $(cache.np)"))
    if _EPIGRAPH_CACHE_ENABLED[] && _epigraph_key_matches(cache, x)
        cache.hits += 1
        return cache.gs, cache.grads, true
    end
    p = Vector{Float64}(x[1:cache.np])      # a copy: the caller's buffer will be reused
    cache.valid = false                      # stays invalid if a solve throws
    foreach(g -> fill!(g, 0.0), cache.grads)
    gs = robust_sidewall_samples!(cache.grads, p, prob, samples)
    cache.gs .= gs
    cache.key = p
    cache.valid = true
    cache.solves += length(samples)
    cache.misses += 1
    return cache.gs, cache.grads, false
end

"""
    epigraph_seed!(cache, p0, prob, samples, g_norm) -> (T, τ₀)

Solve every sample at the stage's starting design, leaving the cache filled so the
first objective call is a hit (no extra solve). Returns the stage constant
`T = min_i g_i(p₀)/g_norm` and a strictly feasible start `τ₀ = (1 − 1e-6)·min_i g_i/(g_norm T)`
(`ccsa_quadratic.c:397` tests the start with `fcval ≤ 0`).
"""
function epigraph_seed!(cache::EpigraphCache, p0::Vector{Float64}, prob,
                        samples::AbstractVector{<:Tuple}, g_norm::Float64)
    gs, _, _ = epigraph_samples!(cache, p0, prob, samples)
    worst = minimum(gs) / g_norm
    (isfinite(worst) && worst > 0) ||
        throw(ArgumentError("epigraph: worst case g/g_norm at the stage start is $worst; T must be finite and > 0"))
    T = worst
    τ0 = (1 - 1e-6) * (minimum(gs) / (g_norm * T))
    return T, τ0
end

"""
    epigraph_constraint(cache, i, prob, samples, g_norm, T) -> (x, grad) -> cᵢ

`cᵢ = τ − gᵢ(p)/(g_norm·T)`, gradient `[−∇gᵢ/(g_norm·T); 1]`. Reads the cache; a miss
(which the evaluation order says cannot happen) recomputes and logs a warning.
"""
function epigraph_constraint(cache::EpigraphCache, i::Int, prob,
                             samples::AbstractVector{<:Tuple}, g_norm::Float64, T::Float64)
    np = cache.np
    scale = g_norm * T
    return function (x, grad)
        gs, grads, hit = epigraph_samples!(cache, x, prob, samples)
        hit || log_warn(:objective,
            "epigraph constraint cache MISS: recomputed every sample (the objective is evaluated first at each point, so this should not happen)";
            constraint=i)
        v = x[np+1] - gs[i] / scale
        if !isempty(grad)
            gi = grads[i]
            @inbounds for j in 1:np
                grad[j] = -gi[j] / scale
            end
            grad[np+1] = 1.0
        end
        return v
    end
end

"""
    epigraph_linewidth(f, np, cs) -> (x, grad) -> cs·f(x[1:np])

Adapt an np-variable linewidth constraint `f(p, grad_p)` (`glc_solid`/`glc_void`) to
x = [p; τ]: pass a COPY of `x[1:np]` and an np-length gradient buffer, scale value
and gradient by `cs`, and write `grad[τ] = 0` explicitly (the C buffer is reused, so
an unwritten entry would carry the previous constraint's value).
"""
function epigraph_linewidth(f, np::Int, cs::Float64)
    return function (x, grad)
        p = Vector{Float64}(x[1:np])
        gbuf = isempty(grad) ? Float64[] : zeros(Float64, np)
        v = f(p, gbuf)
        if !isempty(grad)
            @inbounds for j in 1:np
                grad[j] = cs * gbuf[j]
            end
            grad[np+1] = 0.0
        end
        return cs * v
    end
end
