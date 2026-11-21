"""
    AbstractSolver

Interface for sparse direct solvers. Concrete implementations wrap
Pardiso, UMFPACK, etc.

## Required interface
- `lu!(cache, A)` — Factorize matrix A
- `solve!(cache, b)` — Solve Ax = b (forward)
- `solve_adjoint!(cache, b)` — Solve A'x = b (adjoint)

## Solution storage
- `returns_owned_solutions(solver)` — `true` (the default) declares that creating,
  refactorizing, solving with and releasing a different logical factor, including when any
  of these throws, leave a vector returned earlier by `solve!` or `solve_adjoint!` unchanged
  (its length and values) and leave the caller's own factors usable. Reusing the same Julia
  handle does not make a new factorization the same logical factor. A forward, adjoint or
  filter solve that overwrites its own factor's buffer is compatible with `true`: an export
  copies each result before it releases that factor. A backend whose own later solve on the
  same factor, or that factor's release, rewrites a vector the caller still holds is not
  covered by the declaration and is outside the export's guarantee (issue #41). A backend
  that reuses storage across factors must define it `false`: for example one solution
  vector kept on the solver object, a filter solve that writes into storage a Maxwell solve
  returned, or a new factor that trims or reuses the idle buffer of another factor. A pool
  that recycles storage after release must declare `false` if a new logical factor can
  change a still-retained vector. A method of `returns_owned_solutions` is a capability
  query: it must have no side effects (no logging, printing, lazy initialization or filling
  of a material or capability cache) and must give the same answer for the whole of an
  operation that relies on it. An export queries it at entry.

UMFPACK, Pardiso, MUMPS and cuDSS satisfy the default as written: every solve returns a
newly allocated vector (UMFPACK's `\\`, Pardiso's and MUMPS's `similar(b)`, and cuDSS's
copy of the device result into a new host `Array`), and making or releasing a factor does
not write a vector an earlier solve returned.

`Analysis.design_and_fields` and `Analysis.visualize_results` refuse a solver checked at entry that declares
`false` (see their docstrings).
"""
abstract type AbstractSolver end

# Optional overload points for solvers that support in-place refactorization
# and explicit memory release (e.g., Pardiso).
lu!(solver::AbstractSolver, factor, A) = lu!(solver, A)
filter_lu!(solver::AbstractSolver, factor, A) = filter_lu!(solver, A)
release_factor!(::AbstractSolver, factor) = nothing
returns_owned_solutions(::AbstractSolver) = true   # see "Solution storage" above

"""
    SolverCache{S<:AbstractSolver}

Caches LU factorizations for reuse across optimization iterations.
Separate caches for:
- Maxwell system (complex, A_factor)
- Eigen shift system (complex, A_factor in eigen_cache)
- Filter PDE (real, F_factor)

## Fields
- `A_factor` — Maxwell LU factorization
- `F_factor` — Filter Helmholtz LU factorization
- `F_key` — Operator the filter factorization was built for (radii, filtered FE space and the two filter measures), or `nothing`
- `E2_factor` — E² regularization LU factorization (optional)
- `x` — Solution vector cache
- `am_head` — True if this process performs solves (MPI compatibility)
- `solver` — Solver backend
"""
mutable struct SolverCache{S<:AbstractSolver}
    A_factor::Any              # Maxwell LU
    F_factor::Any              # Filter LU
    F_key::Any                 # (R, space, dΩ_design, dΩ) that F_factor was assembled for
    E2_factor::Any             # E² filter LU (optional)
    x::Vector{ComplexF64}      # Solution vector
    am_head::Bool              # MPI head flag
    solver::S

    function SolverCache{S}(solver::S; am_head::Bool=true) where S<:AbstractSolver
        new{S}(nothing, nothing, nothing, nothing, ComplexF64[], am_head, solver)
    end
end

SolverCache(solver::S; kwargs...) where S<:AbstractSolver = SolverCache{S}(solver; kwargs...)

# ═══════════════════════════════════════════════════════════════════════════════
# Solver interface (to be implemented by concrete solvers)
# ═══════════════════════════════════════════════════════════════════════════════

"""Factorize Maxwell matrix."""
function maxwell_lu!(cache::SolverCache, A)
    if isnothing(cache.A_factor)
        cache.A_factor = lu!(cache.solver, A)
    else
        cache.A_factor = lu!(cache.solver, cache.A_factor, A)
    end
end

"""Solve Maxwell system."""
function maxwell_solve!(cache::SolverCache, b::Vector)
    cache.x = solve!(cache.solver, cache.A_factor, b)
    return cache.x
end

"""Solve adjoint Maxwell system (uses conjugate transpose)."""
function maxwell_solve_adjoint!(cache::SolverCache, b::Vector)
    solve_adjoint!(cache.solver, cache.A_factor, b)
end

"""Factorize eigen shift matrix (A - σE)."""
function eigen_lu!(cache::SolverCache, A)
    if isnothing(cache.A_factor)
        cache.A_factor = lu!(cache.solver, A)
    else
        cache.A_factor = lu!(cache.solver, cache.A_factor, A)
    end
end

"""Solve eigen shift system."""
function eigen_solve!(cache::SolverCache, b::Vector)
    cache.x = solve!(cache.solver, cache.A_factor, b)
    return cache.x
end

"""Solve adjoint eigen shift system."""
function eigen_solve_adjoint!(cache::SolverCache, b::Vector)
    solve_adjoint!(cache.solver, cache.A_factor, b)
end

"""
Factorize filter Helmholtz matrix into `cache.F_factor`. The operator key `cache.F_key` is
cleared, because this method does not know which radii, space or measures `A` came from;
`ensure_filter_factor!` sets the key after it installs a factor.
"""
function filter_lu!(cache::SolverCache, A)
    cache.F_key = nothing
    if isnothing(cache.F_factor)
        cache.F_factor = filter_lu!(cache.solver, A)
    else
        cache.F_factor = filter_lu!(cache.solver, cache.F_factor, A)
    end
end

"""Solve filter system."""
function filter_solve!(cache::SolverCache, b::Vector)
    solve!(cache.solver, cache.F_factor, b)
end

"""Solve adjoint filter system."""
function filter_solve_adjoint!(cache::SolverCache, b::Vector)
    solve_adjoint!(cache.solver, cache.F_factor, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

"""Copy cache for a new frequency (shares solver, clears factors)."""
function copy_cache(cache::SolverCache{S}) where S
    SolverCache{S}(cache.solver; am_head=cache.am_head)
end

"""Check if factorization is cached."""
has_maxwell_factor(cache::SolverCache) = !isnothing(cache.A_factor)
has_filter_factor(cache::SolverCache) = !isnothing(cache.F_factor)
has_eigen_factor(cache::SolverCache) = !isnothing(cache.A_factor)

"""
    clear_filter_factor!(cache) -> cache

Release the filter factorization and forget the operator it was built for. The factor and
its key are detached from the cache before the backend releases the factor, so a release
that throws still leaves the cache empty (the error propagates).
"""
function clear_filter_factor!(cache::SolverCache)
    F = cache.F_factor
    cache.F_factor = nothing
    cache.F_key = nothing
    isnothing(F) || release_factor!(cache.solver, F)
    return cache
end

# ═══════════════════════════════════════════════════════════════════════════════
# Cache Pool — deduplicates caches by (λ, θ, polarization) key
# ═══════════════════════════════════════════════════════════════════════════════

const CacheKey = Union{Tuple{Float64,Float64,Symbol},Tuple{Float64,Float64,Symbol,Float64,Float64,Symbol}}

"""
    SolverCachePool{S<:AbstractSolver}

Pool of solver caches, one per unique (λ, θ, polarization) configuration.
Deduplicates caches so identical configs share the same LU factorization.

Also holds a shared filter cache and a shared eigen cache for shift-invert solves.
"""
mutable struct SolverCachePool{S<:AbstractSolver}
    caches::Dict{CacheKey,SolverCache{S}}
    filter_cache::SolverCache{S}  # Shared for all filter operations
    eigen_cache::SolverCache{S}   # Shared eigen (shift-invert) cache
    solver::S
    am_head::Bool
    eigen_shift::Union{Nothing,Float64}
end

function SolverCachePool(solver::S; am_head::Bool=true) where S<:AbstractSolver
    SolverCachePool{S}(
        Dict{CacheKey,SolverCache{S}}(),
        SolverCache{S}(solver; am_head),
        SolverCache{S}(solver; am_head),
        solver,
        am_head,
        nothing
    )
end

"""Get or create cache for a field config."""
function get_cache!(pool::SolverCachePool{S}, fc::FieldConfig) where S
    key = cache_key(fc)
    get!(pool.caches, key) do
        SolverCache{S}(pool.solver; am_head=pool.am_head)
    end
end

"""Get cache by key directly."""
function get_cache!(pool::SolverCachePool{S}, key::CacheKey) where S
    get!(pool.caches, key) do
        SolverCache{S}(pool.solver; am_head=pool.am_head)
    end
end

"""Number of cached configurations."""
num_cached(pool::SolverCachePool) = length(pool.caches)

"""Clear all Maxwell factorizations (call when pt changes)."""
function clear_maxwell_factors!(pool::SolverCachePool)
    for cache in values(pool.caches)
        if !isnothing(cache.A_factor)
            release_factor!(cache.solver, cache.A_factor)
        end
        cache.A_factor = nothing
    end
end

"""Clear eigen (shift-invert) factorization."""
function clear_eigen_factors!(pool::SolverCachePool)
    if !isnothing(pool.eigen_cache.A_factor)
        release_factor!(pool.eigen_cache.solver, pool.eigen_cache.A_factor)
    end
    pool.eigen_cache.A_factor = nothing
    pool.eigen_shift = nothing
end

"""
Empty the cache pool (remove all cached solvers). Releases the Maxwell, filter and E² factors
of each per-configuration cache, the filter factor of `filter_cache`, and the shift-invert
factor and the filter factor of `eigen_cache`. Each filter factor is cleared together with its
key, since any of these caches can be passed to the Helmholtz filter.
"""
function Base.empty!(pool::SolverCachePool)
    for cache in values(pool.caches)
        if !isnothing(cache.A_factor)
            release_factor!(cache.solver, cache.A_factor)
            cache.A_factor = nothing
        end
        clear_filter_factor!(cache)
        if !isnothing(cache.E2_factor)
            release_factor!(cache.solver, cache.E2_factor)
            cache.E2_factor = nothing
        end
    end
    clear_filter_factor!(pool.filter_cache)
    if !isnothing(pool.eigen_cache.A_factor)
        release_factor!(pool.eigen_cache.solver, pool.eigen_cache.A_factor)
        pool.eigen_cache.A_factor = nothing
    end
    clear_filter_factor!(pool.eigen_cache)
    pool.eigen_shift = nothing
    empty!(pool.caches)
end
