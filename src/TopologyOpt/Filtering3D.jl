"""
    Filtering3D

Helmholtz PDE-based filtering for 3D DOF mode.
Solves: ∫(R²∇pf·∇v + pf·v)dΩ = ∫(p·v)dΩ
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Helmholtz filter assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    assemble_filter_matrix(R, sim) -> SparseMatrix

Assemble Helmholtz filter matrix:
  ∫(∇v · (R² ⊙ ∇u)) dΩ_design + ∫(v·u) dΩ

Matches old code: gradient term only in design region, mass term on full domain.
"""
function assemble_filter_matrix(R::NTuple{3,Float64}, sim)
    tensor_R = TensorValue(R[1]^2, 0.0, 0.0,
                           0.0, R[2]^2, 0.0,
                           0.0, 0.0, R[3]^2)

    A_filter = assemble_matrix(sim.Pf, sim.Pf) do u, v
        ∫((∇(v) ⋅ (tensor_R ⋅ ∇(u))))sim.dΩ_design + ∫(v * u)sim.dΩ
    end

    return A_filter
end

"""
    assemble_filter_rhs(p_vec, sim) -> Vector

Assemble filter RHS: ∫(p·v)dΩ
"""
function assemble_filter_rhs(p_vec::Vector{Float64}, sim)
    # Create FEFunction from vector
    p = FEFunction(sim.P, p_vec)

    b_filter = assemble_vector(sim.Pf) do v
        ∫(v * p)sim.dΩ
    end

    return b_filter
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    filter_operator_key(R, sim) -> NamedTuple

What the Helmholtz filter matrix depends on: the three radii, the filtered FE space `sim.Pf`
and the two integration measures of `assemble_filter_matrix`, `sim.dΩ_design` (gradient term)
and `sim.dΩ` (mass term). The space and the measures are compared by object identity
(`===`), so assigning a new space or measure to `sim` is detected. Changing one of those
objects in place is not: after such an edit call `DistributedEmitterOpt.clear_filter_factor!`
on the cache (or `empty!` on its pool).
"""
filter_operator_key(R::NTuple{3,Float64}, sim) =
    (R=R, space=sim.Pf, dΩ_design=sim.dΩ_design, dΩ=sim.dΩ)

"""True if `key` (as stored in `cache.F_key`) describes the operator for `R` on `sim`."""
function filter_key_matches(key, R::NTuple{3,Float64}, sim)
    isnothing(key) && return false
    key.R == R || return false
    key.space === sim.Pf || return false
    key.dΩ_design === sim.dΩ_design || return false
    key.dΩ === sim.dΩ || return false
    return true
end

"""
    ensure_filter_factor!(cache, sim, R) -> cache

Make `cache.F_factor` the factorization of the filter matrix for radii `R` on `sim`.
A factor built for the same operator (see `filter_operator_key`) is reused as is. Otherwise
the cached factor is released and the matrix is assembled and factorized from scratch,
exactly as on a cache that never held a factor; the key is recorded only after the new
factor is installed. If the release, assembly or factorization throws, the cache is left
with no factor and no key (the previous factor is not kept), and the next call starts again.
"""
function ensure_filter_factor!(cache, sim, R::NTuple{3,Float64})
    if has_filter_factor(cache) && filter_key_matches(cache.F_key, R, sim)
        return cache
    end
    clear_filter_factor!(cache)
    A = assemble_filter_matrix(R, sim)
    filter_lu!(cache, A)
    cache.F_key = filter_operator_key(R, sim)
    return cache
end

"""
    filter_helmholtz!(p_vec, cache, sim, control) -> Vector

Filter design using Helmholtz PDE (3D DOF mode).
Caches the factorization in `cache.F_factor`, keyed by `control.R_filter`, the filtered FE
space `sim.Pf` and the measures `sim.dΩ_design` and `sim.dΩ`; it is rebuilt whenever one of
them changes. The space and measures are compared by identity: after editing one of them in
place, call `DistributedEmitterOpt.clear_filter_factor!(cache)`.

A filter cache belongs to one caller at a time: concurrent calls that share a cache can solve
with the factor another call installed, so give each concurrent caller its own `SolverCache`.
"""
function filter_helmholtz!(p_vec::Vector{Float64}, cache, sim, control)
    if !control.use_filter
        return copy(p_vec)
    end

    # Reuse the cached factor only if it was built for these radii, this space and these measures
    ensure_filter_factor!(cache, sim, control.R_filter)

    # RHS
    b = assemble_filter_rhs(p_vec, sim)

    # Solve
    pf_vec = filter_solve!(cache, b)

    return pf_vec
end

"""
    filter_helmholtz_adjoint!(∂g_∂pf, cache, sim, control) -> Vector

Adjoint of Helmholtz filter.
Since A is symmetric: A^(-T) = A^(-1), so adjoint solve = forward solve.
Uses the factor for the current `control.R_filter`, factorizing if the cached one was
built for other radii, another filtered space or other measures (as `filter_helmholtz!`).
A filter cache belongs to one caller at a time; see `filter_helmholtz!`.
"""
function filter_helmholtz_adjoint!(∂g_∂pf::Vector{Float64}, cache, sim, control)
    if !control.use_filter
        return copy(∂g_∂pf)
    end

    ensure_filter_factor!(cache, sim, control.R_filter)

    # For symmetric A, adjoint = forward
    # But need to map from Pf space back to P space

    # Solve A^T λ = ∂g_∂pf (same as A λ = ... for symmetric)
    λ = filter_solve_adjoint!(cache, ∂g_∂pf)

    # Map to P space via ∫(λ·v)dΩ
    ∂g_∂p = assemble_vector(sim.P) do v
        λ_field = FEFunction(sim.Pf, λ)
        ∫(v * λ_field)sim.dΩ
    end

    return ∂g_∂p
end
