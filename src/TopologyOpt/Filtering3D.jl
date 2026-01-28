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
    filter_helmholtz!(p_vec, cache, sim, control) -> Vector

Filter design using Helmholtz PDE (3D DOF mode).
Caches factorization in cache.F_factor.
"""
function filter_helmholtz!(p_vec::Vector{Float64}, cache, sim, control)
    if !control.use_filter
        return copy(p_vec)
    end

    R = control.R_filter

    # Assemble or use cached
    if !has_filter_factor(cache)
        A = assemble_filter_matrix(R, sim)
        filter_lu!(cache, A)
    end

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
"""
function filter_helmholtz_adjoint!(∂g_∂pf::Vector{Float64}, cache, sim, control)
    if !control.use_filter
        return copy(∂g_∂pf)
    end

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
