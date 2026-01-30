"""
    MUMPSExt

Package extension that provides the actual MUMPS solver implementation.
This file is only loaded when the user has MUMPS.jl installed.
"""
module MUMPSExt

using DistributedEmitterOpt: AbstractSolver, MUMPSSolver
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!
using MUMPS
using SparseArrays: SparseMatrixCSC

# ═══════════════════════════════════════════════════════════════════════════════
# MUMPS Factorization Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MUMPSFactor

Wrapper holding MUMPS factorization for reuse.
"""
mutable struct MUMPSFactor{T}
    mumps::Mumps{T}
    analyzed::Bool
    factored::Bool
end

function MUMPSFactor(A::SparseMatrixCSC{T}; symmetric::Bool=false) where T
    # MUMPS job types:
    #  icntl[4] = 0 (no output), 1 (errors only), 2 (warnings+errors), 3 (full)
    # sym = 0 (unsymmetric), 1 (symmetric positive def), 2 (general symmetric)
    sym = symmetric ? 2 : 0
    mumps = Mumps{T}(sym, default_icntl, default_cntl)

    # Associate matrix (analysis phase)
    associate_matrix!(mumps, A)
    factorize!(mumps)

    MUMPSFactor{T}(mumps, true, true)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interface Implementation
# ═══════════════════════════════════════════════════════════════════════════════

"""Factorize complex Maxwell matrix with MUMPS."""
function lu!(solver::MUMPSSolver, A::SparseMatrixCSC{ComplexF64})
    MUMPSFactor(A; symmetric=solver.symmetric)
end

"""Factorize real filter matrix with MUMPS."""
function filter_lu!(solver::MUMPSSolver, A::SparseMatrixCSC{Float64})
    MUMPSFactor(A; symmetric=true)  # Filter matrices are symmetric
end

"""Solve Ax = b using MUMPS factorization."""
function solve!(::MUMPSSolver, factor::MUMPSFactor{T}, b::Vector{T}) where T
    x = copy(b)
    associate_rhs!(factor.mumps, x)
    solve!(factor.mumps)
    get_solution!(factor.mumps, x)
    return x
end

"""Solve A'x = b (adjoint/transpose system)."""
function solve_adjoint!(::MUMPSSolver, factor::MUMPSFactor{T}, b::Vector{T}) where T
    # MUMPS: icntl[9] = 1 for A^T solve, 2 for A^H solve
    # For complex matrices, we want conjugate transpose (Hermitian adjoint)
    old_icntl9 = factor.mumps.icntl[9]
    factor.mumps.icntl[9] = T <: Complex ? 2 : 1

    x = copy(b)
    associate_rhs!(factor.mumps, x)
    solve!(factor.mumps)
    get_solution!(factor.mumps, x)

    factor.mumps.icntl[9] = old_icntl9
    return x
end

end # module
