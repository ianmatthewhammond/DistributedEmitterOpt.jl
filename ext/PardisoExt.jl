"""
    PardisoExt

Package extension that provides the actual Pardiso solver implementation.
This file is only loaded when the user has Pardiso.jl installed.
"""
module PardisoExt

using DistributedEmitterOpt: AbstractSolver, PardisoSolver
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!
using Pardiso
using SparseArrays: SparseMatrixCSC

# ═══════════════════════════════════════════════════════════════════════════════
# Pardiso Factorization Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PardisoFactor

Wrapper holding Pardiso solver state and factorization.
"""
mutable struct PardisoFactor{T}
    ps::MKLPardisoSolver
    A::SparseMatrixCSC{T}
    mtype::Int
end

function PardisoFactor(A::SparseMatrixCSC{T}, mtype::Int) where T
    ps = MKLPardisoSolver()

    # Set matrix type
    set_matrixtype!(ps, mtype)

    # Initialize solver
    pardisoinit(ps)

    # Number of processors (use all available)
    set_nprocs!(ps, Threads.nthreads())

    # Suppress output (set to 1 for debug)
    set_msglvl!(ps, 0)

    # Analysis + factorization (phases 11 + 22 = 12)
    set_phase!(ps, 12)

    # Dummy RHS for analysis/factor phase
    n = size(A, 1)
    b = zeros(T, n)
    x = zeros(T, n)

    pardiso(ps, x, A, b)

    PardisoFactor{T}(ps, A, mtype)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interface Implementation
# ═══════════════════════════════════════════════════════════════════════════════

"""Factorize complex Maxwell matrix with Pardiso."""
function lu!(solver::PardisoSolver, A::SparseMatrixCSC{ComplexF64})
    PardisoFactor(A, solver.mtype)
end

"""Factorize real filter matrix with Pardiso."""
function filter_lu!(solver::PardisoSolver, A::SparseMatrixCSC{Float64})
    # For real symmetric positive definite filter matrix, use mtype=2
    # For real symmetric indefinite, use mtype=-2
    # Default to symmetric indefinite for safety
    PardisoFactor(A, -2)
end

"""Solve Ax = b using Pardiso factorization."""
function solve!(::PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    # Solve phase (33)
    set_phase!(factor.ps, 33)

    x = zeros(T, length(b))
    pardiso(factor.ps, x, factor.A, b)
    return x
end

"""Solve A'x = b (adjoint/transpose system)."""
function solve_adjoint!(::PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    # Pardiso: iparm[12] controls transpose solve
    #  0 = normal (A x = b)
    #  1 = transpose (A^T x = b)
    #  2 = conjugate transpose (A^H x = b) - for complex

    # Save old value
    old_iparm12 = get_iparm(factor.ps, 12)

    # Set transpose mode
    if T <: Complex
        set_iparm!(factor.ps, 12, 2)  # Conjugate transpose
    else
        set_iparm!(factor.ps, 12, 1)  # Transpose
    end

    # Solve phase
    set_phase!(factor.ps, 33)

    x = zeros(T, length(b))
    pardiso(factor.ps, x, factor.A, b)

    # Restore
    set_iparm!(factor.ps, 12, old_iparm12)

    return x
end

end # module
