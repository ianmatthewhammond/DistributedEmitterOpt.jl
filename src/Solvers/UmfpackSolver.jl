"""
    UmfpackSolver

Default sparse LU solver using SuiteSparse UMFPACK.
"""

import SparseArrays: SparseMatrixCSC
import LinearAlgebra

"""UMFPACK-based solver."""
struct UmfpackSolver <: AbstractSolver end

"""Factorize matrix with UMFPACK."""
function lu!(::UmfpackSolver, A::SparseMatrixCSC)
    LinearAlgebra.lu(A)
end

"""Factorize for filter (real matrix)."""
function filter_lu!(::UmfpackSolver, A::SparseMatrixCSC)
    LinearAlgebra.lu(A)
end

"""Solve Ax = b using cached factorization."""
function solve!(::UmfpackSolver, factor, b::Vector)
    factor \ b
end

"""Solve A'x = b (adjoint system)."""
function solve_adjoint!(::UmfpackSolver, factor, b::Vector)
    factor' \ b
end
