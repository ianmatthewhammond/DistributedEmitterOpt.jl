"""
    PardisoSolver

Optional sparse direct solver using Intel MKL Pardiso.
Requires the user to independently install Pardiso.jl.

The actual implementation is in ext/PardisoExt.jl and only loads
when Pardiso.jl is available.
"""

import SparseArrays: SparseMatrixCSC

"""Pardiso-based solver (requires Pardiso.jl extension)."""
struct PardisoSolver <: AbstractSolver
    # Matrix type for Pardiso:
    #  1 = real structurally symmetric
    #  2 = real symmetric positive definite
    # -2 = real symmetric indefinite
    #  3 = complex structurally symmetric
    #  4 = complex Hermitian positive definite
    # -4 = complex Hermitian indefinite
    #  6 = complex symmetric
    # 11 = real unsymmetric
    # 13 = complex unsymmetric (default for Maxwell)
    mtype::Int
    PardisoSolver(; mtype::Int=13) = new(mtype)
end

# Stub implementations that error if extension not loaded
function lu!(::PardisoSolver, A::SparseMatrixCSC)
    error("Pardiso.jl not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\")")
end

function filter_lu!(::PardisoSolver, A::SparseMatrixCSC)
    error("Pardiso.jl not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\")")
end

function solve!(::PardisoSolver, factor, b::Vector)
    error("Pardiso.jl not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\")")
end

function solve_adjoint!(::PardisoSolver, factor, b::Vector)
    error("Pardiso.jl not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\")")
end
