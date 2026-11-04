"""
    MUMPSSolver

Optional parallel sparse direct solver using MUMPS.
Requires the user to independently install MUMPS.jl.

The actual implementation is in ext/MUMPSExt.jl and only loads
when MUMPS.jl is available.
"""

import SparseArrays: SparseMatrixCSC

"""MUMPS-based solver (requires MUMPS.jl extension)."""
struct MUMPSSolver <: AbstractSolver
    symmetric::Bool
    MUMPSSolver(; symmetric::Bool=false) = new(symmetric)
end

"""
    checked_mumps_sym(symmetric, A) -> Int

MUMPS storage code for `A`: 0 (unsymmetric) or 2 (general symmetric). Symmetric
storage keeps one triangle, so it is refused for a matrix that is not symmetric,
such as an oblique Bloch operator (A(κ)ᵀ = A(−κ)).
"""
function checked_mumps_sym(symmetric::Bool, A)
    symmetric || return 0
    norm(A - transpose(A)) <= 1e-12 * norm(A) ||
        throw(ArgumentError("MUMPSSolver: symmetric storage (sym=2) needs a symmetric matrix; use MUMPSSolver(symmetric=false) for oblique Bloch fields"))
    return 2
end

# Stub implementations that error if extension not loaded
function lu!(::MUMPSSolver, A::SparseMatrixCSC)
    error("MUMPS.jl not loaded. Install with: using Pkg; Pkg.add(\"MUMPS\")")
end

function filter_lu!(::MUMPSSolver, A::SparseMatrixCSC)
    error("MUMPS.jl not loaded. Install with: using Pkg; Pkg.add(\"MUMPS\")")
end

function solve!(::MUMPSSolver, factor, b::Vector)
    error("MUMPS.jl not loaded. Install with: using Pkg; Pkg.add(\"MUMPS\")")
end

function solve_adjoint!(::MUMPSSolver, factor, b::Vector)
    error("MUMPS.jl not loaded. Install with: using Pkg; Pkg.add(\"MUMPS\")")
end
