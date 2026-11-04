"""
    MUMPSExt

Package extension that provides the actual MUMPS solver implementation.
This file is only loaded when the user has MUMPS.jl installed. MUMPS.jl requires
`MPI.Init()` before a solver instance is created.
"""
module MUMPSExt

import DistributedEmitterOpt
using DistributedEmitterOpt: AbstractSolver, MUMPSSolver
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!
import MUMPS
using MUMPS: Mumps, associate_matrix!, associate_rhs!, factorize!
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
    # sym = 0 (unsymmetric) or 2 (general symmetric, refused for a non-symmetric A)
    sym = DistributedEmitterOpt.checked_mumps_sym(symmetric, A)
    mumps = Mumps{T}(sym)

    # Associate matrix, then analysis + factorization
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
    MUMPS.solve!(factor.mumps)
    MUMPS.get_sol!(x, factor.mumps)
    return x
end

"""
Solve Aᴴ x = b (adjoint system). MUMPS's transposed solve (ICNTL(9) ≠ 1) uses the
plain transpose Aᵀ and never conjugates, so the complex case solves
Aᵀ conj(x) = conj(b); the real case is the transposed solve itself.
"""
function solve_adjoint!(::MUMPSSolver, factor::MUMPSFactor{T}, b::Vector{T}) where T
    x = T <: Complex ? conj.(b) : copy(b)
    associate_rhs!(factor.mumps, x)
    MUMPS.solve!(factor.mumps; transposed=true)
    MUMPS.get_sol!(x, factor.mumps)
    return T <: Complex ? conj!(x) : x
end

end # module
