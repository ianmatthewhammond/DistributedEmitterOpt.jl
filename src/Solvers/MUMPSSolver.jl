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
