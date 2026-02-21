"""
    PardisoSolver

Optional sparse direct solver using Intel MKL Pardiso.
Requires the user to independently install Pardiso.jl.

The actual implementation is in ext/PardisoExt.jl and only loads
when both Pardiso.jl and GridapPardiso.jl are available.
"""

import SparseArrays: SparseMatrixCSC

"""
    legacy_pardiso_iparm() -> Vector{Int}

Legacy iparm profile carried over from the old Emitter3DTopOpt solver.
Users can override by passing `iparm=...` to `PardisoSolver`.
"""
function legacy_pardiso_iparm()
    [
        1, 3, 0, 0, 0, 0, 0, 0, 0, 16, 1, 0, 1, 0, 0, 0,
        0, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
end

"""Pardiso-based solver (requires Pardiso.jl extension)."""
Base.@kwdef struct PardisoSolver <: AbstractSolver
    # Matrix type for Pardiso:
    #  1 = real structurally symmetric
    #  2 = real symmetric positive definite
    # -2 = real symmetric indefinite
    #  3 = complex structurally symmetric
    #  4 = complex Hermitian positive definite
    # -4 = complex Hermitian indefinite
    #  6 = complex symmetric
    # 11 = real unsymmetric
    # 13 = complex unsymmetric
    #
    # `mtype`               : Maxwell mtype when `assume_symmetric_maxwell=false`
    # `mtype_symmetric`     : Maxwell mtype when `assume_symmetric_maxwell=true`
    # `filter_mtype`        : Filter/E2 real-system mtype
    # `assume_symmetric_maxwell` : Use symmetric storage/mtype for Maxwell solves
    # `reuse_symbolic`      : Reuse symbolic setup when sparsity pattern is unchanged
    mtype::Int = 13
    mtype_symmetric::Int = 6
    filter_mtype::Int = -2
    assume_symmetric_maxwell::Bool = false
    reuse_symbolic::Bool = true
    msglvl::Int = 0
    nprocs::Int = Threads.nthreads()
    iparm::Union{Nothing,Vector{Int}} = legacy_pardiso_iparm()
end

# Stub implementations that error if extension not loaded
function lu!(::PardisoSolver, A::SparseMatrixCSC)
    error("Pardiso extension not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\"); Pkg.add(\"GridapPardiso\")")
end

function lu!(solver::PardisoSolver, factor, A::SparseMatrixCSC)
    lu!(solver, A)
end

function filter_lu!(::PardisoSolver, A::SparseMatrixCSC)
    error("Pardiso extension not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\"); Pkg.add(\"GridapPardiso\")")
end

function filter_lu!(solver::PardisoSolver, factor, A::SparseMatrixCSC)
    filter_lu!(solver, A)
end

function solve!(::PardisoSolver, factor, b::Vector)
    error("Pardiso extension not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\"); Pkg.add(\"GridapPardiso\")")
end

function solve_adjoint!(::PardisoSolver, factor, b::Vector)
    error("Pardiso extension not loaded. Install with: using Pkg; Pkg.add(\"Pardiso\"); Pkg.add(\"GridapPardiso\")")
end

release_factor!(::PardisoSolver, factor) = nothing
