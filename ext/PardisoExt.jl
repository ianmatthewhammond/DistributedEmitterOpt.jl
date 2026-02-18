"""
    PardisoExt

Package extension that provides the actual Pardiso solver implementation.
This file is only loaded when the user has Pardiso.jl installed.
"""
module PardisoExt

import DistributedEmitterOpt
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import Pardiso: MKLPardisoSolver, set_matrixtype!, pardisoinit, set_nprocs!, set_msglvl!,
                set_iparm!, get_iparm, set_phase!, pardiso
using SparseArrays: SparseMatrixCSC, spzeros, tril

# ═══════════════════════════════════════════════════════════════════════════════
# Pardiso factor wrapper
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct PardisoFactor{T}
    ps::MKLPardisoSolver
    A::SparseMatrixCSC{T,Int}
    mtype::Int
    pattern_hash::UInt
    released::Bool
end

is_symmetric_mtype(mtype::Int) = mtype in (1, 2, -2, 3, 4, -4, 6)

function matrix_for_mtype(A::SparseMatrixCSC{T,Int}, mtype::Int) where T
    is_symmetric_mtype(mtype) ? tril(A) : A
end

function sparsity_hash(A::SparseMatrixCSC{T,Int}, mtype::Int) where T
    h = hash(mtype, hash(size(A)))
    h = hash(A.colptr, h)
    hash(A.rowval, h)
end

function apply_iparm!(ps::MKLPardisoSolver, iparm::Union{Nothing,Vector{Int}})
    isnothing(iparm) && return
    n = min(length(iparm), 64)
    for i in 1:n
        set_iparm!(ps, i, iparm[i])
    end
end

function run_phase!(ps::MKLPardisoSolver, phase::Int, A::SparseMatrixCSC{T,Int}) where T
    set_phase!(ps, phase)
    if phase == -1
        x = Vector{T}()
        b = Vector{T}()
        pardiso(ps, x, A, b)
        return nothing
    end
    n = size(A, 1)
    x = zeros(T, n)
    b = zeros(T, n)
    pardiso(ps, x, A, b)
    return nothing
end

function release_pardiso!(factor::PardisoFactor)
    factor.released && return nothing
    try
        run_phase!(factor.ps, -1, factor.A)
    catch
        # Best-effort release; keep GC fallback behavior if Pardiso release errors.
    end
    factor.A = spzeros(eltype(factor.A), 0, 0)
    factor.released = true
    return nothing
end

function new_pardiso_factor(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{T,Int}, mtype::Int) where T
    Awork = matrix_for_mtype(A, mtype)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, mtype)
    pardisoinit(ps)
    set_nprocs!(ps, solver.nprocs)
    set_msglvl!(ps, solver.msglvl)
    apply_iparm!(ps, solver.iparm)

    run_phase!(ps, 12, Awork) # analysis + numerical factorization

    factor = PardisoFactor{T}(ps, Awork, mtype, sparsity_hash(Awork, mtype), false)
    finalizer(release_pardiso!, factor)
    return factor
end

function refactor_pardiso!(solver::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T},
    A::SparseMatrixCSC{T,Int}, mtype::Int) where T
    Awork = matrix_for_mtype(A, mtype)
    new_hash = sparsity_hash(Awork, mtype)

    can_reuse_symbolic = solver.reuse_symbolic &&
                         !factor.released &&
                         factor.mtype == mtype &&
                         factor.pattern_hash == new_hash

    if can_reuse_symbolic
        set_matrixtype!(factor.ps, mtype)
        apply_iparm!(factor.ps, solver.iparm)
        run_phase!(factor.ps, 22, Awork) # numerical only
        factor.A = Awork
        factor.pattern_hash = new_hash
        factor.released = false
        return factor
    end

    release_pardiso!(factor)
    return new_pardiso_factor(solver, A, mtype)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interface implementation
# ═══════════════════════════════════════════════════════════════════════════════

function lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{ComplexF64,Int})
    mtype = solver.assume_symmetric_maxwell ? solver.mtype_symmetric : solver.mtype
    return new_pardiso_factor(solver, A, mtype)
end

function lu!(solver::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{ComplexF64}, A::SparseMatrixCSC{ComplexF64,Int})
    mtype = solver.assume_symmetric_maxwell ? solver.mtype_symmetric : solver.mtype
    return refactor_pardiso!(solver, factor, A, mtype)
end

function filter_lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{Float64,Int})
    return new_pardiso_factor(solver, A, solver.filter_mtype)
end

function filter_lu!(solver::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{Float64}, A::SparseMatrixCSC{Float64,Int})
    return refactor_pardiso!(solver, factor, A, solver.filter_mtype)
end

function solve!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    set_phase!(factor.ps, 33) # solve
    x = zeros(T, length(b))
    pardiso(factor.ps, x, factor.A, b)
    return x
end

function solve_adjoint!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    old_iparm12 = get_iparm(factor.ps, 12)
    try
        set_iparm!(factor.ps, 12, T <: Complex ? 2 : 1) # A^H for complex, A^T for real
        set_phase!(factor.ps, 33)
        x = zeros(T, length(b))
        pardiso(factor.ps, x, factor.A, b)
        return x
    finally
        set_iparm!(factor.ps, 12, old_iparm12)
    end
end

function release_factor!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor)
    release_pardiso!(factor)
end

end # module
