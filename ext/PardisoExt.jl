"""
    PardisoExt

Intel MKL Pardiso through Pardiso.jl, behind `DistributedEmitterOpt.PardisoSolver`. The
extension loads with Pardiso.jl alone and runs where MKL does (x86-64 Linux, for example);
where MKL is not available, such as on Apple Silicon, use `UmfpackSolver()`.

Pardiso reads Julia's CSC arrays as CSR, so it sees Aᵀ. A matrix stored in full (Maxwell
mtype 13 and filter mtype 11) is solved on one factorization as

    A x = b   →  transposed solve of what Pardiso sees                 (iparm(12) = 2)
    Aᴴ x = b  ⇔  Aᵀ conj(x) = conj(b)  →  plain solve (iparm(12) = 0) on conj(b), conjugated

so oblique Bloch operators, which are not complex symmetric, get true adjoints. Symmetric
types keep one triangle and are used only for a matrix checked to be symmetric: Maxwell
mtype 6, which `checked_maxwell_mtype` picks by default for each factorization whose
matrix is complex symmetric (normal incidence), and filter mtypes -2 and 2. For them the
same plain solve on conj(b) is the exact adjoint.

The analysis is reused while the sparsity pattern matches a copy taken when it ran. A factor
is released once, by `release_factor!` or when its factorization fails. A dropped factor is
released by the finalizer that `Pardiso.pardisoinit` registers (Pardiso.jl 1.1 and later);
MKL empties a handle when it releases it, so that finalizer's repeat release of a released
factor does nothing. A lock per factor guards its handle, whose iparm(12) changes between
solves.
"""
module PardisoExt

import DistributedEmitterOpt
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import Pardiso
using SparseArrays: SparseMatrixCSC, tril

# ---------------------------------------------------------------------------
# Factor wrapper
# ---------------------------------------------------------------------------

mutable struct PardisoFactor{T}
    ps::Pardiso.MKLPardisoSolver
    A::SparseMatrixCSC{T,Int}   # as handed to Pardiso: the lower triangle for symmetric types
    mtype::Int
    n::Int                      # size and pattern of the last analysis (copies)
    colptr::Vector{Int}
    rowval::Vector{Int}
    released::Bool
    lock::ReentrantLock
end

one_triangle(mtype::Int) = mtype in (2, -2, 6)
storage(A, mtype::Int) = one_triangle(mtype) ? tril(A) : A
same_pattern(f::PardisoFactor, A) =
    size(A) == (f.n, f.n) && A.colptr == f.colptr && A.rowval == f.rowval

# iparm(12) for A x = b, kept between calls. A symmetric type solves what it stores.
forward_mode(mtype::Int) = one_triangle(mtype) ? 0 : 2

function maxwell_mtype(solver::DistributedEmitterOpt.PardisoSolver, A)
    mtype = DistributedEmitterOpt.checked_maxwell_mtype(solver, A)
    mtype in (13, 6) ||
        throw(ArgumentError("PardisoSolver: Maxwell mtype=$mtype is not supported; use 13 and 6 (the defaults of mtype and mtype_symmetric)"))
    return mtype
end

function release_pardiso!(f::PardisoFactor)
    f.released && return nothing
    f.released = true
    Pardiso.set_phase!(f.ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(f.ps)
    return nothing
end

# Analysis and/or numerical factorization. A failure releases the handle before rethrowing.
function factorize!(f::PardisoFactor{T}, phase::Pardiso.Phase) where T
    try
        Pardiso.set_phase!(f.ps, phase)
        Pardiso.pardiso(f.ps, f.A, zeros(T, f.n))
    catch
        try
            release_pardiso!(f)
        catch
        end
        rethrow()
    end
    return f
end

function new_factor(solver::DistributedEmitterOpt.PardisoSolver, Awork::SparseMatrixCSC{T,Int},
                    mtype::Int; use_iparm::Bool) where T
    Pardiso.mkl_is_available() ||
        error("PardisoSolver: MKL is not available here (MKL Pardiso needs MKL, for example on x86-64 Linux); use UmfpackSolver()")
    ps = Pardiso.MKLPardisoSolver()
    solver.nprocs > 0 && Pardiso.set_nprocs!(ps, solver.nprocs)
    Pardiso.set_matrixtype!(ps, mtype)
    Pardiso.pardisoinit(ps)
    if use_iparm && !isnothing(solver.iparm)
        for (i, v) in enumerate(DistributedEmitterOpt.checked_pardiso_iparm(solver.iparm))
            Pardiso.set_iparm!(ps, i, v)
        end
    end
    Pardiso.set_iparm!(ps, 12, forward_mode(mtype))
    solver.msglvl > 0 && Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    f = PardisoFactor{T}(ps, Awork, mtype, size(Awork, 1), copy(Awork.colptr), copy(Awork.rowval),
        false, ReentrantLock())
    return factorize!(f, Pardiso.ANALYSIS_NUM_FACT)
end

# New values on the analysed pattern need a numerical factorization only. Otherwise the old
# handle is released and a new factor is built.
function refactor(solver::DistributedEmitterOpt.PardisoSolver, f::PardisoFactor{T},
                  A::SparseMatrixCSC{T,Int}, mtype::Int; use_iparm::Bool) where T
    Awork = storage(A, mtype)
    reused = lock(f.lock) do
        if solver.reuse_symbolic && !f.released && f.mtype == mtype && same_pattern(f, Awork)
            f.A = Awork
            factorize!(f, Pardiso.NUM_FACT)
        else
            release_pardiso!(f)
            nothing
        end
    end
    return reused === nothing ? new_factor(solver, Awork, mtype; use_iparm) : reused
end

# Solve with iparm(12) = mode, then restore the forward setting.
function pardiso_solve(f::PardisoFactor{T}, b::Vector{T}, mode::Int) where T
    lock(f.lock) do
        f.released && throw(ArgumentError("PardisoSolver: this factorization has been released"))
        x = similar(b)
        Pardiso.set_iparm!(f.ps, 12, mode)
        try
            Pardiso.set_phase!(f.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
            Pardiso.pardiso(f.ps, x, f.A, b)
        finally
            Pardiso.set_iparm!(f.ps, 12, forward_mode(f.mtype))
        end
        return x
    end
end

# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

# As before, the iparm profile applies to unsymmetric (full-storage) Maxwell factorizations;
# a symmetric type keeps MKL's defaults.
function lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{ComplexF64,Int})
    mtype = maxwell_mtype(solver, A)
    return new_factor(solver, storage(A, mtype), mtype; use_iparm=!one_triangle(mtype))
end

function lu!(
    solver::DistributedEmitterOpt.PardisoSolver,
    factor::PardisoFactor{ComplexF64},
    A::SparseMatrixCSC{ComplexF64,Int},
)
    mtype = maxwell_mtype(solver, A)
    return refactor(solver, factor, A, mtype; use_iparm=!one_triangle(mtype))
end

function filter_lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{Float64,Int})
    mtype = DistributedEmitterOpt.checked_filter_mtype(solver, A)
    return new_factor(solver, storage(A, mtype), mtype; use_iparm=false)
end

function filter_lu!(
    solver::DistributedEmitterOpt.PardisoSolver,
    factor::PardisoFactor{Float64},
    A::SparseMatrixCSC{Float64,Int},
)
    return refactor(solver, factor, A, DistributedEmitterOpt.checked_filter_mtype(solver, A); use_iparm=false)
end

function solve!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    return pardiso_solve(factor, b, forward_mode(factor.mtype))
end

# Aᴴ x = b ⇔ Aᵀ conj(x) = conj(b), the plain solve of the Aᵀ that Pardiso sees.
function solve_adjoint!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    T <: Complex || return pardiso_solve(factor, b, 0)
    return conj!(pardiso_solve(factor, conj(b), 0))
end

function release_factor!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor)
    lock(() -> release_pardiso!(factor), factor.lock)
end

end # module
