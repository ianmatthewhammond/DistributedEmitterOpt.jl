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
same plain solve on conj(b) is the exact adjoint of the stored symmetric operator (the
matrix itself to within the symmetry tolerance it was accepted under).

The analysis is reused while the sparsity pattern matches a copy taken when it ran. A factor
is released once, by `release_factor!` or when its factorization fails. A dropped factor is
released by the finalizer that `Pardiso.pardisoinit` registers (Pardiso.jl 1.1 and later);
MKL empties a handle when it releases it, so that finalizer's repeat release of a released
factor does nothing. A lock per factor guards its handle, whose iparm(12) changes between
solves.

Iterative refinement (iparm(8)) belongs to the factorization: `refinement_steps` of the
solver that factorized last is in force for every solve on that factor, and `nothing` puts
back the value the handle had when it was created (from the iparm profile for a full-storage
Maxwell factorization, MKL's default otherwise). The handle is not touched when the value is
already the wanted one, so a solver with `refinement_steps = nothing` sets nothing that was
not set before the keyword existed.

Every solve is checked unless `residual_rtol = 0`: ‖b − Op x‖ ≤ `residual_rtol` ‖b‖, where Op
is the operator that solve inverts, built from the matrix the factorization holds (A or Aᵀ for
a matrix stored in full; for a one-triangle type the symmetric matrix that triangle stands
for). A matrix stored in full is the caller's own object, not a copy, so its values must not be
changed in place between a factorization and the solves on it. The check reads x and b and
writes neither, nor the handle. An answer above the limit is
solved once more on the same factorization with refinement on (`resolve_steps`), iparm(8) is
put back, and an error is thrown if the second answer is above the limit too. The tolerance is
that of the solver that solves. A factor counts its checked solves and its second solves
(`residual_checks`, `residual_resolves`) and keeps the residual of the last checked answer
(`last_residual`).
"""
module PardisoExt

import DistributedEmitterOpt
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import Pardiso
using LinearAlgebra: mul!, norm
using SparseArrays: SparseMatrixCSC, tril, findnz, nonzeros, nzrange, rowvals, sparse

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
    refinement_default::Int     # iparm(8) of the handle as created, before any `refinement_steps`
    residual_checks::Int        # solves whose answer was checked against `residual_rtol`
    residual_resolves::Int      # of those, answers above the limit that were solved a second time
    last_residual::Float64      # ‖b − Op x‖/‖b‖ of the last checked answer as returned; NaN before one
end

one_triangle(mtype::Int) = mtype in (2, -2, 6)
storage(A, mtype::Int) = one_triangle(mtype) ? with_diagonal(tril(A)) : A

# MKL Pardiso's symmetric types need every diagonal entry stored, zeros included. In a lower
# triangle the diagonal is the first stored entry of its column.
function with_diagonal(L::SparseMatrixCSC{T}) where T
    n = size(L, 1)
    rows = rowvals(L)
    all(j -> !isempty(nzrange(L, j)) && rows[first(nzrange(L, j))] == j, 1:n) && return L
    I, J, V = findnz(L)
    return sparse(vcat(I, 1:n), vcat(J, 1:n), vcat(V, zeros(T, n)), n, n)   # keeps explicit zeros
end
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

# iparm(8) for the solves on this handle: the solver's `refinement_steps`, or the handle's own
# value when that is `nothing`. Written only when it differs from what the handle holds.
function set_refinement!(ps::Pardiso.MKLPardisoSolver, solver::DistributedEmitterOpt.PardisoSolver,
                         default::Int)
    want = something(solver.refinement_steps, default)
    Pardiso.get_iparm(ps, 8) == want || Pardiso.set_iparm!(ps, 8, want)
    return nothing
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
    DistributedEmitterOpt.checked_pardiso_residual_rtol(solver)
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
    refinement_default = Int(Pardiso.get_iparm(ps, 8))
    set_refinement!(ps, solver, refinement_default)
    f = PardisoFactor{T}(ps, Awork, mtype, size(Awork, 1), copy(Awork.colptr), copy(Awork.rowval),
        false, ReentrantLock(), refinement_default, 0, 0, NaN)
    return factorize!(f, Pardiso.ANALYSIS_NUM_FACT)
end

# New values on the analysed pattern need a numerical factorization only. Otherwise the old
# handle is released and a new factor is built.
function refactor(solver::DistributedEmitterOpt.PardisoSolver, f::PardisoFactor{T},
                  A::SparseMatrixCSC{T,Int}, mtype::Int; use_iparm::Bool) where T
    DistributedEmitterOpt.checked_pardiso_residual_rtol(solver)
    Awork = storage(A, mtype)
    reused = lock(f.lock) do
        if solver.reuse_symbolic && !f.released && f.mtype == mtype && same_pattern(f, Awork)
            f.A = Awork
            set_refinement!(f.ps, solver, f.refinement_default)   # this solver's, as for a new factor
            factorize!(f, Pardiso.NUM_FACT)
        else
            release_pardiso!(f)
            nothing
        end
    end
    return reused === nothing ? new_factor(solver, Awork, mtype; use_iparm) : reused
end

# One solve on the factorization with iparm(12) = mode, written into x; the forward setting is
# restored.
function raw_solve!(x::Vector{T}, f::PardisoFactor{T}, b::Vector{T}, mode::Int) where T
    Pardiso.set_iparm!(f.ps, 12, mode)
    try
        Pardiso.set_phase!(f.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
        Pardiso.pardiso(f.ps, x, f.A, b)
    finally
        Pardiso.set_iparm!(f.ps, 12, forward_mode(f.mtype))
    end
    return x
end

# y = S x for the symmetric matrix S whose lower triangle L, diagonal included, is stored. For
# complex values S is the transpose-symmetric matrix (S = Sᵀ), as Pardiso's mtype 6 reads it.
function symmetric_mul!(y::Vector{T}, L::SparseMatrixCSC{T,Int}, x::Vector{T}) where T
    rows = rowvals(L)
    vals = nonzeros(L)
    fill!(y, zero(T))
    @inbounds for j in 1:size(L, 2)
        xj = x[j]
        s = zero(T)
        for k in nzrange(L, j)
            i = rows[k]
            if i == j
                s += vals[k] * xj
            else
                y[i] += vals[k] * xj         # the stored entry S[i, j]
                s += vals[k] * x[i]          # its mirror S[j, i]
            end
        end
        y[j] += s
    end
    return y
end

# ‖b − Op x‖/‖b‖, with r as workspace, for the system a solve with iparm(12) = mode inverts:
# the symmetric matrix behind a stored triangle (either mode), A for the transposed solve of a
# matrix stored in full (mode 2; Pardiso sees Aᵀ) and Aᵀ for its plain solve (mode 0). A zero b
# asks for a zero residual. A NaN compares false with every limit, so it never passes.
function relative_residual!(r::Vector{T}, f::PardisoFactor{T}, x::Vector{T}, b::Vector{T},
                            mode::Int) where T
    if one_triangle(f.mtype)
        symmetric_mul!(r, f.A, x)
    elseif mode == 2
        mul!(r, f.A, x)
    else
        mul!(r, transpose(f.A), x)
    end
    @. r = b - r
    nr, nb = norm(r), norm(b)
    return nb == 0 ? (nr == 0 ? 0.0 : Inf) : nr / nb
end

# iparm(8) for the second solve of an answer that failed the check: refinement on, with twice
# the steps the handle allows and at least two, in extended precision if the handle asks for it.
resolve_steps(held::Int) = held < 0 ? min(-2, 2held) : max(2, 2held)

# Accept x as the answer of a solve with iparm(12) = mode if its residual is within rtol.
# Otherwise solve once more into x on the same factorization with iterative refinement on, put
# iparm(8) back, and accept that answer or throw. An accepted first answer is not written to,
# and neither is the handle.
function accept_or_refine!(f::PardisoFactor{T}, x::Vector{T}, b::Vector{T}, mode::Int,
                           rtol::Float64) where T
    r = similar(b)
    res1 = relative_residual!(r, f, x, b, mode)
    f.residual_checks += 1
    f.last_residual = res1
    res1 <= rtol && return x
    held = Int(Pardiso.get_iparm(f.ps, 8))
    steps = resolve_steps(held)
    f.residual_resolves += 1                 # counted when attempted
    Pardiso.set_iparm!(f.ps, 8, steps)
    try
        raw_solve!(x, f, b, mode)
    finally
        Pardiso.set_iparm!(f.ps, 8, held)
    end
    res2 = relative_residual!(r, f, x, b, mode)
    f.last_residual = res2
    res2 <= rtol ||
        error("PardisoSolver: solve residual ‖b − op(A)x‖/‖b‖ = $res1 exceeds residual_rtol=$rtol, and it is $res2 after a second solve with iterative refinement (iparm(8)=$steps); mtype $(f.mtype), iparm(12)=$mode, perturbed pivots $(Pardiso.get_iparm(f.ps, 14))")
    @warn "PardisoSolver: solve residual $res1 exceeded residual_rtol=$rtol; a second solve with iterative refinement (iparm(8)=$steps) gave $res2 (mtype $(f.mtype))" maxlog = 5
    return x
end

# Solve with iparm(12) = mode and check the answer against the system that mode inverts. With
# `residual_rtol = 0` the answer is returned unchecked, as before the keyword existed.
function pardiso_solve(solver::DistributedEmitterOpt.PardisoSolver, f::PardisoFactor{T},
                       b::Vector{T}, mode::Int) where T
    rtol = DistributedEmitterOpt.checked_pardiso_residual_rtol(solver)
    lock(f.lock) do
        f.released && throw(ArgumentError("PardisoSolver: this factorization has been released"))
        x = raw_solve!(similar(b), f, b, mode)
        rtol > 0 && accept_or_refine!(f, x, b, mode, rtol)
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

function solve!(solver::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    return pardiso_solve(solver, factor, b, forward_mode(factor.mtype))
end

# Aᴴ x = b ⇔ Aᵀ conj(x) = conj(b), the plain solve of the Aᵀ that Pardiso sees.
function solve_adjoint!(solver::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    T <: Complex || return pardiso_solve(solver, factor, b, 0)
    return conj!(pardiso_solve(solver, factor, conj(b), 0))
end

function release_factor!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor)
    lock(() -> release_pardiso!(factor), factor.lock)
end

end # module
