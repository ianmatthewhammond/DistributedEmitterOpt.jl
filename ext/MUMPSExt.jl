"""
    MUMPSExt

MUMPS through MUMPS.jl 1.x, behind `DistributedEmitterOpt.MUMPSSolver`. When MUMPS.jl has a
`MUMPS.Sequential()` backend (1.7.1 does) the extension runs that sequential MUMPS build, which
needs no MPI; otherwise (1.6.1, for example) it runs MUMPS on one MPI rank and initializes MPI.

MUMPS.jl keeps a job state on each instance: its solve call returns without solving on an
instance that has solved already, and its factorize call does nothing on a factored one.
This extension sets the job itself:
- a new factor runs the analysis (JOB 1) and the factorization (JOB 2); new values on the
  analysed sparsity pattern run JOB 2 alone;
- every right-hand side runs a solve (JOB 3), with ICNTL(9) = 1 for A x = b and 0 for the
  transposed solve. MUMPS transposes without conjugating, so Aᴴ x = b is solved as
  Aᵀ conj(x) = conj(b).

Every phase checks INFOG(1). A factorization that runs out of its estimated workspace
(INFOG(1) = -8, -9, -17 or -20) is repeated with ICNTL(14) doubled, up to four times.
Symmetric storage (sym = 2) hands MUMPS one triangle, which MUMPS.jl takes from the full
matrix.

MUMPS memory lives outside Julia's GC, and MUMPS.jl 1.7 registers no finalizer, so a factor
is released explicitly: by `release_factor!`, when it is refactored on a new pattern, or
when its factorization fails. With MPI, factors still alive at exit are released before MPI
is finalized, since the finalizers of older MUMPS.jl versions would otherwise call MUMPS
after `MPI.Finalize`. All MUMPS calls hold one lock, because MPI runs at the serialized
thread level.
"""
module MUMPSExt

import DistributedEmitterOpt
using DistributedEmitterOpt: MUMPSSolver
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import MUMPS
using MUMPS: Mumps, MPI
using SparseArrays: SparseMatrixCSC

# ═══════════════════════════════════════════════════════════════════════════════
# MUMPS Factorization Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct MUMPSFactor{T}
    mumps::Mumps
    sym::Int
    sequential::Bool            # the MPI-free MUMPS build (MUMPS.Sequential(), MUMPS.jl 1.7.1)
    n::Int                      # size and pattern of the last analysis (copies)
    colptr::Vector{Int}
    rowval::Vector{Int}
    released::Bool
end

const MUMPS_LOCK = ReentrantLock()
const LIVE = WeakKeyDict{MUMPSFactor,Nothing}()
const EXIT_HOOK = Ref(false)
const HAS_SEQUENTIAL = isdefined(MUMPS, :Sequential)

# INFOG(1) values that more workspace (ICNTL(14)) can cure.
const WORKSPACE_ERRORS = (-8, -9, -17, -20)

function ensure_mpi()
    MPI.Initialized() || MPI.Init()
    if !EXIT_HOOK[]
        # Registered after MPI's own Finalize hook, so it runs first (atexit hooks run last-in first-out).
        atexit(() -> foreach(release_mumps!, collect(keys(LIVE))))
        EXIT_HOOK[] = true
    end
    return nothing
end

function check_mumps(m::Mumps, phase)
    m.infog[1] < 0 && error("MUMPS $phase failed: INFOG(1) = $(m.infog[1]), INFOG(2) = $(m.infog[2])")
    return m
end

function run_job!(m::Mumps, job::Int)
    MUMPS.set_job!(m, job)
    MUMPS.invoke_mumps!(m)
    return m
end

# Numerical factorization (JOB 2) on the analysed pattern, repeated with more workspace.
function factorize_numerically!(m::Mumps)
    for attempt in 1:5
        run_job!(m, 2)
        (m.infog[1] in WORKSPACE_ERRORS && attempt < 5) || break
        MUMPS.set_icntl!(m, 14, 2 * max(m.icntl[14], 20); displaylevel=0)
    end
    return check_mumps(m, "factorization")
end

function release_mumps!(f::MUMPSFactor)
    f.released && return nothing
    f.released = true
    delete!(LIVE, f)
    if !f.sequential && MPI.Finalized()
        f.mumps._finalized = true   # MUMPS cannot be called after MPI.Finalize, nor by its finalizer
    else
        MUMPS.finalize!(f.mumps)
    end
    return nothing
end

function mumps_instance(::Type{T}, sym::Int, sequential::Bool) where T
    sequential && return Mumps{T}(sym; backend=MUMPS.Sequential())
    ensure_mpi()
    return HAS_SEQUENTIAL ? Mumps{T}(sym; backend=MUMPS.Parallel()) : Mumps{T}(sym)
end

function new_factor(solver::MUMPSSolver, A::SparseMatrixCSC{T,Int}, sym::Int;
                    sequential::Bool=HAS_SEQUENTIAL) where T
    lock(MUMPS_LOCK) do
        m = mumps_instance(T, sym, sequential)
        if !solver.verbose
            # Errors only (ICNTL 1-4); MUMPS otherwise prints statistics for every phase.
            for (i, v) in ((1, 6), (2, 0), (3, 0), (4, 1))
                MUMPS.set_icntl!(m, i, v; displaylevel=0)
            end
        end
        f = MUMPSFactor{T}(m, sym, sequential, size(A, 1), copy(A.colptr), copy(A.rowval), false)
        LIVE[f] = nothing
        try
            MUMPS.associate_matrix!(m, A; unsafe=true)   # copies the entries
            check_mumps(run_job!(m, 1), "analysis")
            factorize_numerically!(m)
        catch
            release_mumps!(f)
            rethrow()
        end
        return f
    end
end

# New values on the analysed pattern need the numerical factorization only. Otherwise the
# old instance is released and a new factor is built.
function refactor(solver::MUMPSSolver, f::MUMPSFactor{T}, A::SparseMatrixCSC{T,Int}, sym::Int) where T
    reused = lock(MUMPS_LOCK) do
        if !f.released && f.sym == sym && size(A) == (f.n, f.n) && A.colptr == f.colptr && A.rowval == f.rowval
            try
                MUMPS.associate_matrix!(f.mumps, A; unsafe=true)
                factorize_numerically!(f.mumps)
            catch
                release_mumps!(f)
                rethrow()
            end
            f
        else
            release_mumps!(f)
            nothing
        end
    end
    return reused === nothing ? new_factor(solver, A, sym; sequential=f.sequential) : reused
end

# Solve (JOB 3) with ICNTL(9) = 1 (A x = b) or 0 (Aᵀ x = b), then restore ICNTL(9) = 1.
function mumps_solve(f::MUMPSFactor{T}, b::Vector{T}, transposed::Bool) where T
    lock(MUMPS_LOCK) do
        f.released && throw(ArgumentError("MUMPSSolver: this factorization has been released"))
        m = f.mumps
        MUMPS.associate_rhs!(m, b)               # a copy; MUMPS overwrites it with the solution
        MUMPS.set_icntl!(m, 9, transposed ? 0 : 1; displaylevel=0)
        try
            check_mumps(run_job!(m, 3), "solve")
        finally
            MUMPS.set_icntl!(m, 9, 1; displaylevel=0)
        end
        return MUMPS.get_sol!(similar(b), m)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interface Implementation
# ═══════════════════════════════════════════════════════════════════════════════

"""Factorize complex Maxwell matrix with MUMPS."""
function lu!(solver::MUMPSSolver, A::SparseMatrixCSC{ComplexF64,Int})
    new_factor(solver, A, DistributedEmitterOpt.checked_mumps_sym(solver.symmetric, A))
end

function lu!(solver::MUMPSSolver, factor::MUMPSFactor{ComplexF64}, A::SparseMatrixCSC{ComplexF64,Int})
    refactor(solver, factor, A, DistributedEmitterOpt.checked_mumps_sym(solver.symmetric, A))
end

"""Factorize real filter matrix with MUMPS (symmetric storage, checked)."""
function filter_lu!(solver::MUMPSSolver, A::SparseMatrixCSC{Float64,Int})
    new_factor(solver, A, DistributedEmitterOpt.checked_mumps_sym(true, A))
end

function filter_lu!(solver::MUMPSSolver, factor::MUMPSFactor{Float64}, A::SparseMatrixCSC{Float64,Int})
    refactor(solver, factor, A, DistributedEmitterOpt.checked_mumps_sym(true, A))
end

"""Solve Ax = b using MUMPS factorization."""
function solve!(::MUMPSSolver, factor::MUMPSFactor{T}, b::Vector{T}) where T
    mumps_solve(factor, b, false)
end

"""
Solve Aᴴ x = b (adjoint system). MUMPS's transposed solve (ICNTL(9) ≠ 1) uses the
plain transpose Aᵀ and never conjugates, so the complex case solves
Aᵀ conj(x) = conj(b); the real case is the transposed solve itself.
"""
function solve_adjoint!(::MUMPSSolver, factor::MUMPSFactor{T}, b::Vector{T}) where T
    T <: Complex || return mumps_solve(factor, b, true)
    return conj!(mumps_solve(factor, conj(b), true))
end

function release_factor!(::MUMPSSolver, factor::MUMPSFactor)
    lock(() -> release_mumps!(factor), MUMPS_LOCK)
end

end # module
