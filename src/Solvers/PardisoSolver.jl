"""
    PardisoSolver

Optional sparse direct solver using Intel MKL Pardiso through Pardiso.jl.
Requires the user to independently install Pardiso.jl.

The actual implementation is in ext/PardisoExt.jl and loads with Pardiso.jl alone
(`import Pardiso`). MKL Pardiso needs MKL, which is not available everywhere (not on
Apple Silicon, for example); `UmfpackSolver()` runs on every platform.
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
    # Matrix types for Pardiso:
    #  2 = real symmetric positive definite      (filter; one triangle)
    # -2 = real symmetric indefinite             (filter; one triangle)
    #  6 = complex symmetric                     (Maxwell; one triangle)
    # 11 = real unsymmetric                      (filter)
    # 13 = complex unsymmetric                   (Maxwell)
    #
    # `mtype`               : Maxwell mtype when `assume_symmetric_maxwell=false`
    # `mtype_symmetric`     : Maxwell mtype when `assume_symmetric_maxwell=true`
    # `filter_mtype`        : Filter real-system mtype
    # `assume_symmetric_maxwell` : Use symmetric storage/mtype for Maxwell solves
    # `reuse_symbolic`      : Reuse symbolic setup when sparsity pattern is unchanged
    # `msglvl`              : Pardiso statistics output when > 0
    # `nprocs`              : MKL Pardiso threads when > 0 (a process-wide MKL setting);
    #                         0 keeps MKL's own count (MKL_NUM_THREADS or all cores)
    # `iparm`               : Profile for unsymmetric Maxwell solves (checked by
    #                         `checked_pardiso_iparm`); `nothing` keeps MKL's defaults.
    #                         iparm(12) is set by the backend for each solve.
    mtype::Int = 13
    mtype_symmetric::Int = 6
    filter_mtype::Int = -2
    assume_symmetric_maxwell::Bool = false
    reuse_symbolic::Bool = true
    msglvl::Int = 0
    nprocs::Int = 0
    iparm::Union{Nothing,Vector{Int}} = legacy_pardiso_iparm()
end

"Validate a Maxwell matrix before a symmetric storage mode discards a triangle."
function checked_maxwell_mtype(solver::PardisoSolver, A)
    mtype = solver.assume_symmetric_maxwell ? solver.mtype_symmetric : solver.mtype
    if mtype in (1, 2, -2, 3, 4, -4, 6)
        reflected = mtype in (4, -4) ? adjoint(A) : transpose(A)
        norm(A - reflected) <= 1e-12 * norm(A) ||
            throw(ArgumentError("PardisoSolver: Maxwell matrix is incompatible with symmetric mtype=$mtype; oblique Bloch fields require assume_symmetric_maxwell=false and mtype=13"))
    end
    return mtype
end

"Validate the filter mtype, and the matrix before a symmetric type discards a triangle."
function checked_filter_mtype(solver::PardisoSolver, A)
    mtype = solver.filter_mtype
    mtype in (-2, 2, 11) ||
        throw(ArgumentError("PardisoSolver: filter_mtype=$mtype is not supported; use -2 (default), 2 or 11"))
    mtype == 11 || norm(A - transpose(A)) <= 1e-12 * norm(A) ||
        throw(ArgumentError("PardisoSolver: filter_mtype=$mtype keeps one triangle, but the filter matrix is not symmetric; use filter_mtype=11"))
    return mtype
end

"""
    checked_pardiso_iparm(iparm) -> iparm

Check an iparm profile before it replaces MKL's defaults. The backend relies on iparm(1) = 1
(otherwise MKL ignores the array, including iparm(12), which selects the transposed solve),
iparm(6) = 0 (the solution is written to x and b is kept), iparm(28) = 0 (double precision)
and iparm(35) = 0 (one-based indices).
"""
function checked_pardiso_iparm(iparm::AbstractVector{<:Integer})
    length(iparm) == 64 ||
        throw(ArgumentError("PardisoSolver: iparm must have 64 entries, got $(length(iparm))"))
    for (i, want, why) in ((1, 1, "use the given values; with 0 MKL ignores iparm(12)"),
                           (6, 0, "solution in x, b kept"),
                           (28, 0, "double precision"),
                           (35, 0, "one-based indices"))
        iparm[i] == want ||
            throw(ArgumentError("PardisoSolver: iparm($i) must be $want ($why), got $(iparm[i])"))
    end
    return iparm
end

# Stub implementations that error if extension not loaded
const PARDISO_NOT_LOADED = "Pardiso extension not loaded. Install Pardiso.jl (using Pkg; Pkg.add(\"Pardiso\")) and load it with `import Pardiso`; MKL Pardiso needs MKL, so use UmfpackSolver() where MKL is not available"

function lu!(::PardisoSolver, A::SparseMatrixCSC)
    error(PARDISO_NOT_LOADED)
end

function lu!(solver::PardisoSolver, factor, A::SparseMatrixCSC)
    lu!(solver, A)
end

function filter_lu!(::PardisoSolver, A::SparseMatrixCSC)
    error(PARDISO_NOT_LOADED)
end

function filter_lu!(solver::PardisoSolver, factor, A::SparseMatrixCSC)
    filter_lu!(solver, A)
end

function solve!(::PardisoSolver, factor, b::Vector)
    error(PARDISO_NOT_LOADED)
end

function solve_adjoint!(::PardisoSolver, factor, b::Vector)
    error(PARDISO_NOT_LOADED)
end

release_factor!(::PardisoSolver, factor) = nothing
