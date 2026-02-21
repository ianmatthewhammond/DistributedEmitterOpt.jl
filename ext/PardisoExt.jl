"""
    PardisoExt

Package extension that provides the Pardiso solver implementation via GridapPardiso
symbolic/numerical setup, matching the legacy Emitter3DTopOpt workflow.

This extension loads only when both Pardiso.jl and GridapPardiso.jl are available.
"""
module PardisoExt

import DistributedEmitterOpt
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import GridapPardiso
using SparseArrays: SparseMatrixCSC, spzeros, tril

# ---------------------------------------------------------------------------
# Factor wrapper
# ---------------------------------------------------------------------------

mutable struct PardisoFactor{T}
    symbolic::Any
    numerical::Any
    psolver::Any
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

function release_pardiso!(factor::PardisoFactor)
    factor.released && return nothing

    if !isnothing(factor.symbolic)
        try
            # Run GridapPardiso's own finalizer exactly once.
            finalize(factor.symbolic)
        catch
            # Best-effort cleanup. Keep idempotent behavior if finalization fails.
        end
    end

    factor.symbolic = nothing
    factor.numerical = nothing
    factor.psolver = nothing
    factor.A = spzeros(eltype(factor.A), 0, 0)
    factor.released = true
    return nothing
end

function gridap_solver(solver::DistributedEmitterOpt.PardisoSolver, mtype::Int; use_iparm::Bool)
    msglvl = solver.msglvl == 0 ? GridapPardiso.MSGLVL_QUIET : solver.msglvl
    if use_iparm && !isnothing(solver.iparm)
        return GridapPardiso.PardisoSolver(mtype=mtype, iparm=solver.iparm, msglvl=msglvl)
    else
        return GridapPardiso.PardisoSolver(mtype=mtype, msglvl=msglvl)
    end
end

function new_pardiso_factor(
    solver::DistributedEmitterOpt.PardisoSolver,
    A::SparseMatrixCSC{T,Int},
    mtype::Int;
    use_iparm::Bool,
) where T
    Awork = matrix_for_mtype(A, mtype)
    psolver = gridap_solver(solver, mtype; use_iparm)
    symbolic = GridapPardiso.symbolic_setup(psolver, Awork)
    numerical = GridapPardiso.numerical_setup(symbolic, Awork)

    factor = PardisoFactor{T}(symbolic, numerical, psolver, Awork, mtype, sparsity_hash(Awork, mtype), false)
    finalizer(release_pardiso!, factor)
    return factor
end

function refactor_pardiso!(
    solver::DistributedEmitterOpt.PardisoSolver,
    factor::PardisoFactor{T},
    A::SparseMatrixCSC{T,Int},
    mtype::Int;
    use_iparm::Bool,
) where T
    Awork = matrix_for_mtype(A, mtype)
    new_hash = sparsity_hash(Awork, mtype)

    can_reuse_symbolic = solver.reuse_symbolic &&
                         !factor.released &&
                         factor.mtype == mtype &&
                         factor.pattern_hash == new_hash &&
                         !isnothing(factor.symbolic)

    if can_reuse_symbolic
        factor.numerical = GridapPardiso.numerical_setup(factor.symbolic, Awork)
        factor.A = Awork
        factor.pattern_hash = new_hash
        factor.released = false
        return factor
    end

    release_pardiso!(factor)
    return new_pardiso_factor(solver, A, mtype; use_iparm)
end

# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

function lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{ComplexF64,Int})
    mtype = solver.assume_symmetric_maxwell ? solver.mtype_symmetric : solver.mtype
    # Match legacy behavior: custom iparm profile was used for unsymmetric Maxwell only.
    use_iparm = !solver.assume_symmetric_maxwell
    return new_pardiso_factor(solver, A, mtype; use_iparm)
end

function lu!(
    solver::DistributedEmitterOpt.PardisoSolver,
    factor::PardisoFactor{ComplexF64},
    A::SparseMatrixCSC{ComplexF64,Int},
)
    mtype = solver.assume_symmetric_maxwell ? solver.mtype_symmetric : solver.mtype
    use_iparm = !solver.assume_symmetric_maxwell
    return refactor_pardiso!(solver, factor, A, mtype; use_iparm)
end

function filter_lu!(solver::DistributedEmitterOpt.PardisoSolver, A::SparseMatrixCSC{Float64,Int})
    return new_pardiso_factor(solver, A, solver.filter_mtype; use_iparm=false)
end

function filter_lu!(
    solver::DistributedEmitterOpt.PardisoSolver,
    factor::PardisoFactor{Float64},
    A::SparseMatrixCSC{Float64,Int},
)
    return refactor_pardiso!(solver, factor, A, solver.filter_mtype; use_iparm=false)
end

function solve!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    x = similar(b)
    GridapPardiso.solve!(x, factor.numerical, b)
    return x
end

function solve_adjoint!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor{T}, b::Vector{T}) where T
    # Match legacy solveconj path: x = conj(solve(A, conj(b))).
    x = similar(b)
    GridapPardiso.solve!(x, factor.numerical, conj.(b))
    return conj.(x)
end

function release_factor!(::DistributedEmitterOpt.PardisoSolver, factor::PardisoFactor)
    release_pardiso!(factor)
end

end # module
