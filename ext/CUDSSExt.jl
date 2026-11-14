"""
    CUDSSExt

NVIDIA cuDSS through CUDSS.jl (1.0 or later), behind `DistributedEmitterOpt.CUDSSSolver`. The
extension loads with `using CUDSS` and needs a CUDA-capable GPU.

Matrices arrive as Julia CSC on the host and are converted to CSR on the device. A Maxwell
matrix checked to be complex symmetric is factored from its lower triangle with structure
"S" (LDLᵀ: for complex values cuDSS uses the transpose, not the Hermitian, factorization);
any other matrix is factored with structure "G" (general LDU). Real filter matrices follow
the same rule. Solves are

    A x = b    →  plain solve
    Aᴴ x = b   →  "S": A conj(x) = conj(b), a plain solve on conj(b), conjugated
                  "G": cuDSS solve mode 'C' (conjugate transpose; 'T' for real matrices)

so oblique Bloch operators, which are not complex symmetric, get true adjoints.

Every solve is checked against the operator it is meant to invert, built from the full
matrix A on the device: A for a forward solve, Aᴴ for an adjoint solve, whatever the stored
structure. A matrix accepted as "S" is symmetric only to `SYMMETRY_RTOL`, so its factor is
used as the correction solver of up to `refine_steps` refinement steps on that true residual;
a solve that still misses `residual_rtol` throws.

The analysis (reordering and symbolic factorization) is reused while the structure and the
pattern of the stored matrix match a copy taken when it ran; new values then cost one
numerical factorization. After every factorization cuDSS's device-side `info` is checked and
the number of perturbed pivots is kept in the factor (`npivots`). A factor whose update or
factorization fails is released, never left half updated.

Handles and lifetimes. CUDSS.jl lends each Julia task a cuDSS handle from a pool and takes it back
when the task is collected, and its objects are destroyed by GC finalizers on any thread. Here
every factor is built on one handle owned by this extension (one per CUDA context, never
returned), all cuDSS work is serialized by one process-wide lock with the device synchronized
around each call, and the solve mode is set on every solve, so a factor can be used from any task.
Each factor's cuDSS objects are held in a registry and destroyed only under the lock: by
`release_factor!` at once (together with the factor's device arrays), or, for a factor dropped
without release, at the next call into the extension after the GC has collected it.
"""
module CUDSSExt

import DistributedEmitterOpt
import DistributedEmitterOpt: lu!, filter_lu!, solve!, solve_adjoint!, release_factor!
import CUDSS
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, tril

const CUDACore = CUDSS.CUDACore
const cuSPARSE = CUDSS.cuSPARSE
const DEO = DistributedEmitterOpt

"Serializes all cuDSS work: the extension's handle is shared by every factor and is not thread safe."
const CUDSS_LOCK = ReentrantLock()

"The extension's own cuDSS handle per CUDA context: created under CUDSS_LOCK, never destroyed."
const HANDLES = Dict{Any,Any}()

"cuDSS solvers of live factors, by factor id: their objects are destroyed only under CUDSS_LOCK."
const LIVE = Dict{UInt64,Any}()

"Ids of factors collected without release_factor! (pushed by finalizers, drained under CUDSS_LOCK)."
const DROPPED = UInt64[]
const DROPPED_LOCK = Threads.SpinLock()
const NEXT_ID = Threads.Atomic{UInt64}(0)

# Call with CUDSS_LOCK held.
function own_handle()
    ctx = CUDACore.context()
    return get!(() -> CUDSS.cudssCreate(), HANDLES, ctx)
end

"Wait for all device work, whichever stream (task) issued it."
devsync() = isdefined(CUDACore, :device_synchronize) ? CUDACore.device_synchronize() : CUDACore.synchronize()

# ---------------------------------------------------------------------------
# Factor wrapper
# ---------------------------------------------------------------------------

mutable struct CudssFactor{T}
    solver::Any                 # CUDSS.CudssSolver, or nothing once released
    structure::String           # "S" (lower triangle, A = Aᵀ) or "G"
    stored::Any                 # device CSR matrix handed to cuDSS (kept alive)
    full::Any                   # device CSR of the full A, for residuals
    n::Int                      # size and pattern of the stored matrix at the last analysis (copies)
    colptr::Vector{Int}
    rowval::Vector{Int}
    xbuf::Any                   # device work vectors for the factorization phases
    bbuf::Any
    npivots::Int                # perturbed pivots in the last factorization (-1 if unavailable)
    released::Bool
    id::UInt64                  # key of its cuDSS solver in LIVE
end

# Finalizer: never touches the GPU or CUDSS_LOCK; the next locked call destroys the objects.
function dropped!(f::CudssFactor)
    f.released && return nothing
    lock(DROPPED_LOCK)
    try
        push!(DROPPED, f.id)
    finally
        unlock(DROPPED_LOCK)
    end
    return nothing
end

storage(A::SparseMatrixCSC, structure::String) = structure == "S" ? tril(A) : A
same_pattern(f::CudssFactor, M::SparseMatrixCSC) =
    size(M) == (f.n, f.n) && M.colptr == f.colptr && M.rowval == f.rowval

device(M::SparseMatrixCSC) = cuSPARSE.CuSparseMatrixCSR(M)

# Call with CUDSS_LOCK held. Destroys the cuDSS data, configuration and matrix descriptor of one
# registered solver (not the handle); idempotent.
function destroy!(id::UInt64)
    s = pop!(LIVE, id, nothing)
    s === nothing && return nothing
    try
        devsync()
    catch
    end
    for obj in (s.data, s.config, s.matrix)
        try
            finalize(obj)
        catch
        end
    end
    return nothing
end

# Call with CUDSS_LOCK held: destroy the solvers of factors the GC has collected.
function drain_dropped!()
    ids = lock(DROPPED_LOCK) do
        ids = copy(DROPPED)
        empty!(DROPPED)
        ids
    end
    foreach(destroy!, ids)
    return nothing
end

free_device!(a) = (isdefined(CUDACore, :unsafe_free!) && a isa CUDACore.CuArray) && CUDACore.unsafe_free!(a)
function free_device_matrix!(M)
    M === nothing && return nothing
    for name in (:rowPtr, :colVal, :nzVal)
        hasproperty(M, name) && free_device!(getproperty(M, name))
    end
    return nothing
end

# Call with CUDSS_LOCK held. Destroys the factor's cuDSS objects and frees its device arrays;
# idempotent.
function release_cudss!(f::CudssFactor)
    f.released && return nothing
    f.released = true
    destroy!(f.id)
    try
        free_device_matrix!(f.stored)
        f.full === f.stored || free_device_matrix!(f.full)
        free_device!(f.xbuf)
        free_device!(f.bbuf)
    catch
    end
    f.solver = nothing
    f.stored = nothing
    f.full = nothing
    f.xbuf = nothing
    f.bbuf = nothing
    return nothing
end

# Numerical factorization (and the analysis when `analyse`), then cuDSS's info. Call with
# CUDSS_LOCK held. A failure releases the factor before rethrowing.
function factorize!(f::CudssFactor; analyse::Bool)
    try
        devsync()
        analyse && CUDSS.cudss("analysis", f.solver, f.xbuf, f.bbuf; asynchronous=false)
        CUDSS.cudss("factorization", f.solver, f.xbuf, f.bbuf; asynchronous=false)
        devsync()
        info = CUDSS.cudss_get(f.solver, "info")
        info == 0 || error("CUDSSSolver: cuDSS factorization failed (structure $(f.structure), info=$info)")
        f.npivots = try
            Int(CUDSS.cudss_get(f.solver, "npivots"))
        catch
            -1
        end
    catch
        release_cudss!(f)
        rethrow()
    end
    return f
end

function new_factor(A::SparseMatrixCSC{T,Int}, structure::String) where T
    CUDACore.functional() || error("CUDSSSolver: no usable CUDA GPU (CUDACore.functional() is false)")
    M = storage(A, structure)
    lock(CUDSS_LOCK) do
        drain_dropped!()
        stored = device(M)
        full = structure == "S" ? device(A) : stored
        matrix = CUDSS.CudssMatrix(stored, structure, structure == "S" ? 'L' : 'F')
        solver = CUDSS.CudssSolver(matrix, CUDSS.CudssConfig(), CUDSS.CudssData(own_handle()))
        n = size(A, 1)
        id = Threads.atomic_add!(NEXT_ID, UInt64(1))
        LIVE[id] = solver
        f = CudssFactor{T}(solver, structure, stored, full, n, copy(M.colptr), copy(M.rowval),
            CUDACore.CuVector{T}(undef, n), CUDACore.CuVector{T}(undef, n), -1, false, id)
        finalizer(dropped!, f)
        factorize!(f; analyse=true)
    end
end

# New values on the analysed pattern and structure need a numerical factorization only.
# Otherwise the old factor is released and a new one is built. The device copies are made
# before the factor is touched; any failure after that releases it.
function refactor(solver::DEO.CUDSSSolver, f::CudssFactor{T}, A::SparseMatrixCSC{T,Int},
                  structure::String) where T
    M = storage(A, structure)
    reused = lock(CUDSS_LOCK) do
        drain_dropped!()
        if solver.reuse_analysis && !f.released && f.structure == structure && same_pattern(f, M)
            stored = device(M)
            full = structure == "S" ? device(A) : stored
            try
                CUDSS.cudss_update(f.solver, stored)
                f.stored = stored
                f.full = full
            catch
                release_cudss!(f)
                rethrow()
            end
            factorize!(f; analyse=false)
        else
            release_cudss!(f)
            nothing
        end
    end
    return reused === nothing ? new_factor(A, structure) : reused
end

# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

# One cuDSS solve of the factored matrix with the given mode ('N', 'T' or 'C'). The mode is set
# on every call, so no earlier call (or failure) can leave a stale one.
function raw_solve!(f::CudssFactor, x, b, mode::Char)
    CUDSS.cudss_set(f.solver, "solve_mode", mode)
    devsync()
    CUDSS.cudss("solve", f.solver, x, b; asynchronous=false)
    devsync()
    return x
end

# z ≈ op(A)⁻¹ r from the factor, where op is the identity (forward) or the adjoint.
function correction(f::CudssFactor{T}, r, adjoint::Bool) where T
    adjoint || return raw_solve!(f, similar(r), r, 'N')
    if f.structure == "S"
        T <: Complex || return raw_solve!(f, similar(r), r, 'N')
        return conj!(raw_solve!(f, similar(r), conj(r), 'N'))
    end
    return raw_solve!(f, similar(r), r, T <: Complex ? 'C' : 'T')
end

# The true operator, from the full matrix: A x, or Aᴴ x (Aᵀ x for real matrices).
apply(f::CudssFactor, x, adjoint::Bool) = adjoint ? f.full' * x : f.full * x

# Solve op(A) x = b on the device, check the residual of op(A) itself, refine if needed.
function checked_solve(solver::DEO.CUDSSSolver, f::CudssFactor{T}, b::Vector{T}, adjoint::Bool) where T
    lock(CUDSS_LOCK) do
        drain_dropped!()
        f.released && throw(ArgumentError("CUDSSSolver: this factorization has been released"))
        bd = CUDACore.CuVector{T}(b)
        x = correction(f, bd, adjoint)
        solver.residual_rtol > 0 || return Array(x)
        nb = norm(bd)
        nb == 0 && return Array(x)
        rel = Inf
        for step in 0:solver.refine_steps
            r = bd .- apply(f, x, adjoint)
            rel = norm(r) / nb
            (isfinite(rel) && rel <= solver.residual_rtol) && return Array(x)
            step == solver.refine_steps && break
            x .+= correction(f, r, adjoint)
        end
        error("CUDSSSolver: $(adjoint ? "adjoint" : "forward") solve residual $rel exceeds residual_rtol=$(solver.residual_rtol) after $(solver.refine_steps) refinement steps (structure $(f.structure), npivots $(f.npivots))")
    end
end

# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

lu!(solver::DEO.CUDSSSolver, A::SparseMatrixCSC{ComplexF64,Int}) =
    new_factor(A, DEO.checked_cudss_structure(DEO.checked_cudss_options(solver), A))

lu!(solver::DEO.CUDSSSolver, factor::CudssFactor{ComplexF64}, A::SparseMatrixCSC{ComplexF64,Int}) =
    refactor(solver, factor, A, DEO.checked_cudss_structure(DEO.checked_cudss_options(solver), A))

filter_lu!(solver::DEO.CUDSSSolver, A::SparseMatrixCSC{Float64,Int}) =
    (DEO.checked_cudss_options(solver); new_factor(A, DEO.checked_cudss_filter_structure(A)))

filter_lu!(solver::DEO.CUDSSSolver, factor::CudssFactor{Float64}, A::SparseMatrixCSC{Float64,Int}) =
    refactor(DEO.checked_cudss_options(solver), factor, A, DEO.checked_cudss_filter_structure(A))

solve!(solver::DEO.CUDSSSolver, factor::CudssFactor{T}, b::Vector{T}) where T =
    checked_solve(solver, factor, b, false)

solve_adjoint!(solver::DEO.CUDSSSolver, factor::CudssFactor{T}, b::Vector{T}) where T =
    checked_solve(solver, factor, b, true)

release_factor!(::DEO.CUDSSSolver, factor::CudssFactor) = lock(() -> release_cudss!(factor), CUDSS_LOCK)

end # module
