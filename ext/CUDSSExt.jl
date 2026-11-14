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
the number of perturbed pivots is kept in the factor (`npivots`). A factor whose construction,
update or factorization fails is released, never left half built or half updated.

Handles, contexts and lifetimes. CUDSS.jl lends each Julia task a cuDSS handle from a pool and
takes it back when the task is collected, and its objects are destroyed by GC finalizers on any
thread. Here every factor is built, in the CUDA context current at its creation, on a handle owned
by this extension (one per context, never returned), and all its later work runs in that context.
Every cuDSS object (data, configuration, the matrix descriptor and the dense descriptors of the
factorization buffers) is held in a registry until it is destroyed under one process-wide lock;
the dense descriptors of each solve are created and destroyed under the same lock. All cuDSS work
is serialized by that lock and runs on one stream the extension owns per context (set on the
handle once, so every analysis binds to it; the calling task is switched to it for the duration of
each operation), and the solve mode is set on every solve, so a factor can be used from any task.
`release_factor!` destroys a factor's cuDSS objects and frees its device arrays at once; a factor
dropped without release is destroyed at the next call into the extension after the GC has
collected it. If cleanup cannot start (the context cannot be made current, or the stream cannot be
synchronized), the objects stay registered and are retried at the next call. Once a destroy has
started it is never repeated: cuDSS consumes a valid object when its cleanup begins, even if the
cleanup then reports an error (cuDSS 1.0 API reference), so a repeat would destroy it twice.
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

"Serializes all cuDSS work: a handle is shared by every factor of its context and is not thread safe."
const CUDSS_LOCK = ReentrantLock()

"The extension's own cuDSS handle per CUDA context: created under CUDSS_LOCK, never destroyed."
const HANDLES = Dict{Any,Any}()

"The extension's own CUDA stream per context: every cuDSS call and every array operation of the extension runs on it."
const STREAMS = Dict{Any,Any}()

"cuDSS objects of live factors, by factor id, as (context, objects): destroyed only under CUDSS_LOCK."
const LIVE = Dict{UInt64,Any}()

"Ids of factors collected without release_factor! (pushed by finalizers, drained under CUDSS_LOCK)."
const DROPPED = UInt64[]
const DROPPED_LOCK = Threads.SpinLock()
const NEXT_ID = Threads.Atomic{UInt64}(0)

"Test hook: while true, `destroy!` fails before any cleanup starts, as a failed context switch would."
const CLEANUP_FAULT = Ref(false)

# Streams. cuDSS data keeps the stream its analysis ran on, so a factor executes there whichever
# task calls it, and CUDA.jl gives each task its own non-blocking stream. The extension therefore
# owns one stream per context, sets it on its handle once (so every analysis binds to it), and runs
# every locked operation with the calling task switched to it (`on_device`): uploads, cuDSS calls,
# residuals, synchronization, frees and destruction are all ordered on that one stream.
# Call with CUDSS_LOCK held, in the context.
own_stream(ctx) = get!(() -> CUDACore.CuStream(), STREAMS, ctx)
function own_handle(ctx)
    get!(HANDLES, ctx) do
        h = CUDSS.cudssCreate()
        CUDSS.cudssSetStream(h, own_stream(ctx))
        h
    end
end

"Run `f` in context `ctx` on the extension's stream for it. Call with CUDSS_LOCK held."
on_device(f, ctx) = CUDACore.context!(() -> CUDACore.stream!(f, own_stream(ctx)), ctx)

"Wait for the current stream (the extension's, inside `on_device`)."
devsync() = CUDACore.synchronize()

# Destroy CUDSS.jl objects, each exactly once: `finalize` runs the object's CUDSS.jl finalizer (the
# cuDSS destroy call) and removes it, so the GC never destroys the object again. A destroy that
# reports an error has still consumed the object (cuDSS 1.0 API reference: "A valid ... object is
# consumed once cleanup begins, including when cleanup reports an error"), so it is not retried;
# Julia reports the error on stderr, as for any failing finalizer.
finalize_all!(objs) = foreach(finalize, objs)

free_device!(a) = (isdefined(CUDACore, :unsafe_free!) && a isa CUDACore.CuArray) && CUDACore.unsafe_free!(a)
function free_device_matrix!(M)
    M === nothing && return nothing
    for name in (:rowPtr, :colVal, :nzVal)
        hasproperty(M, name) && free_device!(getproperty(M, name))
    end
    return nothing
end

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
    xdesc::Any                  # their cuDSS dense descriptors
    bdesc::Any
    ctx::Any                    # CUDA context the factor lives in
    npivots::Int                # perturbed pivots in the last factorization (-1 if unavailable)
    released::Bool
    id::UInt64                  # key of its cuDSS objects in LIVE
end

storage(A::SparseMatrixCSC, structure::String) = structure == "S" ? tril(A) : A
same_pattern(f::CudssFactor, M::SparseMatrixCSC) =
    size(M) == (f.n, f.n) && M.colptr == f.colptr && M.rowval == f.rowval

device(M::SparseMatrixCSC) = cuSPARSE.CuSparseMatrixCSR(M)

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

# Call with CUDSS_LOCK held. Destroys one registered set of cuDSS objects (not the handle) on its
# context's stream; idempotent. Returns false, keeping the entry (and so the objects, and the device
# arrays their descriptors hold) rooted for a retry, if cleanup cannot start: the context cannot be
# made current or the stream cannot be synchronized. Once the destroy calls start, every object is
# consumed (see `finalize_all!`) and the entry goes.
function destroy!(id::UInt64)
    entry = get(LIVE, id, nothing)
    entry === nothing && return true
    ctx, objs = entry
    started = false
    try
        CLEANUP_FAULT[] && error("CUDSSExt: injected cleanup failure (test hook)")
        on_device(ctx) do
            devsync()
            started = true
            finalize_all!(objs)
        end
    catch
        started || return false
    end
    delete!(LIVE, id)
    return true
end

queue_dropped!(id::UInt64) = lock(() -> push!(DROPPED, id), DROPPED_LOCK)

# Call with CUDSS_LOCK held: destroy the objects of factors the GC has collected (those that
# cannot be destroyed now stay registered and are retried).
function drain_dropped!()
    ids = lock(DROPPED_LOCK) do
        ids = copy(DROPPED)
        empty!(DROPPED)
        ids
    end
    foreach(id -> destroy!(id) || queue_dropped!(id), ids)
    return nothing
end

# Call with CUDSS_LOCK held. Destroys the factor's cuDSS objects and frees its device arrays;
# idempotent. If cleanup cannot start, the factor is still released (it cannot be used again) and
# its registered objects are queued, like a dropped factor's, for the next call to retry.
function release_cudss!(f::CudssFactor)
    f.released && return nothing
    f.released = true
    # Device arrays are freed only once the cuDSS objects using them are destroyed; otherwise they
    # stay referenced by those (still registered) objects and go with them.
    if !destroy!(f.id)
        queue_dropped!(f.id)
    else
        try
            on_device(f.ctx) do
                free_device_matrix!(f.stored)
                f.full === f.stored || free_device_matrix!(f.full)
                free_device!(f.xbuf)
                free_device!(f.bbuf)
            end
        catch
        end
    end
    f.solver = nothing
    f.stored = nothing
    f.full = nothing
    f.xbuf = nothing
    f.bbuf = nothing
    f.xdesc = nothing
    f.bdesc = nothing
    return nothing
end

# Numerical factorization (and the analysis when `analyse`), then cuDSS's info. Call with
# CUDSS_LOCK held, in the factor's context. A failure releases the factor before rethrowing.
function factorize!(f::CudssFactor; analyse::Bool)
    try
        devsync()
        analyse && CUDSS.cudss("analysis", f.solver, f.xdesc, f.bdesc; asynchronous=false)
        CUDSS.cudss("factorization", f.solver, f.xdesc, f.bdesc; asynchronous=false)
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

# Device arrays first, then cuDSS objects, then registration: a failure on the way releases
# everything acquired so far.
function new_factor(A::SparseMatrixCSC{T,Int}, structure::String) where T
    CUDACore.functional() || error("CUDSSSolver: no usable CUDA GPU (CUDACore.functional() is false)")
    M = storage(A, structure)
    n = size(A, 1)
    colptr, rowval = copy(M.colptr), copy(M.rowval)
    ctx = CUDACore.context()
    lock(CUDSS_LOCK) do
        drain_dropped!()
        on_device(ctx) do
            arrays = Any[]
            objs = Any[]
            f = nothing
            try
                stored = device(M); push!(arrays, stored)
                full = structure == "S" ? device(A) : stored
                full === stored || push!(arrays, full)
                xbuf = CUDACore.CuVector{T}(undef, n); push!(arrays, xbuf)
                bbuf = CUDACore.CuVector{T}(undef, n); push!(arrays, bbuf)
                data = CUDSS.CudssData(own_handle(ctx)); push!(objs, data)
                config = CUDSS.CudssConfig(); push!(objs, config)
                matrix = CUDSS.CudssMatrix(stored, structure, structure == "S" ? 'L' : 'F'); push!(objs, matrix)
                xdesc = CUDSS.CudssMatrix(xbuf); push!(objs, xdesc)
                bdesc = CUDSS.CudssMatrix(bbuf); push!(objs, bdesc)
                solver = CUDSS.CudssSolver(matrix, config, data)
                id = Threads.atomic_add!(NEXT_ID, UInt64(1))
                f = CudssFactor{T}(solver, structure, stored, full, n, colptr, rowval, xbuf, bbuf,
                    xdesc, bdesc, ctx, -1, false, id)
                LIVE[id] = (ctx, objs)
                finalizer(dropped!, f)
            catch
                f === nothing || pop!(LIVE, f.id, nothing)
                try
                    devsync()
                catch
                end
                finalize_all!(objs)
                foreach(a -> (try a isa CUDACore.CuArray ? free_device!(a) : free_device_matrix!(a) catch end), arrays)
                f === nothing || (f.released = true)
                rethrow()
            end
            factorize!(f; analyse=true)
        end
    end
end

# New values on the analysed pattern and structure need a numerical factorization only: they are
# copied into the values array cuDSS already has, so every array cuDSS has seen (row offsets, column
# indices, values) stays the same allocation for the life of the factor. (cuDSS does not document
# whether its data keeps the pointers of the analysed matrix; a factor that replaced and freed them
# hit an illegal address in a later factorization.) Otherwise the old factor is released and a new
# one is built. The device copies are made before the factor is touched; any failure after that
# releases it.
function refactor(solver::DEO.CUDSSSolver, f::CudssFactor{T}, A::SparseMatrixCSC{T,Int},
                  structure::String) where T
    M = storage(A, structure)
    reused = lock(CUDSS_LOCK) do
        drain_dropped!()
        if solver.reuse_analysis && !f.released && f.structure == structure && same_pattern(f, M)
            on_device(f.ctx) do
                fresh = device(M)                     # same pattern, so the same CSR value order
                full = f.full === f.stored ? nothing : device(A)   # residual matrix, unseen by cuDSS
                try
                    copyto!(f.stored.nzVal, fresh.nzVal)
                    if full !== nothing
                        old_full, f.full = f.full, full
                        free_device_matrix!(old_full)
                    end
                catch
                    release_cudss!(f)
                    rethrow()
                finally
                    free_device_matrix!(fresh)
                end
                factorize!(f; analyse=false)
                f
            end
        else
            release_cudss!(f)
            nothing
        end
    end
    return reused === nothing ? new_factor(A, structure) : reused
end

# ---------------------------------------------------------------------------
# Solves (call with CUDSS_LOCK held, in the factor's context)
# ---------------------------------------------------------------------------

# One cuDSS solve of the factored matrix with the given mode ('N', 'T' or 'C'). The mode is set
# on every call, so no earlier call (or failure) can leave a stale one; the dense descriptors of
# this solve are destroyed before the lock is released.
function raw_solve!(f::CudssFactor, x, b, mode::Char)
    CUDSS.cudss_set(f.solver, "solve_mode", mode)
    X = CUDSS.CudssMatrix(x)
    try
        B = CUDSS.CudssMatrix(b)
        try
            CUDSS.cudss("solve", f.solver, X, B; asynchronous=false)
            devsync()
        finally
            finalize(B)
        end
    finally
        finalize(X)
    end
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
    DEO.checked_cudss_options(solver)
    lock(CUDSS_LOCK) do
        drain_dropped!()
        f.released && throw(ArgumentError("CUDSSSolver: this factorization has been released"))
        length(b) == f.n ||
            throw(DimensionMismatch("CUDSSSolver: right-hand side of length $(length(b)) for a factor of size $(f.n)"))
        on_device(f.ctx) do
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

release_factor!(::DEO.CUDSSSolver, factor::CudssFactor) =
    lock(() -> (drain_dropped!(); release_cudss!(factor)), CUDSS_LOCK)

end # module
