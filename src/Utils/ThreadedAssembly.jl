# ═══════════════════════════════════════════════════════════════════════════════
# Threaded cell loops and split Maxwell assembly
# ═══════════════════════════════════════════════════════════════════════════════
#
# One evaluation (objective and gradient) is dozens of Gridap cell loops: integrals,
# vector assemblies (adjoint sources, sensitivities) and the Maxwell matrix. Gridap
# runs each of them on one thread. This file makes them use every Julia thread, and
# stops reassembling the part of the Maxwell matrix that does not depend on the design.
#
# 1. Threaded loops. `sum` over a lazy cell array (every `sum(∫(...)dΩ)`) and the
#    numeric loop of `assemble_vector` are split into contiguous chunks of cells, one
#    task per chunk. Each task has its own Gridap caches and its own output; outputs
#    are added at the end, also in parallel. Buffers are kept between calls.
#    The methods are more specific than Gridap's own (they dispatch on Gridap types),
#    and defer to Gridap whenever threading is off or a term is not a plain
#    cell-vector/cell-ids pair.
#
# 2. Split assembly. A(pt) = A_bg + A_d(pt), where only
#        A_d(pt) = -k² ∫ v·(ε_m(pt) u) dΩ_design
#    depends on the design. A_bg is obtained once per (simulation, physical
#    parameters) as A(pt₀) − A_d(pt₀) from the ordinary assembly, so it holds every
#    design-independent term of whichever formulation is in use (legacy, Bloch, PML)
#    without restating it here. Afterwards an evaluation computes the element matrices
#    of A_d (threaded) and gathers them straight into a copy of A_bg through an index
#    map built once. No thread holds a copy of the matrix. The result has A_bg's
#    sparsity pattern on every evaluation, so a solver's symbolic analysis stays valid.
#    The second evaluation of each plan is checked against the ordinary assembly.
#
# Results differ from the serial ones only by the order of floating-point additions.
# Switches: THREADED[], SPLIT_ASSEMBLY[]; `clear_assembly_caches!()` frees the memory.

import Gridap.Arrays: LazyArray, array_cache, getindex!, testitem, lazy_sum, return_cache, AddEntriesMap
import Gridap.FESpaces: GenericSparseMatrixAssembler, SparseMatrixAssembler, get_assembly_strategy,
    map_cell_rows, map_cell_cols, _numeric_loop_vector!, collect_cell_matrix, get_fe_basis,
    get_trial_fe_basis

"Threaded cell loops (integrals, vector assembly, design-term matrices) when Julia has more than one thread."
const THREADED = Ref(true)
"Cell loops shorter than this run serially."
const THREAD_MIN_CELLS = Ref(2000)
"Number of chunks per loop; 0 means one per Julia thread. Tests set it to exercise chunking on one thread."
const THREAD_CHUNKS = Ref(0)
"Assemble only the design-dependent part of the Maxwell matrix on each evaluation."
const SPLIT_ASSEMBLY = Ref(true)

thread_chunks() = THREAD_CHUNKS[] > 0 ? THREAD_CHUNKS[] : Threads.nthreads()
use_threads(n::Integer) = THREADED[] && thread_chunks() > 1 && n >= THREAD_MIN_CELLS[]

"Contiguous ranges covering 1:n, at most `k` of them, none empty."
function chunk_ranges(n::Integer, k::Integer=thread_chunks())
    k = max(1, min(k, n))
    return [(1+div((i - 1) * n, k)):div(i * n, k) for i in 1:k]
end

"Run `f(i, range)` for every chunk, each on its own task. Safe to call from inside another task."
function foreach_chunk(f::F, ranges) where {F}
    length(ranges) == 1 && return f(1, ranges[1])
    @sync for i in eachindex(ranges)
        Threads.@spawn f(i, ranges[i])
    end
    return nothing
end

# ── Reused buffers ───────────────────────────────────────────────────────────

const _BUFFER_LOCK = ReentrantLock()
const _BUFFERS = Dict{Tuple{DataType,Int},Vector{Any}}()

"`k` vectors of element type `T` and length `n`, contents undefined."
function take_buffers(::Type{T}, n::Int, k::Int) where {T}
    out = Vector{Vector{T}}(undef, k)
    lock(_BUFFER_LOCK) do
        pool = get!(() -> Any[], _BUFFERS, (T, n))
        for i in 1:k
            out[i] = isempty(pool) ? Vector{T}(undef, n) : pop!(pool)::Vector{T}
        end
    end
    return out
end

function give_buffers(bufs::Vector{Vector{T}}) where {T}
    isempty(bufs) && return nothing
    lock(_BUFFER_LOCK) do
        pool = get!(() -> Any[], _BUFFERS, (T, length(bufs[1])))
        append!(pool, bufs)
    end
    return nothing
end

"b[j] += Σ_k parts[k][j], in parallel over j."
function add_parts!(b::Vector, parts::Vector{<:Vector})
    foreach_chunk(chunk_ranges(length(b))) do _, r
        @inbounds for p in parts
            @simd for j in r
                b[j] += p[j]
            end
        end
    end
    return b
end

# ── Integrals ────────────────────────────────────────────────────────────────

# The type of a Gridap cache is not inferable where it is created, so every loop below takes its
# caches as arguments: the loop then compiles for their concrete types and does not allocate.
_sum_chunk(a::LazyArray{G,T,1}, cells) where {G,T} = _sum_cells(array_cache(a), a, cells, zero(T))

function _sum_cells(cache, a, cells, s::T) where {T}
    for i in cells
        s += getindex!(cache, a, i)::T
    end
    return s
end

function Base.sum(a::LazyArray{G,T,1}) where {G,T<:Number}
    n = length(a)
    (use_threads(n) && isconcretetype(T)) || return lazy_sum(array_cache(a), a)
    ranges = chunk_ranges(n)
    parts = Vector{T}(undef, length(ranges))
    foreach_chunk(ranges) do i, r
        parts[i] = _sum_chunk(a, r)
    end
    return sum(parts)
end

# ── Vector assembly ──────────────────────────────────────────────────────────

# Each chunk runs in its own function, so no cache can be shared between tasks through a closure.
function _vector_chunk!(buf::Vector, cellvec, cellids, cells)
    fill!(buf, zero(eltype(buf)))
    return _vector_cells!(buf, array_cache(cellvec), array_cache(cellids), cellvec, cellids, cells)
end

function _vector_cells!(buf::Vector, vals_cache, rows_cache, cellvec, cellids, cells)
    for cell in cells
        rows = getindex!(rows_cache, cellids, cell)
        vals = getindex!(vals_cache, cellvec, cell)
        @inbounds for i in eachindex(rows)
            r = rows[i]
            r > 0 && (buf[r] += vals[i])
        end
    end
    return buf
end

function _serial_vector_term!(b, cellvec, cellids)
    rows_cache = array_cache(cellids)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache, cellvec, 1)
    rows1 = getindex!(rows_cache, cellids, 1)
    add_cache = return_cache(AddEntriesMap(+), b, vals1, rows1)
    _numeric_loop_vector!(b, (add_cache, vals_cache, rows_cache), cellvec, cellids)
    return b
end

_plain_vector_term(cellvec, cellids) =
    testitem(cellvec) isa AbstractVector{<:Number} && testitem(cellids) isa AbstractVector{<:Integer}

function Gridap.FESpaces.numeric_loop_vector!(b::Vector{T}, a::GenericSparseMatrixAssembler,
    vecdata) where {T<:Number}
    if !(THREADED[] && thread_chunks() > 1)
        return invoke(Gridap.FESpaces.numeric_loop_vector!, Tuple{Any,SparseMatrixAssembler,Any}, b, a, vecdata)
    end
    strategy = get_assembly_strategy(a)
    for (cellvec, _cellids) in zip(vecdata...)
        cellids = map_cell_rows(strategy, _cellids)
        n = length(cellvec)
        n > 0 || continue
        if !use_threads(n) || !_plain_vector_term(cellvec, cellids)
            _serial_vector_term!(b, cellvec, cellids)
            continue
        end
        ranges = chunk_ranges(n)
        bufs = take_buffers(T, length(b), length(ranges))
        foreach_chunk(ranges) do i, r
            _vector_chunk!(bufs[i], cellvec, cellids, r)
        end
        add_parts!(b, bufs)
        give_buffers(bufs)
    end
    return b
end

# ── Split Maxwell assembly ───────────────────────────────────────────────────

"""
Everything needed to assemble the Maxwell matrix of one (simulation, physical
parameters) pair from its design term alone. See the top of this file.
"""
mutable struct SplitMaxwellPlan
    sim::Any                                 # held so the simulation cannot be collected and its identity reused
    key::Tuple
    A_bg::SparseMatrixCSC{ComplexF64,Int}    # design-independent part
    nloc::Int                                # entries of one element matrix
    ptr::Vector{Int}                         # contributions to nzval[p] are E[src[ptr[p]:ptr[p+1]-1]]
    src::Union{Vector{Int32},Vector{Int}}    # linear indices into E (32-bit whenever they fit)
    E::Matrix{ComplexF64}                    # nloc × (design cells): element matrices of the design term
    ncalls::Int
    checked::Bool
end

const _PLAN_LOCK = ReentrantLock()
const _PLANS = SplitMaxwellPlan[]
const MAX_SPLIT_PLANS = Ref(8)

"Free the buffers of the threaded loops and every split-assembly plan."
function clear_assembly_caches!()
    lock(_BUFFER_LOCK) do
        empty!(_BUFFERS)
    end
    lock(_PLAN_LOCK) do
        empty!(_PLANS)
    end
    return nothing
end

# Every field of the physical parameters except the polarization, which only selects the source.
_phys_key(phys) = Tuple(getfield(phys, f) for f in fieldnames(typeof(phys)) if f !== :polarization)

function _find_plan(sim, key)
    lock(_PLAN_LOCK) do
        for p in _PLANS
            p.sim === sim && isequal(p.key, key) && return p
        end
        return nothing
    end
end

function _store_plan!(plan::SplitMaxwellPlan)
    lock(_PLAN_LOCK) do
        length(_PLANS) >= MAX_SPLIT_PLANS[] && popfirst!(_PLANS)
        push!(_PLANS, plan)
    end
    return plan
end

"Cell matrices, row ids and column ids of the design term −k² ∫ v·(ε_m(pt) u) dΩ_design."
function _design_term_data(pt, sim, phys)
    εₘ = (p -> ε_design_wf(p, phys)) ∘ pt
    k = phys.ω
    u = get_trial_fe_basis(sim.U)
    v = get_fe_basis(sim.V)
    matdata = collect_cell_matrix(sim.U, sim.V, -(k^2) * ∫(v ⋅ (εₘ * u))sim.dΩ_design)
    length(matdata[1]) == 1 || return nothing
    strategy = get_assembly_strategy(SparseMatrixAssembler(sim.U, sim.V))
    cellmat = matdata[1][1]
    rows = map_cell_rows(strategy, matdata[2][1])
    cols = map_cell_cols(strategy, matdata[3][1])
    (length(cellmat) > 0 && testitem(cellmat) isa AbstractMatrix{<:Number} &&
     testitem(rows) isa AbstractVector{<:Integer} && testitem(cols) isa AbstractVector{<:Integer}) || return nothing
    return cellmat, rows, cols
end

_element_chunk!(E::Matrix{ComplexF64}, cellmat, cells) = _element_cells!(E, array_cache(cellmat), cellmat, cells)

function _element_cells!(E::Matrix{ComplexF64}, cache, cellmat, cells)
    nloc = size(E, 1)
    for cell in cells
        m = getindex!(cache, cellmat, cell)
        length(m) == nloc || error("split assembly: element matrices of different sizes")
        @inbounds for i in 1:nloc
            E[i, cell] = m[i]
        end
    end
    return nothing
end

"Fill `E` with the element matrices of `cellmat`, one column per cell."
function _element_matrices!(E::Matrix{ComplexF64}, cellmat)
    n = length(cellmat)
    ranges = use_threads(n) ? chunk_ranges(n) : [1:n]
    foreach_chunk(ranges) do _, r
        _element_chunk!(E, cellmat, r)
    end
    return E
end

# Position of entry (row, col) in A's stored values; 0 if it is not stored.
@inline function _nz_position(A::SparseMatrixCSC, row::Int, col::Int)
    lo = A.colptr[col]
    hi = A.colptr[col+1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        r = A.rowval[mid]
        if r < row
            lo = mid + 1
        elseif r > row
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

_positions_chunk!(where_::Matrix{Int}, A, rows, cols, cells) =
    _positions_cells!(where_, A, array_cache(rows), array_cache(cols), rows, cols, cells)

function _positions_cells!(where_::Matrix{Int}, A, rows_cache, cols_cache, rows, cols, cells)
    for cell in cells
        rs = getindex!(rows_cache, rows, cell)
        cs = getindex!(cols_cache, cols, cell)
        nr = length(rs)
        length(rs) * length(cs) == size(where_, 1) || error("split assembly: cells with different numbers of unknowns")
        @inbounds for j in eachindex(cs), i in eachindex(rs)
            r = Int(rs[i])
            c = Int(cs[j])
            p = (r > 0 && c > 0) ? _nz_position(A, r, c) : 0
            (r > 0 && c > 0 && p == 0) && error("split assembly: a design-term entry is outside the matrix pattern")
            where_[(j-1)*nr+i, cell] = p
        end
    end
    return nothing
end

"Gather map: for every stored entry of `A`, the element-matrix entries that add into it."
function _gather_map(A::SparseMatrixCSC, rows, cols, nloc::Int)
    ncell = length(rows)
    where_ = Matrix{Int}(undef, nloc, ncell)
    ranges = use_threads(ncell) ? chunk_ranges(ncell) : [1:ncell]
    foreach_chunk(ranges) do _, r
        _positions_chunk!(where_, A, rows, cols, r)
    end
    ptr = zeros(Int, nnz(A) + 1)
    @inbounds for p in where_
        p > 0 && (ptr[p+1] += 1)
    end
    ptr[1] = 1
    @inbounds for p in 1:nnz(A)
        ptr[p+1] += ptr[p]
    end
    I = length(where_) <= typemax(Int32) ? Int32 : Int
    src = Vector{I}(undef, ptr[end] - 1)
    next = copy(ptr)
    @inbounds for lin in eachindex(where_)
        p = where_[lin]
        p > 0 || continue
        src[next[p]] = lin
        next[p] += 1
    end
    return ptr, src
end

function _gather_chunk!(nz, bg, ptr, src, E, sgn, positions)
    @inbounds for p in positions
        s = zero(ComplexF64)
        for j in ptr[p]:(ptr[p+1]-1)
            s += E[src[j]]
        end
        nz[p] = bg[p] + sgn * s
    end
    return nothing
end

"nz = bg + sgn·(gathered element matrices), in parallel over the stored entries."
function _gather!(nz::Vector{ComplexF64}, bg::Vector{ComplexF64}, plan_ptr, plan_src, E, sgn)
    n = length(nz)
    ranges = use_threads(n) ? chunk_ranges(n) : [1:n]
    foreach_chunk(ranges) do _, r
        _gather_chunk!(nz, bg, plan_ptr, plan_src, E, sgn, r)
    end
    return nz
end

function _parallel_copy(v::Vector)
    out = similar(v)
    n = length(v)
    ranges = use_threads(n) ? chunk_ranges(n) : [1:n]
    foreach_chunk(ranges) do _, r
        copyto!(out, first(r), v, first(r), length(r))
    end
    return out
end

"Whether `assemble_maxwell` may use the split assembly for these parameters."
split_assembly_applies(sim, phys) = SPLIT_ASSEMBLY[] && hasproperty(sim, :dΩ_design)

"""
    assemble_maxwell_split(pt, sim, phys, direct) -> SparseMatrixCSC or nothing

Maxwell matrix from the cached design-independent part plus the design term.
`direct(pt, sim, phys)` is the ordinary assembly; it is called on the first
evaluation of a (simulation, parameters) pair, to build the plan, and once more on
the second, to check the split result. Returns `nothing` when the design term has a
shape this path does not handle, and the caller then assembles directly.
"""
function assemble_maxwell_split(pt, sim, phys, direct::F) where {F}
    key = _phys_key(phys)
    plan = _find_plan(sim, key)
    data = _design_term_data(pt, sim, phys)
    data === nothing && return nothing
    cellmat, rows, cols = data
    if plan === nothing
        A = direct(pt, sim, phys)
        A isa SparseMatrixCSC{ComplexF64,Int} || return A
        nloc = length(testitem(cellmat))
        ptr, src = _gather_map(A, rows, cols, nloc)
        E = Matrix{ComplexF64}(undef, nloc, length(cellmat))
        _element_matrices!(E, cellmat)
        bg = similar(A.nzval)
        _gather!(bg, A.nzval, ptr, src, E, -1.0)          # A_bg = A(pt₀) − A_d(pt₀)
        A_bg = SparseMatrixCSC(size(A, 1), size(A, 2), copy(A.colptr), copy(A.rowval), bg)
        _store_plan!(SplitMaxwellPlan(sim, key, A_bg, nloc, ptr, src, E, 1, false))
        return A
    end
    size(plan.E, 2) == length(cellmat) || error("split assembly: the design region changed under a cached plan")
    _element_matrices!(plan.E, cellmat)
    bg = plan.A_bg
    nz = Vector{ComplexF64}(undef, nnz(bg))
    _gather!(nz, bg.nzval, plan.ptr, plan.src, plan.E, 1.0)
    A = SparseMatrixCSC(size(bg, 1), size(bg, 2), _parallel_copy(bg.colptr), _parallel_copy(bg.rowval), nz)
    plan.ncalls += 1
    if !plan.checked
        ref = direct(pt, sim, phys)
        same = ref.colptr == A.colptr && ref.rowval == A.rowval
        err = same ? norm(ref.nzval - A.nzval) / norm(ref.nzval) : Inf
        if !(err <= 1e-12)
            lock(_PLAN_LOCK) do
                filter!(p -> p !== plan, _PLANS)
            end
            SPLIT_ASSEMBLY[] = false
            log_warn(:assembly, "split Maxwell assembly disagrees with the ordinary assembly; split assembly is now off",
                rel_error=err, same_pattern=same)
            return ref
        end
        plan.checked = true
    end
    return A
end
