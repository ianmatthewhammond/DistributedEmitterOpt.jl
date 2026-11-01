"""
    MmaccsaBackend

Thin Julia wrapper over the standalone NLopt-MMA-CCSA shared library
(`libmmaccsa.{dylib,so}`) — the project at github.com/ianmatthewhammond/NLopt-MMA-CCSA.

Provides `minimize!` and `MmaState` as the optimizer surface, plus
`save_state` / `load_state` for process-portable checkpoints. All sign and
constraint conventions match the standalone library: minimization, with
inequality constraints `g(p) ≤ 0`.

Library resolution order:
  1. `ENV["JULIA_NLOPT_MMACCSA_LIB"]`
  2. `~/GitHub/NLopt-MMA-CCSA/build/libmmaccsa.<ext>`
  3. `~/github/NLopt-MMA-CCSA/build/libmmaccsa.<ext>`  (lowercase checkout directory)
"""
module MmaccsaBackend

using Libdl

export MmaState, minimize!, save_state, load_state, free_state!,
       library_path, library_available, state_header, state_dims,
       dual_algorithm_supported

const NLOPT_LD_LBFGS = Cint(11)
const NLOPT_LD_MMA = Cint(24)
const NLOPT_INVALID_ARGS = Cint(-2)

"""
Dual-subproblem solvers understood by the library. `:mma` is what upstream NLopt uses
(its default `dual_algorithm`); `:lbfgs` is Luksan PLIS, which can fail with
`NLOPT_FAILURE` when its line search stalls on a converged but badly conditioned dual.
"""
const DUAL_ALGORITHMS = (mma=NLOPT_LD_MMA, lbfgs=NLOPT_LD_LBFGS)

function _dual_code(dual_algorithm::Symbol)
    haskey(DUAL_ALGORITHMS, dual_algorithm) ||
        throw(ArgumentError("dual_algorithm must be :mma or :lbfgs (got :$dual_algorithm)"))
    return DUAL_ALGORITHMS[dual_algorithm]
end

const _libpath = Ref{String}("")

function library_path()::String
    if isempty(_libpath[])
        ext = "." * Libdl.dlext
        candidates = String[]
        env = get(ENV, "JULIA_NLOPT_MMACCSA_LIB", "")
        if !isempty(env)
            isfile(env) || error("JULIA_NLOPT_MMACCSA_LIB does not name an existing shared library: $env. " *
                                 "Build NLopt-MMA-CCSA and set this variable to libmmaccsa.")
            _libpath[] = env
            return env
        end
        push!(candidates, joinpath(homedir(), "GitHub", "NLopt-MMA-CCSA", "build", "libmmaccsa" * ext))
        push!(candidates, joinpath(homedir(), "github", "NLopt-MMA-CCSA", "build", "libmmaccsa" * ext))
        for c in candidates
            if isfile(c)
                _libpath[] = c
                return c
            end
        end
        tried = join(candidates, ", ")
        error("libmmaccsa not found. Build NLopt-MMA-CCSA or set JULIA_NLOPT_MMACCSA_LIB. Tried: $tried")
    end
    return _libpath[]
end

function library_available()::Bool
    try
        library_path()
        return true
    catch
        return false
    end
end

# ---------------------------------------------------------------------------
# C structs (must match src/nlopt.h in NLopt-MMA-CCSA)
# ---------------------------------------------------------------------------

struct nlopt_stopping
    n::Cuint
    minf_max::Cdouble
    ftol_rel::Cdouble
    ftol_abs::Cdouble
    xtol_rel::Cdouble
    xtol_abs::Ptr{Cdouble}
    x_weights::Ptr{Cdouble}
    nevals_p::Ptr{Cint}
    maxeval::Cint
    maxtime::Cdouble
    start::Cdouble
    force_stop::Ptr{Cint}
    stop_msg::Ptr{Ptr{Cchar}}
end

struct nlopt_constraint
    m::Cuint
    f::Ptr{Cvoid}
    mf::Ptr{Cvoid}
    pre::Ptr{Cvoid}
    f_data::Ptr{Cvoid}
    tol::Ptr{Cdouble}
end

# ---------------------------------------------------------------------------
# Trampolines: turn Julia closures into C function pointers
# ---------------------------------------------------------------------------

# The C side calls these; they unwrap the Julia callable from f_data.

mutable struct ObjectiveBox
    f::Function   # signature: (x::Vector{Float64}, grad::Vector{Float64}) -> Float64
end

mutable struct ConstraintBox
    g::Function   # signature: (x::Vector{Float64}, grad::Vector{Float64}) -> Float64
end

mutable struct CheckpointBox
    callback::Function # signature: (x::Vector{Float64}, minf::Float64, nevals::Int, k::Int) -> nothing
end

function _objective_trampoline(n::Cuint, x_ptr::Ptr{Cdouble}, grad_ptr::Ptr{Cdouble},
                               data::Ptr{Cvoid})::Cdouble
    box = unsafe_pointer_to_objref(data)::ObjectiveBox
    x = unsafe_wrap(Array, x_ptr, n)
    grad = grad_ptr == C_NULL ? Float64[] : unsafe_wrap(Array, grad_ptr, n)
    return box.f(x, grad)::Float64
end

function _constraint_trampoline(n::Cuint, x_ptr::Ptr{Cdouble}, grad_ptr::Ptr{Cdouble},
                                data::Ptr{Cvoid})::Cdouble
    box = unsafe_pointer_to_objref(data)::ConstraintBox
    x = unsafe_wrap(Array, x_ptr, n)
    grad = grad_ptr == C_NULL ? Float64[] : unsafe_wrap(Array, grad_ptr, n)
    return box.g(x, grad)::Float64
end

function _checkpoint_trampoline(n::Cuint, x_ptr::Ptr{Cdouble}, minf::Cdouble,
                                nevals::Cint, k::Cuint, data::Ptr{Cvoid})::Cvoid
    data == C_NULL && return nothing
    box = unsafe_pointer_to_objref(data)::CheckpointBox
    x = unsafe_wrap(Array, x_ptr, n)
    box.callback(x, Float64(minf), Int(nevals), Int(k))
    return nothing
end

const OBJ_CFN = Ref{Ptr{Cvoid}}(C_NULL)
const CON_CFN = Ref{Ptr{Cvoid}}(C_NULL)
const CHECKPOINT_CFN = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    OBJ_CFN[] = @cfunction(_objective_trampoline, Cdouble,
                           (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    CON_CFN[] = @cfunction(_constraint_trampoline, Cdouble,
                           (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    CHECKPOINT_CFN[] = @cfunction(_checkpoint_trampoline, Cvoid,
                                  (Cuint, Ptr{Cdouble}, Cdouble, Cint, Cuint, Ptr{Cvoid}))
end

# ---------------------------------------------------------------------------
# Checkpoint state handle
# ---------------------------------------------------------------------------

mutable struct MmaState
    ref::Base.RefValue{Ptr{Cvoid}}
    function MmaState()
        st = new(Ref{Ptr{Cvoid}}(C_NULL))
        finalizer(free_state!, st)
        return st
    end
end

function free_state!(st::MmaState)
    if st.ref[] != C_NULL
        ccall((:mma_state_free, library_path()), Cvoid, (Ptr{Cvoid},), st.ref[])
        st.ref[] = C_NULL
    end
    return nothing
end

function save_state(path::AbstractString, st::MmaState)
    st.ref[] == C_NULL && error("MmaState: nothing to save (state is null)")
    nbytes = Ref{Csize_t}(0)
    ret = ccall((:mma_state_serialized_size, library_path()), Cint,
                (Ptr{Cvoid}, Ref{Csize_t}), st.ref[], nbytes)
    ret < 0 && error("mma_state_serialized_size failed: $ret")
    bytes = Vector{UInt8}(undef, Int(nbytes[]))
    ret = ccall((:mma_state_serialize, library_path()), Cint,
                (Ptr{Cvoid}, Ptr{UInt8}, Csize_t),
                st.ref[], bytes, nbytes[])
    ret < 0 && error("mma_state_serialize failed: $ret")
    open(io -> write(io, bytes), path, "w")
    return path
end

function load_state(path::AbstractString)::MmaState
    bytes = read(path)
    st = MmaState()
    ret = ccall((:mma_state_deserialize, library_path()), Cint,
                (Ptr{UInt8}, Csize_t, Ref{Ptr{Cvoid}}),
                bytes, length(bytes), st.ref)
    ret < 0 && error("mma_state_deserialize failed: $ret")
    return st
end

# ---------------------------------------------------------------------------
# Serialized-state header (read WITHOUT calling C)
# ---------------------------------------------------------------------------

# Mirror of `mma_state_serial_header` in NLopt-MMA-CCSA src/mma_state.c
# (MMA_STATE_SERIAL_VERSION 1). Julia lays out an isbits struct with the C ABI's
# alignment, so field offsets match on x86_64 and aarch64 (80 bytes).
struct MmaStateSerialHeader
    magic::Cuint
    version::Cuint
    state_magic::Cuint
    algorithm::Cuint
    n::Cuint
    m::Cuint
    workspace_len::Csize_t
    rho::Cdouble
    k::Cuint
    infeasibility::Cdouble
    feasible::Cint
    fcur::Cdouble
    minf::Cdouble
end

const MMA_STATE_SERIAL_MAGIC = 0x4d4d4153
const MMA_STATE_SERIAL_VERSION = 0x00000001

"""
    state_header(path) -> MmaStateSerialHeader

Read the header of a serialized MMA/CCSA state file without handing it to C.

Why this exists: `mma_state_restore` demands that the saved `n` (variables) and
`m` (constraints) equal the current problem's, and on a mismatch the C optimizer
returns a negative code — which `minimize!` passes back WITHOUT throwing. A caller
that does not check `ret` then gets a silent no-op optimization. Read the header
first and refuse the mismatch by name (see `state_dims`).
"""
function state_header(path::AbstractString)::MmaStateSerialHeader
    nb = sizeof(MmaStateSerialHeader)
    bytes = read(path)
    length(bytes) >= nb ||
        throw(ArgumentError("MMA state file $path is $(length(bytes)) bytes, shorter than its $nb-byte header"))
    h = reinterpret(MmaStateSerialHeader, bytes[1:nb])[1]
    h.magic == MMA_STATE_SERIAL_MAGIC ||
        throw(ArgumentError("MMA state file $path has bad magic 0x$(string(h.magic, base=16)) (expected 0x$(string(MMA_STATE_SERIAL_MAGIC, base=16)))"))
    h.version == MMA_STATE_SERIAL_VERSION ||
        throw(ArgumentError("MMA state file $path has serial version $(h.version); this reader knows $(MMA_STATE_SERIAL_VERSION)"))
    return h
end

"""
    state_dims(path) -> (n, m)

Number of variables and constraints a serialized state was written for.
"""
function state_dims(path::AbstractString)
    h = state_header(path)
    return (Int(h.n), Int(h.m))
end

# ---------------------------------------------------------------------------
# Dual-solver capability check
# ---------------------------------------------------------------------------

_probe_dual_objective(n::Cuint, x::Ptr{Cdouble}, grad::Ptr{Cdouble}, ::Ptr{Cvoid})::Cdouble =
    (grad != C_NULL && unsafe_store!(grad, 2 * unsafe_load(x)); unsafe_load(x)^2)

const _dual_support = Dict{Tuple{String,Symbol},Bool}()

"""
    dual_algorithm_supported(dual_algorithm) -> Bool

Whether the loaded library can solve the dual subproblem with `dual_algorithm`.
Builds of the library that predate the MMA dual reject `NLOPT_LD_MMA` with
`NLOPT_INVALID_ARGS`; this asks the library directly, on a one-variable problem,
instead of finding out at the first dual solve of a long run.
"""
function dual_algorithm_supported(dual_algorithm::Symbol)::Bool
    code = _dual_code(dual_algorithm)
    lib = library_path()
    return get!(_dual_support, (lib, dual_algorithm)) do
        opt = ccall((:nlopt_create, library_path()), Ptr{Cvoid}, (Cint, Cuint), code, 1)
        opt == C_NULL && error("nlopt_create failed in $lib")
        try
            f = @cfunction(_probe_dual_objective, Cdouble, (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
            ccall((:nlopt_set_min_objective, library_path()), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), opt, f, C_NULL)
            ccall((:nlopt_set_lower_bounds, library_path()), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), opt, [-1.0])
            ccall((:nlopt_set_upper_bounds, library_path()), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), opt, [1.0])
            x = [0.5]
            minf = Ref{Cdouble}(0.0)
            ret = ccall((:nlopt_optimize_limited, library_path()), Cint,
                        (Ptr{Cvoid}, Ptr{Cdouble}, Ref{Cdouble}, Cint, Cdouble), opt, x, minf, 50, 0.0)
            ret != NLOPT_INVALID_ARGS
        finally
            ccall((:nlopt_destroy, library_path()), Cvoid, (Ptr{Cvoid},), opt)
        end
    end
end

# ---------------------------------------------------------------------------
# Top-level minimize
# ---------------------------------------------------------------------------

"""
    minimize!(algorithm, f, p; kwargs...) -> (ret, fmin, p_opt)

Minimize `f(x, grad) -> Float64` over `p` with bounds `[lb, ub]`.

- `algorithm`: `:mma` or `:ccsaq`.
- `f`: closure returning the objective value; if `grad` is non-empty, must
  fill it in place with ∂f/∂x.
- `p`: initial point (mutated to optimal).
- `lb`, `ub`: bounds vectors of length `length(p)`.
- `maxeval`: hard cap on objective evaluations.
- `ftol_rel`, `ftol_abs`, `xtol_rel`, `minf_max`: standard NLopt stopping
  criteria; `0.0` (or `-Inf` for `minf_max`) disables.
- `constraints`: vector of `(g, tol)` pairs, each enforced as `g(p) ≤ 0`.
- `state`: optional `MmaState` to feed/receive checkpoint data. If the state
  is non-null on entry, the optimizer resumes from it; on exit, the state
  holds the live C workspace. Serialize it inside `checkpoint_callback`, paired
  with the callback's point and evaluation count: a hard evaluation-limit exit
  can leave uncommitted trial data in the workspace after the last callback.
- `checkpoint_callback`: optional callback invoked only after the C optimizer
  has saved a committed checkpoint state.
- `dual_algorithm`: solver for the CCSA dual subproblem, `:mma` (default, as in
  upstream NLopt) or `:lbfgs`. The dual carries no state between outer iterations
  beyond the dual variables, which live in the checkpoint state, so a state saved
  under one dual solver resumes under the other.
"""
function minimize!(algorithm::Symbol, f::Function, p::Vector{Float64};
                   lb::Vector{Float64},
                   ub::Vector{Float64},
                   maxeval::Integer,
                   ftol_rel::Float64 = 0.0,
                   ftol_abs::Float64 = 0.0,
                   xtol_rel::Float64 = 0.0,
                   minf_max::Float64 = -Inf,
                   constraints::Vector = Tuple{Function,Float64}[],
                   state::Union{MmaState,Nothing} = nothing,
                   inner_maxeval::Integer = 0,
                   verbose::Integer = 0,
                   rho_init::Float64 = 1.0,
                   sigma_init::Union{Nothing,Vector{Float64}} = nothing,
                   checkpoint_callback::Union{Nothing,Function} = nothing,
                   dual_algorithm::Symbol = :mma)

    n = length(p)
    dual_code = _dual_code(dual_algorithm)
    dual_algorithm_supported(dual_algorithm) ||
        error("the NLopt-MMA-CCSA library at $(library_path()) does not support dual_algorithm=:$dual_algorithm; " *
              "rebuild it from a version with the MMA dual, or pass dual_algorithm=:lbfgs")
    length(lb) == n || throw(ArgumentError("lb length $(length(lb)) != p length $n"))
    length(ub) == n || throw(ArgumentError("ub length $(length(ub)) != p length $n"))

    obj_box = ObjectiveBox(f)
    con_boxes = ConstraintBox[ConstraintBox(c[1]) for c in constraints]
    con_tols = Float64[c[2] for c in constraints]

    nlopt_constraints = nlopt_constraint[
        nlopt_constraint(Cuint(1), CON_CFN[], C_NULL, C_NULL,
                         pointer_from_objref(con_boxes[i]),
                         pointer(con_tols, i))
        for i in eachindex(con_boxes)
    ]
    fc_ptr = isempty(nlopt_constraints) ? Ptr{nlopt_constraint}(C_NULL) : pointer(nlopt_constraints)
    m = Cuint(length(nlopt_constraints))

    nevals = Ref{Cint}(0)
    force_stop = Ref{Cint}(0)
    stop = Ref(nlopt_stopping(
        Cuint(n), minf_max, ftol_rel, ftol_abs, xtol_rel,
        C_NULL, C_NULL,
        Base.unsafe_convert(Ptr{Cint}, nevals),
        Cint(maxeval), 0.0, 0.0,
        Base.unsafe_convert(Ptr{Cint}, force_stop),
        C_NULL,
    ))

    dual_opt = ccall((:nlopt_create, library_path()), Ptr{Cvoid},
                     (Cint, Cuint), dual_code, m)

    state_ref = state === nothing ? Ref{Ptr{Cvoid}}(C_NULL) : state.ref
    minf = Ref{Cdouble}(0.0)
    checkpoint_box = checkpoint_callback === nothing ? nothing : CheckpointBox(checkpoint_callback)
    checkpoint_ptr = checkpoint_box === nothing ? Ptr{Cvoid}(C_NULL) : pointer_from_objref(checkpoint_box)
    checkpoint_cfn = checkpoint_box === nothing ? Ptr{Cvoid}(C_NULL) : CHECKPOINT_CFN[]

    sigma_buf = sigma_init === nothing ? Float64[] : copy(sigma_init)
    sigma_ptr = isempty(sigma_buf) ? Ptr{Cdouble}(C_NULL) : pointer(sigma_buf)
    if sigma_init !== nothing
        length(sigma_init) == n || throw(ArgumentError("sigma_init length $(length(sigma_init)) != p length $n"))
    end

    ret = try
        GC.@preserve obj_box con_boxes con_tols nlopt_constraints lb ub p stop nevals force_stop sigma_buf checkpoint_box begin
            if algorithm === :mma
                ccall((:mma_minimize, library_path()), Cint,
                    (Cuint, Ptr{Cvoid}, Ptr{Cvoid},
                     Cuint, Ptr{nlopt_constraint},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ref{nlopt_stopping},
                     Ptr{Cvoid}, Cint, Cuint, Cdouble, Ptr{Cdouble}, Ref{Ptr{Cvoid}},
                     Ptr{Cvoid}, Ptr{Cvoid}),
                    Cuint(n), OBJ_CFN[], pointer_from_objref(obj_box),
                    m, fc_ptr,
                    pointer(lb), pointer(ub),
                    pointer(p), minf,
                    stop,
                    dual_opt, Cint(inner_maxeval), Cuint(verbose), rho_init,
                    sigma_ptr, state_ref, checkpoint_cfn, checkpoint_ptr)
            elseif algorithm === :ccsaq
                ccall((:ccsa_quadratic_minimize, library_path()), Cint,
                    (Cuint, Ptr{Cvoid}, Ptr{Cvoid},
                     Cuint, Ptr{nlopt_constraint},
                     Ptr{Cvoid},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ref{nlopt_stopping},
                     Ptr{Cvoid}, Cint, Cuint, Cdouble, Ptr{Cdouble}, Ref{Ptr{Cvoid}},
                     Ptr{Cvoid}, Ptr{Cvoid}),
                    Cuint(n), OBJ_CFN[], pointer_from_objref(obj_box),
                    m, fc_ptr,
                    C_NULL,
                    pointer(lb), pointer(ub),
                    pointer(p), minf,
                    stop,
                    dual_opt, Cint(inner_maxeval), Cuint(verbose), rho_init,
                    sigma_ptr, state_ref, checkpoint_cfn, checkpoint_ptr)
            else
                error("unknown standalone algorithm: $algorithm (use :mma or :ccsaq)")
            end
        end
    finally
        ccall((:nlopt_destroy, library_path()), Cvoid, (Ptr{Cvoid},), dual_opt)
    end

    if state !== nothing
        state.ref[] = state_ref[]
    end

    return Int(ret), minf[], p
end

end # module
