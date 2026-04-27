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
  3. `~/github/NLopt-MMA-CCSA/build/libmmaccsa.<ext>`  (engaging convention)
"""
module MmaccsaBackend

using Libdl

export MmaState, minimize!, save_state, load_state, free_state!,
       library_path, library_available

const NLOPT_LD_LBFGS = Cint(11)

const _libpath = Ref{String}("")

function library_path()::String
    if isempty(_libpath[])
        ext = "." * Libdl.dlext
        candidates = String[]
        env = get(ENV, "JULIA_NLOPT_MMACCSA_LIB", "")
        !isempty(env) && push!(candidates, env)
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

const OBJ_CFN = Ref{Ptr{Cvoid}}(C_NULL)
const CON_CFN = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    OBJ_CFN[] = @cfunction(_objective_trampoline, Cdouble,
                           (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    CON_CFN[] = @cfunction(_constraint_trampoline, Cdouble,
                           (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
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
  reflects the final iteration.
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
                   sigma_init::Union{Nothing,Vector{Float64}} = nothing)

    n = length(p)
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
                     (Cint, Cuint), NLOPT_LD_LBFGS, m)

    state_ref = state === nothing ? Ref{Ptr{Cvoid}}(C_NULL) : state.ref
    minf = Ref{Cdouble}(0.0)

    sigma_buf = sigma_init === nothing ? Float64[] : copy(sigma_init)
    sigma_ptr = isempty(sigma_buf) ? Ptr{Cdouble}(C_NULL) : pointer(sigma_buf)
    if sigma_init !== nothing
        length(sigma_init) == n || throw(ArgumentError("sigma_init length $(length(sigma_init)) != p length $n"))
    end

    ret = try
        GC.@preserve obj_box con_boxes con_tols nlopt_constraints lb ub p stop nevals force_stop sigma_buf begin
            if algorithm === :mma
                ccall((:mma_minimize, library_path()), Cint,
                    (Cuint, Ptr{Cvoid}, Ptr{Cvoid},
                     Cuint, Ptr{nlopt_constraint},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ref{nlopt_stopping},
                     Ptr{Cvoid}, Cint, Cuint, Cdouble, Ptr{Cdouble}, Ref{Ptr{Cvoid}}),
                    Cuint(n), OBJ_CFN[], pointer_from_objref(obj_box),
                    m, fc_ptr,
                    pointer(lb), pointer(ub),
                    pointer(p), minf,
                    stop,
                    dual_opt, Cint(inner_maxeval), Cuint(verbose), rho_init,
                    sigma_ptr, state_ref)
            elseif algorithm === :ccsaq
                ccall((:ccsa_quadratic_minimize, library_path()), Cint,
                    (Cuint, Ptr{Cvoid}, Ptr{Cvoid},
                     Cuint, Ptr{nlopt_constraint},
                     Ptr{Cvoid},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble},
                     Ref{nlopt_stopping},
                     Ptr{Cvoid}, Cint, Cuint, Cdouble, Ptr{Cdouble}, Ref{Ptr{Cvoid}}),
                    Cuint(n), OBJ_CFN[], pointer_from_objref(obj_box),
                    m, fc_ptr,
                    C_NULL,
                    pointer(lb), pointer(ub),
                    pointer(p), minf,
                    stop,
                    dual_opt, Cint(inner_maxeval), Cuint(verbose), rho_init,
                    sigma_ptr, state_ref)
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
