import SparseArrays: SparseMatrixCSC

"""
    CUDSSSolver

Optional sparse direct solver on an NVIDIA GPU, using NVIDIA cuDSS through CUDSS.jl (1.0 or
later). The implementation is in ext/CUDSSExt.jl and loads with `using CUDSS`; it needs a
CUDA-capable GPU.

Each Maxwell factorization is decided on the full matrix, as for `PardisoSolver`. A complex
symmetric matrix (normal incidence) is factored as LDLᵀ from its lower triangle (cuDSS
structure "S", which for complex values is the transpose, not the Hermitian, factorization);
any other matrix (oblique Bloch fields) is factored as a general LDU (structure "G"). The
analysis is reused while the sparsity pattern and the structure stay the same, so each design
update pays one numerical factorization on the GPU.

Adjoints are exact for both structures: for "S", Aᴴx = b is solved as A conj(x) = conj(b);
for "G", cuDSS solves with the conjugate transpose of the factored matrix.

Every solve is checked against the operator it is meant to invert, A or Aᴴ, built from the full
matrix (so a matrix accepted as symmetric only to `SYMMETRY_RTOL` still gets its own adjoint):
the residual is computed on the GPU in double precision, and if it exceeds `residual_rtol`
(relative to ‖b‖) up to `refine_steps` steps of iterative refinement are taken before an error is
thrown. cuDSS work is serialized process-wide, so factors may be used from any task.

`precision = :single` stores the values of "S" factors in single precision (the factor values
take half the memory; the double matrix kept for residuals, indices and workspace do not) and
refines each solve in double precision toward the tighter of `refine_rtol` (default 1e-11) and
`residual_rtol`; every returned answer meets `residual_rtol` (with `residual_rtol = 0`, the
refinement target) or the solve throws. Convergence needs the matrix to be well enough
conditioned for a single-precision factor; "G" factors stay in double precision.
"""
Base.@kwdef struct CUDSSSolver <: AbstractSolver
    # `assume_symmetric_maxwell` : `nothing` (default) picks LDLᵀ ("S") for each factorization
    #                         whose matrix is verified complex symmetric (`symmetry_defect(A) ≤
    #                         SYMMETRY_RTOL`) and general LDU ("G") otherwise; `true` forces "S"
    #                         (refused for a matrix that is not symmetric), `false` forces "G"
    # `reuse_analysis`      : Reuse the cuDSS analysis while the sparsity pattern is unchanged
    # `precision`           : `:double` (default) or `:single`. `:single` factors symmetric ("S")
    #                         matrices in single precision and refines every solve in double
    #                         precision against the full matrix; general ("G") matrices stay in
    #                         double (cuDSS's single-precision general factor is not reliable here)
    # `residual_rtol`       : Largest accepted ‖b − op(A)x‖/‖b‖ after a solve; 0 disables the check
    #                         (for `:double`; a single-precision factor is always refined)
    # `refine_rtol`         : Refinement continues while the residual exceeds this; `nothing` means
    #                         `residual_rtol` for a double factor and 1e-11 for a single one
    # `refine_steps`        : Refinement steps allowed per solve; `nothing` means 3 for a double
    #                         factor and 20 for a single one
    assume_symmetric_maxwell::Union{Nothing,Bool} = nothing
    reuse_analysis::Bool = true
    precision::Symbol = :double
    residual_rtol::Float64 = 1e-9
    refine_rtol::Union{Nothing,Float64} = nothing
    refine_steps::Union{Nothing,Int} = nothing
end

"""
    cudss_storage_type(solver, T, structure) -> element type of the factor

`T` for `precision = :double`. For `:single`, the single-precision type of `T` when the structure
is "S", and `T` for "G". An unknown `precision` is refused.
"""
function cudss_storage_type(solver::CUDSSSolver, ::Type{T}, structure::AbstractString) where T
    solver.precision in (:double, :single) ||
        throw(ArgumentError("CUDSSSolver: precision must be :double or :single, got :$(solver.precision)"))
    (solver.precision === :single && structure == "S") || return T
    return T <: Complex ? ComplexF32 : Float32
end

"Refinement target and step limit for a factor of element type `Ts` solving systems of type `T`."
function cudss_refinement(solver::CUDSSSolver, ::Type{T}, ::Type{Ts}) where {T,Ts}
    single = Ts !== T
    rtol = something(solver.refine_rtol, single ? 1e-11 : solver.residual_rtol)
    steps = something(solver.refine_steps, single ? 20 : 3)
    return rtol, steps
end

"""
    checked_cudss_structure(solver, A) -> "S" or "G"

The cuDSS structure for one Maxwell factorization, decided on the full matrix before "S"
discards the upper triangle. A forced "S" is refused for a matrix that is not symmetric.
"""
function checked_cudss_structure(solver::CUDSSSolver, A)
    LinearAlgebra.checksquare(A)
    choice = solver.assume_symmetric_maxwell
    choice === false && return "G"
    defect = symmetry_defect(A)
    if choice === true
        defect <= SYMMETRY_RTOL ||
            throw(ArgumentError("CUDSSSolver: assume_symmetric_maxwell=true, but the Maxwell matrix is not complex symmetric (‖A − Aᵀ‖/‖A‖ = $defect); oblique Bloch fields need the general structure, which the default assume_symmetric_maxwell=nothing picks for them"))
        return "S"
    end
    return defect <= SYMMETRY_RTOL ? "S" : "G"
end

"Validate a CUDSSSolver's options (finite residual_rtol ≥ 0; refine_steps ≥ 0 and refine_rtol > 0 when given; precision); returns the solver."
function checked_cudss_options(solver::CUDSSSolver)
    (isfinite(solver.residual_rtol) && solver.residual_rtol >= 0) ||
        throw(ArgumentError("CUDSSSolver: residual_rtol must be finite and ≥ 0 (0 disables the check), got $(solver.residual_rtol)"))
    (solver.refine_steps === nothing || solver.refine_steps >= 0) ||
        throw(ArgumentError("CUDSSSolver: refine_steps must be ≥ 0 or nothing, got $(solver.refine_steps)"))
    (solver.refine_rtol === nothing || (isfinite(solver.refine_rtol) && solver.refine_rtol > 0)) ||
        throw(ArgumentError("CUDSSSolver: refine_rtol must be finite and > 0, or nothing, got $(solver.refine_rtol)"))
    solver.precision in (:double, :single) ||
        throw(ArgumentError("CUDSSSolver: precision must be :double or :single, got :$(solver.precision)"))
    return solver
end

"Real filter matrices: LDLᵀ when symmetric, general LDU otherwise."
checked_cudss_filter_structure(A) = (LinearAlgebra.checksquare(A); symmetry_defect(A) <= SYMMETRY_RTOL ? "S" : "G")

# Stub implementations that error if the extension is not loaded
const CUDSS_NOT_LOADED = "CUDSS extension not loaded. Install CUDSS.jl 1.0 or later (using Pkg; Pkg.add(\"CUDSS\")) and load it with `using CUDSS`; CUDSSSolver needs an NVIDIA GPU, so use PardisoSolver() or UmfpackSolver() without one"

lu!(::CUDSSSolver, A::SparseMatrixCSC) = error(CUDSS_NOT_LOADED)
lu!(solver::CUDSSSolver, factor, A::SparseMatrixCSC) = lu!(solver, A)
filter_lu!(::CUDSSSolver, A::SparseMatrixCSC) = error(CUDSS_NOT_LOADED)
filter_lu!(solver::CUDSSSolver, factor, A::SparseMatrixCSC) = filter_lu!(solver, A)
solve!(::CUDSSSolver, factor, b::Vector) = error(CUDSS_NOT_LOADED)
solve_adjoint!(::CUDSSSolver, factor, b::Vector) = error(CUDSS_NOT_LOADED)
release_factor!(::CUDSSSolver, factor) = nothing
