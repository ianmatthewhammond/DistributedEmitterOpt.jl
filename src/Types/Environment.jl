"""
    Environment

Physical environment: material properties for the simulation domain.
Materials can be specified as:
- String: lookup from data files (e.g., "Ag", "Au")
- Float64: direct refractive index (default √1.77 for water-like medium)
"""

const MaterialSpec = Union{String,Float64}

"""
    Environment(; mat_design, mat_substrate=mat_design, mat_fluid=sqrt(1.77))

## Fields
- `mat_design` — Design region material (typically metal)
- `mat_substrate` — Substrate material (defaults to mat_design)
- `mat_fluid` — Fluid/background material (default √1.77)
- `logger_cfg` — Runtime logging config (default off)
"""
Base.@kwdef struct Environment
    mat_design::MaterialSpec
    mat_substrate::MaterialSpec = mat_design
    mat_fluid::MaterialSpec = sqrt(1.77)
    bot_PEC::Bool = false
    logger_cfg::RunLogConfig = RunLogConfig()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Material resolution
# ═══════════════════════════════════════════════════════════════════════════════

"""
    resolve_index(mat::MaterialSpec, λ::Float64) → ComplexF64

Get complex refractive index at wavelength λ (nm).
- String: lookup from material database
- Float64: use directly as real index
"""
function resolve_index(mat::String, λ::Float64)::ComplexF64
    n_interp, k_interp = get_interp(mat)
    n_interp(λ) - im * k_interp(λ)
end

resolve_index(n::Float64, λ::Float64)::ComplexF64 = n + 0.0im

"""Get design material index at wavelength."""
design_index(env::Environment, λ::Float64) = resolve_index(env.mat_design, λ)

"""Get substrate material index at wavelength."""
substrate_index(env::Environment, λ::Float64) = resolve_index(env.mat_substrate, λ)

"""Get fluid material index at wavelength."""
fluid_index(env::Environment, λ::Float64) = resolve_index(env.mat_fluid, λ)

Base.show(io::IO, env::Environment) =
    print(io, "Environment(design=$(env.mat_design), substrate=$(env.mat_substrate), fluid=$(env.mat_fluid))")
