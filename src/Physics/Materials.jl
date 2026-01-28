"""
    Materials

Refractive index interpolation from data files. Supports common plasmonic
materials (Ag, Au, Cu) with wavelength-dependent n(λ) and k(λ).
"""

import Interpolations

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

"""
    refindex(material::String) -> (n_interp, k_interp)

Load refractive index data for a material and return interpolation functions.
The interpolants are callable with wavelength in nm.

## Example
```julia
n_Ag, k_Ag = refindex("Ag")
n_532 = n_Ag(532.0)  # Real part at 532 nm
k_532 = k_Ag(532.0)  # Imaginary part
```
"""
function refindex(material::String)
    filename = joinpath(@__DIR__, "..", "..", "data", "materials", "$(material).txt")

    if !isfile(filename)
        error("Material file not found: $filename")
    end

    # Parse data file (format: wavelength[μm] n k, tab-separated, with # comments)
    lines = readlines(filename)

    # Filter out comments and empty lines
    data_lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), lines)

    # Parse each data line
    λ_raw = Float64[]
    n_raw = Float64[]
    k_raw = Float64[]

    for line in data_lines
        # Split on whitespace (tabs or spaces)
        parts = split(strip(line))
        if length(parts) >= 3
            push!(λ_raw, parse(Float64, parts[1]) * 1e3)  # Convert μm to nm
            push!(n_raw, parse(Float64, parts[2]))
            push!(k_raw, parse(Float64, parts[3]))
        end
    end

    if isempty(λ_raw)
        error("No valid data found in material file: $filename")
    end

    # Ensure ascending order
    if λ_raw[end] < λ_raw[1]
        λ_raw = reverse(λ_raw)
        n_raw = reverse(n_raw)
        k_raw = reverse(k_raw)
    end

    # Build interpolants
    n_interp = Interpolations.linear_interpolation((λ_raw,), n_raw)
    k_interp = Interpolations.linear_interpolation((λ_raw,), k_raw)

    return n_interp, k_interp
end

"""
    complex_index(material::String, λ::Float64) -> ComplexF64

Get complex refractive index n + ik at wavelength λ (nm).
"""
function complex_index(material::String, λ::Float64)
    n_interp, k_interp = refindex(material)
    n_interp(λ) + im * k_interp(λ)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Common material presets
# ═══════════════════════════════════════════════════════════════════════════════

# Cache loaded interpolants
const _MATERIAL_CACHE = Dict{String,Tuple{Any,Any}}()

"""Get cached interpolant (loads once per session)."""
function get_interp(material::String)
    if !haskey(_MATERIAL_CACHE, material)
        _MATERIAL_CACHE[material] = refindex(material)
    end
    return _MATERIAL_CACHE[material]
end

"""Clear material cache."""
clear_material_cache!() = empty!(_MATERIAL_CACHE)

# ═══════════════════════════════════════════════════════════════════════════════
# Plasmon wavelength helper
# ═══════════════════════════════════════════════════════════════════════════════

"""
    plasmon_period(n_fluid, n_metal, λ) -> Float64

Estimate surface plasmon polariton wavelength: λ_spp = λ / Re(√(εm·εf / (εm + εf)))
"""
function plasmon_period(n_fluid::Real, n_metal::ComplexF64, λ::Float64)
    εf = n_fluid^2
    εm = n_metal^2
    λ / real(sqrt(εm * εf / (εm + εf)))
end
