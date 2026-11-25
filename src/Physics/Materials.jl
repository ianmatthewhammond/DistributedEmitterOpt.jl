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

`refindex` loads the requested table afresh and returns its interpolants. `resolve_index`
uses a cached copy instead; see `clear_material_cache!` for refreshing it.

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

    λ_raw, n_raw, k_raw = read_index_table(filename)

    # Build interpolants
    n_interp = Interpolations.linear_interpolation((λ_raw,), n_raw)
    k_interp = Interpolations.linear_interpolation((λ_raw,), k_raw)

    return n_interp, k_interp
end

# A field is a number by how it is written, whatever its size: a decimal literal such as
# `0.5`, `-2` or `1e309`, a hexadecimal floating literal such as `0x1p-1` or `0x1p1024`,
# or anything `Float64` parses (e.g. `Inf`, `NaN`).
const _DECIMAL_TOKEN = r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
const _HEX_TOKEN = r"^[+-]?0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?$"
_is_number_token(p::AbstractString) =
    occursin(_DECIMAL_TOKEN, p) || occursin(_HEX_TOKEN, p) || tryparse(Float64, p) !== nothing

# The Float64 value of a number field. A literal outside the Float64 range is converted
# through BigFloat, so a too-large one becomes ±Inf (and is then refused as not finite).
function _number_value(p::AbstractString)
    v = tryparse(Float64, p)
    v === nothing || return v
    b = tryparse(BigFloat, String(p))
    return b === nothing ? NaN : Float64(b)
end

"""
    read_index_table(filename) -> (λ_nm, n, k)

Parse a refractive-index table with rows `wavelength[μm]  n  k`. Fields may be
separated by commas, whitespace, or both. Blank lines and whole-line `#` comments are
skipped, as is one header line before the first data row; a header is exactly three
non-empty fields none of which is written as a number (e.g. `lambda n k`). Whether a field
is a number is decided by how it is written (decimal or hexadecimal floating literal),
whatever its size, so `1e309` and `0x1p1024` are numbers.
Every other line must hold exactly three numbers, otherwise an error naming the file and
line is raised (a first row containing an empty field or any numeric field cannot qualify
as a header). A number too large for `Float64` becomes infinite on conversion. The wavelength
converted to nm must be positive and finite and n, k must be finite; the check is made on
the values that are stored, and names the file and line. At least two rows are required
and the wavelengths must be strictly monotonic (descending tables are reversed to
ascending). The order is checked once every row has been read, so a row that cannot be
read is reported even when an earlier row breaks the order; an ordering error names the
file and the source line of the first row that repeats or reverses the order. Wavelengths
are returned in nm.
"""
function read_index_table(filename::AbstractString)
    λ_raw = Float64[]
    n_raw = Float64[]
    k_raw = Float64[]
    lines = Int[]
    header_allowed = true

    for (lineno, line) in enumerate(readlines(filename))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue

        parts = split(s, r"\s*,\s*|\s+")

        if length(parts) != 3 || !all(_is_number_token, parts)
            # One header line (three non-empty, non-numeric names, e.g. "lambda n k") is
            # allowed before the data. A row with any numeric field is data, not a header.
            if header_allowed && isempty(λ_raw) && length(parts) == 3 &&
               all(p -> !isempty(p) && !_is_number_token(p), parts)
                header_allowed = false
                continue
            end
            error("Malformed data in material file $filename, line $lineno: " *
                  "expected 3 numeric fields (wavelength[μm], n, k), got \"$s\"")
        end
        vals = _number_value.(parts)
        λ_nm = vals[1] * 1e3  # Convert μm to nm
        (λ_nm > 0 && isfinite(λ_nm) && isfinite(vals[2]) && isfinite(vals[3])) ||
            error("Malformed data in material file $filename, line $lineno: " *
                  "the wavelength must be positive and finite (in nm) and n, k finite, got \"$s\"")
        header_allowed = false
        push!(λ_raw, λ_nm)
        push!(n_raw, vals[2])
        push!(k_raw, vals[3])
        push!(lines, lineno)
    end

    length(λ_raw) >= 2 ||
        error("Material file $filename needs at least 2 data rows, found $(length(λ_raw))")

    # Strictly monotonic in file order (ascending or descending)
    dir = sign(λ_raw[2] - λ_raw[1])
    for i in 2:length(λ_raw)
        (dir != 0 && sign(λ_raw[i] - λ_raw[i-1]) == dir) ||
            error("Wavelengths in material file $filename must be strictly monotonic: " *
                  "line $(lines[i]) repeats or reverses the order")
    end

    # Ensure ascending order
    if dir < 0
        reverse!(λ_raw); reverse!(n_raw); reverse!(k_raw)
    end

    return λ_raw, n_raw, k_raw
end

"""
    complex_index(material::String, λ::Float64) -> ComplexF64

Get complex refractive index n - ik at wavelength λ (nm) (the package convention for absorbing media).

Calls `refindex`, which loads the requested table afresh, so it can differ from
`resolve_index` for the same material after the file has changed; see `clear_material_cache!`.
"""
function complex_index(material::String, λ::Float64)
    n_interp, k_interp = refindex(material)
    n_interp(λ) - im * k_interp(λ)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Common material presets
# ═══════════════════════════════════════════════════════════════════════════════

# Cache loaded interpolants
const _MATERIAL_CACHE = Dict{String,Tuple{Any,Any}}()

"""
    get_interp(material) -> (n_interp, k_interp)

Interpolants for `material` from a module-level cache keyed by the material name. The first
call for a name loads the table with `refindex` and stores the result; later calls return the
stored interpolants without reading the file. A failed load leaves the name uncached, so the
next call retries. See `clear_material_cache!` for refreshing.
"""
function get_interp(material::String)
    if !haskey(_MATERIAL_CACHE, material)
        _MATERIAL_CACHE[material] = refindex(material)
    end
    return _MATERIAL_CACHE[material]
end

"""
    clear_material_cache!()

Empty the named-material interpolant cache used by `get_interp` and `resolve_index`.

Call `DistributedEmitterOpt.clear_material_cache!()` after editing a table. The next cache miss
reloads it. Existing interpolants retain their values.

Clearing does not recompute values already derived from a table. In particular, a tilted
`FieldConfig` built by the convenience constructors stores the fluid index at its wavelength.
For a Bloch configuration, `build_phys_params` throws `ArgumentError` if the fluid index's real
part differs from `n_incident` by more than `1e-12` or its imaginary magnitude exceeds `1e-12`.
"""
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
