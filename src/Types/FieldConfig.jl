"""
    FieldConfig

Configuration for a single field solve (frequency, angle, polarization).
"""
struct FieldConfig
    λ::Float64           # Wavelength (nm)
    θ::Float64           # Angle (degrees)
    polarization::Symbol # :x or :y
    weight::Float64      # Weight in objective sum
end

# Convenience constructors
FieldConfig(λ::Float64; θ::Float64=0.0, pol::Symbol=:y, weight::Float64=1.0) =
    FieldConfig(λ, θ, pol, weight)

# Cache key excludes weight (same LU factorization)
cache_key(fc::FieldConfig) = (fc.λ, fc.θ, fc.polarization)

# Equality for deduplication (ignores weight)
function configs_equal(a::FieldConfig, b::FieldConfig)
    cache_key(a) == cache_key(b)
end

# Angular frequency
ω(fc::FieldConfig) = 2π / fc.λ

Base.show(io::IO, fc::FieldConfig) =
    print(io, "FieldConfig(λ=$(fc.λ), θ=$(fc.θ)°, $(fc.polarization), w=$(fc.weight))")
