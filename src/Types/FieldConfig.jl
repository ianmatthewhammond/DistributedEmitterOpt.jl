"""
    FieldConfig(λ; θ=0, tilt_x=0, tilt_y=θ, pol=:y, weight=1, n_incident=1)

The downward incident direction is `(sin(tilt_x), sin(tilt_y), -sqrt(1-sx²-sy²))`.
Angles are in degrees in a lossless incident medium of index `n_incident` (air by
default). With exp(+iωt), physical E = envelope * exp(-i k_parallel⋅r),
so positive tilts describe a ray toward positive x/y and negative z. Legacy
`θ` is the y tilt; it cannot be combined with `tilt_y`. Grazing directions are rejected.

`:x` is the normalized projection of x onto the plane perpendicular to that
direction; `:y` completes its orthonormal transverse basis, with positive y at
normal incidence. For x-only tilt these are p/s, for y-only tilt s/p. The basis
is explicit for two-axis tilts and is not generally the s/p basis there.

Old normal-incidence calls retain the legacy assembly exactly. Explicit tilt
keywords select the corrected Bloch/specular-admittance formulation even at
normal incidence. `formulation=:legacy` is accepted only at normal incidence.
Weights affect the objective, never the field-solve cache key.
"""
struct FieldConfig
    λ::Float64
    θ::Float64
    polarization::Symbol
    weight::Float64
    tilt_x::Float64
    n_incident::Float64
    formulation::Symbol
end

function FieldConfig(λ::Real; θ::Real=0, tilt_x=nothing, tilt_y=nothing,
                     pol::Symbol=:y, weight::Real=1, n_incident::Real=1,
                     formulation=nothing)
    θ != 0 && tilt_y !== nothing && throw(ArgumentError("FieldConfig: use θ or tilt_y, not both"))
    tx = tilt_x === nothing ? 0.0 : Float64(tilt_x)
    ty = tilt_y === nothing ? Float64(θ) : Float64(tilt_y)
    mode = formulation === nothing ?
        ((tilt_x === nothing && tilt_y === nothing && θ == 0) ? :legacy : :bloch) : formulation
    pol in (:x, :y) || throw(ArgumentError("FieldConfig: polarization must be :x or :y"))
    mode in (:legacy, :bloch) || throw(ArgumentError("FieldConfig: unknown formulation $mode"))
    all(isfinite, (λ, tx, ty, weight, n_incident)) || throw(ArgumentError("FieldConfig: values must be finite"))
    λ > 0 && n_incident > 0 || throw(ArgumentError("FieldConfig: wavelength and incident index must be positive"))
    abs(tx) < 90 && abs(ty) < 90 && sind(tx)^2 + sind(ty)^2 < 1 ||
        throw(ArgumentError("FieldConfig: tilt_x and tilt_y must define a propagating nongrazing direction"))
    mode == :legacy && (tx != 0 || ty != 0) && throw(ArgumentError("FieldConfig: legacy formulation is normal-incidence only"))
    FieldConfig(Float64(λ), ty, pol, Float64(weight), tx, Float64(n_incident), mode)
end
FieldConfig(λ::Real, θ::Real, pol::Symbol, weight::Real) = FieldConfig(λ; θ, pol, weight)

cache_key(fc::FieldConfig) = fc.formulation == :legacy ? (fc.λ, fc.θ, fc.polarization) :
    (fc.λ, fc.θ, fc.polarization, fc.tilt_x, fc.n_incident, fc.formulation)
field_config(key::Tuple{Float64,Float64,Symbol}) = FieldConfig(key[1]; θ=key[2], pol=key[3])
field_config(key::Tuple{Float64,Float64,Symbol,Float64,Float64,Symbol}) =
    FieldConfig(key[1]; tilt_y=key[2], pol=key[3], tilt_x=key[4], n_incident=key[5], formulation=key[6])
configs_equal(a::FieldConfig, b::FieldConfig) = cache_key(a) == cache_key(b)
ω(fc::FieldConfig) = 2π / fc.λ
inplane_wavevector(fc::FieldConfig) = ω(fc) * fc.n_incident * VectorValue(sind(fc.tilt_x), sind(fc.θ), 0.0)

function polarization_vector(fc::FieldConfig)
    sx, sy = sind(fc.tilt_x), sind(fc.θ)
    d = VectorValue(sx, sy, -sqrt(1-sx^2-sy^2))
    ex = VectorValue(1.0, 0.0, 0.0) - sx*d
    ex = ex / norm(ex)
    fc.polarization == :x ? ex : -cross(d, ex)
end

Base.show(io::IO, fc::FieldConfig) = print(io,
    "FieldConfig(λ=$(fc.λ), tilt_x=$(fc.tilt_x)°, tilt_y=$(fc.θ)°, $(fc.polarization), w=$(fc.weight))")
