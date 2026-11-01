"""
    PML

Perfectly matched layers on the ±z ends of the unit cell, by complex
coordinate stretching in z only.

The time convention of this package is exp(+iωt) (indices n - ik, outgoing
waves e^{∓ik z} going up / down). The stretch that attenuates outgoing waves
in that convention is

    s(z) = κ(u) - i σ(u),   u ∈ [0, 1] the depth into the layer,
    κ(u) = 1 + (κmax - 1) uᵐ,   σ(u) = σmax uᵐ.

A z-only stretch (sx = sy = 1, sz = s) is equivalent to the anisotropic
material ε diag(sy sz/sx, sx sz/sy, sx sy/sz), i.e.

    ε → ε  diag(s, s, 1/s)        μ⁻¹ → μ⁻¹ diag(1/s, 1/s, s)

so the weak form keeps the plain curl and gains two diagonal tensors. No
derivative of s enters, and the bilinear form stays complex symmetric.

σmax is set from R, the round-trip *amplitude* attenuation of the continuous
layer at normal incidence: R = exp(-2 k₀ n σmax d / (m + 1)) (power ~R²). An
order travelling at angle θ inside the layer is attenuated by R^cos(θ). What
actually returns also depends on the outer-face termination and on the mesh.
"""

"""
    ZPML(; R=1e-8, m=3.0, κmax=1.0)

User-facing PML recipe, set on `Environment(pml=ZPML(...))`. The layer
thicknesses come from the mesh (volumes `PMLBottom` / `PMLTop`, see
`genmesh(...; pml=true)`).

- `R` — round-trip amplitude attenuation at normal incidence (continuous layer)
- `m` — polynomial grading order
- `κmax` — real stretch at the outer wall (≥ 1; > 1 speeds evanescent decay)
"""
Base.@kwdef struct ZPML
    R::Float64 = 1e-8
    m::Float64 = 3.0
    κmax::Float64 = 1.0
end

"""
    ZPMLParams

Resolved PML for one frequency: where each layer starts (`z_bot`, `z_top`,
the faces touching the physical region), how thick it is, and its σmax.
A thickness of 0 disables that side.
"""
struct ZPMLParams
    z_bot::Float64
    d_bot::Float64
    σ_bot::Float64
    z_top::Float64
    d_top::Float64
    σ_top::Float64
    m::Float64
    κmax::Float64
end

"""σmax giving round-trip amplitude attenuation R through thickness d in a medium of index n."""
pml_sigma(R, m, k0, n, d) = d > 0 ? -(m + 1) * log(R) / (2 * k0 * real(n) * d) : 0.0

"""
    ZPMLParams(spec, k0, n_bot, n_top, z_bot, d_bot, z_top, d_top)

Resolve a `ZPML` recipe for wavenumber k0 = 2π/λ. `n_bot` / `n_top` are the
indices of the half-spaces the layers continue (substrate / fluid).
"""
function ZPMLParams(spec::ZPML, k0, n_bot, n_top, z_bot, d_bot, z_top, d_top)
    spec.R > 0 && spec.R < 1 || error("ZPML: R must be in (0, 1), got $(spec.R)")
    isfinite(spec.m) && spec.m > 0 || error("ZPML: grading order m must be finite and > 0, got $(spec.m)")
    isfinite(spec.κmax) && spec.κmax >= 1 || error("ZPML: κmax must be finite and ≥ 1, got $(spec.κmax)")
    isfinite(k0) && k0 > 0 || error("ZPML: k0 must be finite and > 0, got $k0")
    for (side, n, d) in (("bottom", n_bot, d_bot), ("top", n_top, d_top))
        isfinite(d) && d >= 0 || error("ZPML: $side thickness must be finite and ≥ 0, got $d")
        d > 0 && !(isfinite(real(n)) && real(n) > 0) &&
            error("ZPML: $side layer needs a medium with finite Re(n) > 0, got $n")
    end
    ZPMLParams(z_bot, d_bot, pml_sigma(spec.R, spec.m, k0, n_bot, d_bot),
        z_top, d_top, pml_sigma(spec.R, spec.m, k0, n_top, d_top),
        spec.m, spec.κmax)
end

"""Stretch factor s(z) (1 outside the layers)."""
function pml_stretch(z, pml::ZPMLParams)
    if pml.d_bot > 0 && z < pml.z_bot
        u = min((pml.z_bot - z) / pml.d_bot, 1.0)^pml.m
        return complex(1 + (pml.κmax - 1) * u, -pml.σ_bot * u)
    elseif pml.d_top > 0 && z > pml.z_top
        u = min((z - pml.z_top) / pml.d_top, 1.0)^pml.m
        return complex(1 + (pml.κmax - 1) * u, -pml.σ_top * u)
    end
    return complex(1.0)
end

"""Mass tensor diag(s, s, 1/s) at point x."""
function pml_Λε(x, pml::ZPMLParams)
    s = pml_stretch(x[3], pml)
    TensorValue{3,3,ComplexF64}(s, 0, 0, 0, s, 0, 0, 0, 1 / s)
end

"""Curl tensor diag(1/s, 1/s, s) at point x."""
function pml_Λν(x, pml::ZPMLParams)
    s = pml_stretch(x[3], pml)
    TensorValue{3,3,ComplexF64}(1 / s, 0, 0, 0, 1 / s, 0, 0, 0, s)
end
