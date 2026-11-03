# Analytic reference: plane wave on a uniform film on a substrate, incidence medium above.
#
# Independent of the FEM code. No dependencies.
#
# Conventions (chosen to match the FEM code's signs, stated so they can be checked):
#   time dependence      exp(+i ω t)
#   refractive index     n = n' - i n''  (n'' > 0 is loss)
#   z axis               up; incidence medium (index n0) at large z, substrate (index n2) below
#   incident wave        travels toward -z
#   in-plane wavevector  kpar = k0 * (sx, sy), with (sx, sy) = n0 * sin(theta) * (unit vector)
#                        theta is measured from the z axis IN THE INCIDENCE MEDIUM
#   normal wavenumber    q_j = sqrt(k0^2 n_j^2 - |kpar|^2), root with Im q <= 0 (decay along travel)
#
# Admittances for the TANGENTIAL electric field (ratio of tangential H to tangential E, up to
# a common constant):   s:  Y_j = q_j           p:  Y_j = k0^2 n_j^2 / q_j
#
# s = tangential E perpendicular to kpar.  p = tangential E parallel to kpar (E also has a z part).

module ObliqueTMM

export qz, admittance, interface_rt, film_rt, film_rt_matrix, RT, pol_of, reflect_tangential,
    selfcheck

"Normal wavenumber in a medium of index n for in-plane wavenumber kpar. Root with Im q <= 0."
function qz(k0::Real, n::Number, kpar::Real)
    q = sqrt(complex(k0^2 * n^2 - kpar^2))
    if imag(q) > 0 || (imag(q) == 0 && real(q) < 0)
        q = -q
    end
    return q
end

"Tangential-E admittance. pol is :s or :p."
function admittance(pol::Symbol, k0::Real, n::Number, kpar::Real)
    q = qz(k0, n, kpar)
    pol === :s && return q
    pol === :p && return k0^2 * n^2 / q
    error("pol must be :s or :p, got $pol")
end

"Amplitude r, t of the tangential E at one interface, going from medium i into medium j."
function interface_rt(Yi, Yj)
    r = (Yi - Yj) / (Yi + Yj)
    t = 2Yi / (Yi + Yj)
    return r, t
end

"""
    film_rt(pol, k0, kpar, n0, n1, d, n2) -> (r, t)

Airy sum. r, t are ratios of TANGENTIAL E: r at the top face of the film, t at the bottom face,
both relative to the incident tangential E at the top face.
"""
function film_rt(pol::Symbol, k0::Real, kpar::Real, n0, n1, d::Real, n2)
    Y0 = admittance(pol, k0, n0, kpar)
    Y1 = admittance(pol, k0, n1, kpar)
    Y2 = admittance(pol, k0, n2, kpar)
    r01, t01 = interface_rt(Y0, Y1)
    r12, t12 = interface_rt(Y1, Y2)
    ph = exp(-im * qz(k0, n1, kpar) * d)
    den = 1 + r01 * r12 * ph^2
    return (r01 + r12 * ph^2) / den, t01 * t12 * ph / den
end

"""
    film_rt_matrix(pol, k0, kpar, n0, layers, n2) -> (r, t)

Second, separately coded route: characteristic (Abeles) matrices for any stack.
`layers` is a vector of (n, d) from top to bottom. Used to cross-check `film_rt`.
"""
function film_rt_matrix(pol::Symbol, k0::Real, kpar::Real, n0, layers, n2)
    Y0 = admittance(pol, k0, n0, kpar)
    Y2 = admittance(pol, k0, n2, kpar)
    # State (E_t, H_t) with H_t = Y * E_t for a wave going down. Propagate from substrate up.
    E, H = complex(1.0), Y2 * 1.0
    for (n, d) in reverse(layers)
        Y = admittance(pol, k0, n, kpar)
        δ = qz(k0, n, kpar) * d
        E, H = cos(δ) * E + im * sin(δ) / Y * H, im * Y * sin(δ) * E + cos(δ) * H
    end
    # At the top face: E = Ei + Er, H = Y0 (Ei - Er)
    Ei = (E + H / Y0) / 2
    Er = (E - H / Y0) / 2
    return Er / Ei, 1 / Ei
end

"Power reflectance and transmittance from the tangential-E amplitudes."
function RT(pol::Symbol, k0::Real, kpar::Real, n0, n1, d::Real, n2)
    r, t = film_rt(pol, k0, kpar, n0, n1, d, n2)
    Y0 = admittance(pol, k0, n0, kpar)
    Y2 = admittance(pol, k0, n2, kpar)
    return abs2(r), real(Y2) / real(Y0) * abs2(t)
end

"""
    pol_of(pol_axis, tilt_axis) -> :s or :p

Single-axis tilt. `pol_axis` (:x or :y) is the direction of the tangential E, `tilt_axis`
(:x or :y) the direction of kpar. Parallel is p, perpendicular is s.
"""
pol_of(pol_axis::Symbol, tilt_axis::Symbol) = pol_axis === tilt_axis ? :p : :s

"""
    reflect_tangential(Et, k0, s, n0, n1, d, n2) -> (Er_t, Et_sub)

General tilt. `Et` is the incident tangential E (2-vector), `s = (sx, sy)` the in-plane
direction sines (kpar = k0 * s). Splits Et into its p part (along s) and s part (across s).
At s = (0, 0) the two coefficients are equal, so the split does not matter.
"""
function reflect_tangential(Et, k0::Real, s, n0, n1, d::Real, n2)
    kp = k0 * hypot(s[1], s[2])
    if kp == 0
        r, t = film_rt(:s, k0, 0.0, n0, n1, d, n2)
        return r .* Et, t .* Et
    end
    kh = (s[1], s[2]) ./ hypot(s[1], s[2])
    sh = (-kh[2], kh[1])
    ap = Et[1] * kh[1] + Et[2] * kh[2]
    as = Et[1] * sh[1] + Et[2] * sh[2]
    rp, tp = film_rt(:p, k0, kp, n0, n1, d, n2)
    rs, ts = film_rt(:s, k0, kp, n0, n1, d, n2)
    Er = (rp * ap * kh[1] + rs * as * sh[1], rp * ap * kh[2] + rs * as * sh[2])
    Etr = (tp * ap * kh[1] + ts * as * sh[1], tp * ap * kh[2] + ts * as * sh[2])
    return Er, Etr
end

# ---------------------------------------------------------------------------------------------
# Self-checks. Each one can fail; the last block shows that they do (a negative control).
# ---------------------------------------------------------------------------------------------

function selfcheck(; verbose=true)
    ok = true
    say(name, val, tol) = begin
        pass = val < tol
        ok &= pass
        verbose && println(rpad(name, 62), pass ? "PASS  " : "FAIL  ", val)
    end
    k0 = 2π / 532.0

    # 1. Normal incidence, no film: textbook ((n0-n2)/(n0+n2))^2.
    R, T = RT(:s, k0, 0.0, 1.0, 1.0, 0.0, 1.77)
    say("normal incidence, bare interface, R", abs(R - ((1 - 1.77) / (1 + 1.77))^2), 1e-14)

    # 2. Textbook Fresnel in cosine form, written out separately, bare interface at 30 and 60 deg.
    for θ in (30.0, 60.0)
        n0, n2 = 1.0, 1.77
        ci = cosd(θ)
        ct = sqrt(1 - (n0 * sind(θ) / n2)^2)
        Rs = ((n0 * ci - n2 * ct) / (n0 * ci + n2 * ct))^2
        Rp = ((n2 * ci - n0 * ct) / (n2 * ci + n0 * ct))^2
        kp = k0 * n0 * sind(θ)
        say("Fresnel cosine form, s, theta=$θ", abs(RT(:s, k0, kp, n0, n0, 0.0, n2)[1] - Rs), 1e-14)
        say("Fresnel cosine form, p, theta=$θ", abs(RT(:p, k0, kp, n0, n0, 0.0, n2)[1] - Rp), 1e-14)
    end

    # 3. Brewster: p reflection vanishes at tan(theta) = n2/n0, s does not.
    θB = atand(1.77)
    kp = k0 * sind(θB)
    say("Brewster angle, Rp = 0", RT(:p, k0, kp, 1.0, 1.0, 0.0, 1.77)[1], 1e-28)
    say("Brewster angle, Rs is not 0 (expect > 0.1)", 0.1 - RT(:s, k0, kp, 1.0, 1.0, 0.0, 1.77)[1], 0.0)

    # 4. Energy balance for a lossless film, both polarizations, several angles and thicknesses.
    worst = 0.0
    for pol in (:s, :p), θ in (0.0, 10.0, 30.0, 55.0, 80.0), d in (0.0, 35.0, 80.0, 400.0)
        R, T = RT(pol, k0, k0 * sind(θ), 1.0, 2.0, d, 1.77)
        worst = max(worst, abs(R + T - 1))
    end
    say("lossless film, max |R + T - 1|", worst, 1e-13)

    # 5. Airy sum against the characteristic-matrix route (separately coded), lossy film included.
    worst = 0.0
    for pol in (:s, :p), θ in (0.0, 20.0, 30.0, 70.0), n1 in (2.0, 0.05 - 3.3im, 1.3 - 0.4im)
        kp = k0 * sind(θ)
        a = film_rt(pol, k0, kp, 1.0, n1, 80.0, 1.77)
        b = film_rt_matrix(pol, k0, kp, 1.0, [(n1, 80.0)], 1.77)
        worst = max(worst, abs(a[1] - b[1]), abs(a[2] - b[2]))
    end
    say("Airy vs characteristic matrix, max |diff|", worst, 1e-12)

    # 6. Lossy film absorbs: R + T < 1, and never > 1.
    R, T = RT(:p, k0, k0 * sind(30.0), 1.0, 0.05 - 3.3im, 40.0, 1.77)
    say("lossy film, R + T < 1", R + T - 1, 0.0)

    # 7. Half-wave film of index n1 is absent at its own internal normal: R equals bare interface.
    n1 = 2.0
    θ = 30.0
    kp = k0 * sind(θ)
    d = π / real(qz(k0, n1, kp))
    say("half-wave film is invisible", abs(RT(:s, k0, kp, 1.0, n1, d, 1.77)[1] - RT(:s, k0, kp, 1.0, 1.0, 0.0, 1.77)[1]), 1e-13)

    # 8. Total internal reflection from the dense side: R = 1, substrate field evanescent.
    kp = k0 * 1.77 * sind(60.0)
    say("total internal reflection, R = 1", abs(RT(:s, k0, kp, 1.77, 1.77, 0.0, 1.0)[1] - 1), 1e-13)

    # 9. General tilt reduces to the single-axis cases, and s = 0 is polarization independent.
    kp = k0 * sind(30.0)
    Er, _ = reflect_tangential((1.0, 0.0), k0, (0.0, sind(30.0)), 1.0, 2.0, 80.0, 1.77)
    say("pol x, tilt y is s", abs(Er[1] - film_rt(:s, k0, kp, 1.0, 2.0, 80.0, 1.77)[1]) + abs(Er[2]), 1e-14)
    Er, _ = reflect_tangential((1.0, 0.0), k0, (sind(30.0), 0.0), 1.0, 2.0, 80.0, 1.77)
    say("pol x, tilt x is p", abs(Er[1] - film_rt(:p, k0, kp, 1.0, 2.0, 80.0, 1.77)[1]) + abs(Er[2]), 1e-14)
    say("pol_of mapping", Float64(!(pol_of(:x, :y) === :s && pol_of(:y, :y) === :p &&
                                    pol_of(:x, :x) === :p && pol_of(:y, :x) === :s)), 0.5)

    # 10. Diagonal tilt mixes polarizations: x-polarized in gives some y-polarized out.
    Er, _ = reflect_tangential((1.0, 0.0), k0, (sind(30.0), sind(30.0)) ./ sqrt(2), 1.0, 2.0, 80.0, 1.77)
    say("diagonal tilt gives cross-polarized reflection (expect > 1e-3)", 1e-3 - abs(Er[2]), 0.0)

    # Negative control: the substrate admittance k0 n2 cos(theta) (angle of the incidence medium
    # used inside the substrate) must NOT reproduce Fresnel. If this "passes" the checks are blind.
    θ = 30.0
    kp = k0 * sind(θ)
    Y0 = admittance(:s, k0, 1.0, kp)
    Ywrong = k0 * 1.77 * cosd(θ)
    Rwrong = abs2((Y0 - Ywrong) / (Y0 + Ywrong))
    Rright = RT(:s, k0, kp, 1.0, 1.0, 0.0, 1.77)[1]
    verbose && println(rpad("negative control: n2 cos(theta) admittance differs", 62),
        abs(Rwrong - Rright) > 1e-3 ? "PASS  " : "FAIL  ", Rwrong, " vs ", Rright)
    ok &= abs(Rwrong - Rright) > 1e-3

    return ok
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .ObliqueTMM
    ok = ObliqueTMM.selfcheck()
    println(ok ? "\nALL SELF-CHECKS PASS" : "\nSELF-CHECK FAILURE")
    k0 = 2π / 532.0
    println("\nReference table: air / film n=2.0, d=80 nm / substrate n=1.77, lambda=532 nm")
    println("theta   Rs        Ts        Rp        Tp")
    for θ in (0.0, 10.0, 20.0, 30.0, 45.0)
        Rs, Ts = ObliqueTMM.RT(:s, k0, k0 * sind(θ), 1.0, 2.0, 80.0, 1.77)
        Rp, Tp = ObliqueTMM.RT(:p, k0, k0 * sind(θ), 1.0, 2.0, 80.0, 1.77)
        println(rpad(θ, 8), round(Rs, digits=6), "  ", round(Ts, digits=6), "  ", round(Rp, digits=6), "  ", round(Tp, digits=6))
    end
    exit(ok ? 0 : 1)
end
