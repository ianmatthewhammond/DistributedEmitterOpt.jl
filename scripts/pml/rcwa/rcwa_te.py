#!/usr/bin/env python3
"""
Independent RCWA / Fourier modal method for a 1D lamellar dielectric grating,
TE polarization (E along the ridges, y), normal incidence.  numpy only.

Geometry (lengths in nm), z increases downward (into the structure):
    z < 0       superstrate, index n_f          (incidence side)
    0 < z < h   grating layer: ridge n_r for |x| < w/2 (mod P), gap n_g elsewhere
    z > h       substrate, index n_s
Time convention exp(-i w t).  Fields: Ey(x,z) = sum_m S_m(z) exp(i k_m x),
k_m = m 2 pi / P (normal incidence).

Two independent ways of getting the grating-layer modes:
  (1) RCWA: eigen-decomposition of the truncated Fourier operator
          d^2 S/dz'^2 = (Kx^2 - E) S,   z' = k0 z,
      E the Toeplitz matrix of eps Fourier coefficients (Laurent's rule is the
      correct factorization for TE, where eps multiplies the continuous Ey).
  (2) "True" (exact) modal method: the analytic lamellar eigenmodes, sin/cos
      in each slab, with propagation constants from the transcendental
      dispersion relation (even and odd families), root-found by bracketing.
      Their Fourier coefficients are computed by Gauss-Legendre quadrature.
Both feed the same boundary-matching step (continuity of Ey and dEy/dz at
z=0 and z=h, projected on the Fourier orders), which is itself checked against
the analytic Fresnel / transfer-matrix result for a uniform slab.

Only decaying exponentials exp(-q k0 z), exp(q k0 (z-h)) appear in the
matching system, so it is unconditionally stable for any h and N.
"""
import json
import os
import sys
import time

# keep it single-process / single-threaded (BLAS/LAPACK)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

LAM = 532.0
N_F = 1.33     # superstrate
N_R = 2.0      # ridges
N_G = 1.33     # gaps (same as superstrate)
N_S = 1.45     # substrate
H = 150.0

KZ_SNAP = 1e-13   # |n^2 - kx^2| below this is treated as exactly grazing (kz = 0)


# --------------------------------------------------------------------------
# common pieces
# --------------------------------------------------------------------------
def kz_norm(n, kx):
    """Normalized kz/k0 = sqrt(n^2 - kx^2) on the branch Re>=0, Im>=0.
    Returns (kz, grazing_mask)."""
    val = n * n - kx * kx
    kz = np.sqrt(val.astype(complex))
    graz = np.abs(val) < KZ_SNAP
    kz[graz] = 0.0
    # enforce branch (np.sqrt principal branch already gives Re>=0 and, for
    # real negative val, +i; keep explicit for clarity)
    kz = np.where(kz.imag < 0, -kz, kz)
    return kz, graz


def orders(N):
    if N % 2 != 1:
        raise ValueError("N (total number of orders) must be odd")
    M = (N - 1) // 2
    return np.arange(-M, M + 1)


def match_layer(W, q, k0h, Y1, Y3, m):
    """Boundary matching for one layer with modal field
        S(z') = W [ exp(-q z') c+ + exp(q (z' - k0h)) c- ],
    superstrate field  delta exp(i Y1 z') + R exp(-i Y1 z'),
    substrate field    T exp(i Y3 (z' - k0h)).
    Returns complex amplitude vectors R, T and the 2N x 2N system condition no."""
    N = len(m)
    X = np.exp(-q * k0h)
    delta = (m == 0).astype(complex)
    WQ = W * q[None, :]
    iY1W = 1j * Y1[:, None] * W
    iY3W = 1j * Y3[:, None] * W
    A = np.zeros((2 * N, 2 * N), dtype=complex)
    A[:N, :N] = iY1W - WQ
    A[:N, N:] = (iY1W + WQ) * X[None, :]
    A[N:, :N] = -(WQ + iY3W) * X[None, :]
    A[N:, N:] = WQ - iY3W
    b = np.zeros(2 * N, dtype=complex)
    b[:N] = 2j * Y1 * delta
    c = np.linalg.solve(A, b)
    cp, cm = c[:N], c[N:]
    R = W @ (cp + X * cm) - delta
    T = W @ (X * cp + cm)
    return R, T, np.linalg.cond(A)


def efficiencies(R, T, Y1, Y3, m):
    inc = Y1[m == 0][0].real
    Re = np.abs(R) ** 2 * Y1.real / inc
    Te = np.abs(T) ** 2 * Y3.real / inc
    return Re, Te


# --------------------------------------------------------------------------
# (1) RCWA eigenmodes
# --------------------------------------------------------------------------
def eps_fourier(P, w, er, eg, kmax):
    """Fourier coefficients eps_k = (1/P) int eps(x) exp(-i k K x) dx,
    ridge of width w centred at x=0."""
    k = np.arange(-kmax, kmax + 1)
    c = (er - eg) * (w / P) * np.sinc(k * w / P)
    c = c.astype(complex)
    c[k == 0] += eg
    return c


def rcwa_modes(P, w, n_ridge, n_gap, lam, m):
    M = (len(m) - 1) // 2
    kx = m * (lam / P)
    ec = eps_fourier(P, w, n_ridge ** 2, n_gap ** 2, 2 * M)
    E = ec[(m[:, None] - m[None, :]) + 2 * M]
    A = np.diag(kx ** 2).astype(complex) - E
    if np.allclose(A, A.conj().T, atol=1e-14, rtol=0):
        ev, W = np.linalg.eigh(A)          # real symmetric for a centred ridge
        ev = ev.astype(complex)
        W = W.astype(complex)
    else:
        ev, W = np.linalg.eig(A)
    q = np.sqrt(ev)                         # principal branch: Re q >= 0
    return W, q, ev


def solve_rcwa(P, w, h, N, n_sup=N_F, n_ridge=N_R, n_gap=N_G, n_sub=N_S,
               lam=LAM):
    m = orders(N)
    kx = m * (lam / P)
    Y1, g1 = kz_norm(n_sup, kx)
    Y3, g3 = kz_norm(n_sub, kx)
    W, q, ev = rcwa_modes(P, w, n_ridge, n_gap, lam, m)
    if np.min(np.abs(q)) < 1e-9 and h > 0:
        raise RuntimeError("near-zero layer eigenvalue: modal basis degenerate")
    k0h = 2 * np.pi * h / lam
    R, T, cond = match_layer(W, q, k0h, Y1, Y3, m)
    Re, Te = efficiencies(R, T, Y1, Y3, m)
    return dict(m=m, kx=kx, Y1=Y1, Y3=Y3, graz1=g1, graz3=g3, R=Re, T=Te,
                Ramp=R, Tamp=T, cond=cond, ev=ev)


# --------------------------------------------------------------------------
# (2) exact lamellar eigenmodes ("true modal method")
# --------------------------------------------------------------------------
def _sinc_over(a, t):
    """sin(a t)/a, entire in a (a may be complex)."""
    a = np.asarray(a, dtype=complex)
    small = np.abs(a * t) < 1e-8
    out = np.where(small, t * (1 - (a * t) ** 2 / 6), np.sin(a * t) / np.where(small, 1, a))
    return out


def _disp(lmb, wn, gn, er, eg, parity):
    """Dispersion functions (real for real lmb) in normalized units
    (lengths times k0, lmb = (gamma/k0)^2).
    even: a sin(a w/2) cos(b g/2) + b cos(a w/2) sin(b g/2)
    odd : cos(a w/2) sin(b g/2)/b + sin(a w/2)/a cos(b g/2)"""
    a = np.sqrt((er - lmb).astype(complex))
    b = np.sqrt((eg - lmb).astype(complex))
    ca, cb = np.cos(a * wn / 2), np.cos(b * gn / 2)
    if parity == "even":
        f = (a * a) * _sinc_over(a, wn / 2) * cb + (b * b) * ca * _sinc_over(b, gn / 2)
    else:
        f = ca * _sinc_over(b, gn / 2) + _sinc_over(a, wn / 2) * cb
    return f.real


def exact_roots(P, w, n_ridge, n_gap, lam, count, parity):
    """First `count` roots lmb (descending) of the dispersion relation.
    Scans lmb = er - u^2 on a fine u grid, then bisects each sign change."""
    k0 = 2 * np.pi / lam
    wn, gn = k0 * w, k0 * (P - w)
    er, eg = n_ridge ** 2, n_gap ** 2
    du = (2 * np.pi / (k0 * P)) / 400.0
    roots = []
    u0 = 1e-9
    chunk = 20000
    while len(roots) < count:
        u = u0 + du * np.arange(chunk + 1)
        f = _disp(er - u * u, wn, gn, er, eg, parity)
        s = np.sign(f)
        idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
        for i in idx:
            lo, hi = u[i], u[i + 1]
            flo = f[i]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = _disp(np.array([er - mid * mid]), wn, gn, er, eg, parity)[0]
                if fm == 0:
                    lo = hi = mid
                    break
                if np.sign(fm) == np.sign(flo):
                    lo, flo = mid, fm
                else:
                    hi = mid
                if hi - lo < 1e-15 * max(1.0, abs(mid)):
                    break
            roots.append(er - (0.5 * (lo + hi)) ** 2)
            if len(roots) >= count:
                break
        # tiny exact zeros on grid points (unlikely) -- flag
        if np.any(f == 0):
            print("WARNING: dispersion function exactly zero on a grid point", file=sys.stderr)
        u0 = u[-1]
    return np.array(roots[:count])


def exact_mode_profile(x, lmb, P, w, n_ridge, n_gap, lam, parity):
    """Mode profile on x in [-w/2, P - w/2] (one period), continuous with
    continuous derivative; unnormalized."""
    k0 = 2 * np.pi / lam
    a = k0 * np.sqrt(complex(n_ridge ** 2 - lmb))
    b = k0 * np.sqrt(complex(n_gap ** 2 - lmb))
    g = P - w
    ridge = np.abs(x) <= w / 2
    xg = x - P / 2
    if parity == "even":
        phi = np.where(ridge, np.cos(b * g / 2) * np.cos(a * x),
                       np.cos(a * w / 2) * np.cos(b * xg))
    else:
        phi = np.where(ridge, _sinc_over(b, g / 2) * _sinc_over(a, x),
                       -_sinc_over(a, w / 2) * _sinc_over(b, xg))
    return phi.real


def exact_modes(P, w, n_ridge, n_gap, lam, m, nquad=600):
    M = (len(m) - 1) // 2
    lam_e = exact_roots(P, w, n_ridge, n_gap, lam, M + 1, "even")
    lam_o = exact_roots(P, w, n_ridge, n_gap, lam, M, "odd") if M > 0 else np.array([])
    # Gauss-Legendre on the two pieces of [-w/2, P-w/2]
    t, wt = np.polynomial.legendre.leggauss(nquad)
    pieces = [(-w / 2, w / 2), (w / 2, P - w / 2)]
    xs = np.concatenate([0.5 * (b - a) * t + 0.5 * (b + a) for a, b in pieces])
    ws = np.concatenate([0.5 * (b - a) * wt for a, b in pieces])
    kx = 2 * np.pi * m / P
    F = np.exp(-1j * np.outer(kx, xs)) * ws[None, :] / P     # (N, nx)
    cols, lmbs = [], []
    for par, ls in (("even", lam_e), ("odd", lam_o)):
        for lmb in ls:
            phi = exact_mode_profile(xs, lmb, P, w, n_ridge, n_gap, lam, par)
            phi = phi / np.sqrt(np.sum(ws * phi * phi) / P)    # unit L2 norm
            cols.append(F @ phi)
            lmbs.append(lmb)
    W = np.array(cols).T
    lmbs = np.array(lmbs)
    q = np.sqrt((-lmbs).astype(complex))    # same convention as RCWA: S ~ exp(-q k0 z)
    return W, q, lam_e, lam_o


def solve_exact_modal(P, w, h, N, n_sup=N_F, n_ridge=N_R, n_gap=N_G,
                      n_sub=N_S, lam=LAM):
    m = orders(N)
    kx = m * (lam / P)
    Y1, _ = kz_norm(n_sup, kx)
    Y3, _ = kz_norm(n_sub, kx)
    W, q, le, lo = exact_modes(P, w, n_ridge, n_gap, lam, m)
    R, T, cond = match_layer(W, q, 2 * np.pi * h / lam, Y1, Y3, m)
    Re, Te = efficiencies(R, T, Y1, Y3, m)
    return dict(m=m, R=Re, T=Te, cond=cond, lam_even=le, lam_odd=lo)


# --------------------------------------------------------------------------
# analytic reference
# --------------------------------------------------------------------------
def fresnel_slab(n1, n2, n3, h, lam):
    """Normal-incidence slab n2 of thickness h between n1 (incident) and n3."""
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    ph = np.exp(1j * 2 * np.pi * n2 * h / lam)
    den = 1 + r12 * r23 * ph * ph
    r = (r12 + r23 * ph * ph) / den
    t = t12 * t23 * ph / den
    return abs(r) ** 2, (n3 / n1) * abs(t) ** 2


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def prop_mask(res):
    Rp = (res["Y1"].real > 0)
    Tp = (res["Y3"].real > 0)
    return Rp, Tp


def by_order(res, which, mask):
    return {int(mm): float(v) for mm, v in zip(res["m"], res[which]) if mask[list(res["m"]).index(mm)]}


def print_table(res, title):
    Rp, Tp = prop_mask(res)
    print(f"  {title}")
    print("     m        R_m                 T_m")
    for i, mm in enumerate(res["m"]):
        if Rp[i] or Tp[i] or res["graz1"][i] or res["graz3"][i]:
            rs = f"{res['R'][i]:.12e}" if Rp[i] else ("grazing(0)" if res["graz1"][i] else "evanescent")
            ts = f"{res['T'][i]:.12e}" if Tp[i] else ("grazing(0)" if res["graz3"][i] else "evanescent")
            print(f"   {mm:+3d}  {rs:>20s}  {ts:>20s}")
    tot = res["R"].sum() + res["T"].sum()
    print(f"   sum R = {res['R'].sum():.15f}  sum T = {res['T'].sum():.15f}  "
          f"R+T-1 = {tot - 1:+.3e}   cond = {res['cond']:.2e}")


NLIST = [21, 41, 81, 161, 321, 641]


def convergence(P, w, h, label, out):
    print(f"\n=== {label}: P={P}, w={w}, h={h}  convergence in N (total orders) ===")
    rows = []
    for N in NLIST:
        t0 = time.time()
        res = solve_rcwa(P, w, h, N)
        rows.append(res)
        Rp, Tp = prop_mask(res)
        print(f"\n N = {N}   ({time.time() - t0:.2f} s)")
        print_table(res, "")
    # digits table
    last, prev = rows[-1], rows[-2]
    Rp, Tp = prop_mask(last)
    print(f"\n  change N={NLIST[-2]} -> {NLIST[-1]} (abs), per propagating order:")
    for i, mm in enumerate(last["m"]):
        if Rp[i] or Tp[i]:
            j = list(prev["m"]).index(mm)
            dR = abs(last["R"][i] - prev["R"][j]) if Rp[i] else float("nan")
            dT = abs(last["T"][i] - prev["T"][j]) if Tp[i] else float("nan")
            print(f"   {mm:+3d}  dR = {dR:.2e}   dT = {dT:.2e}")
    out["convergence"] = {
        str(N): {"R": by_order(r, "R", prop_mask(r)[0]),
                 "T": by_order(r, "T", prop_mask(r)[1]),
                 "R_plus_T": float(r["R"].sum() + r["T"].sum())}
        for N, r in zip(NLIST, rows)}
    return rows


def main():
    np.set_printoptions(linewidth=140)
    results = {"meta": {
        "units": "nm", "lambda0": LAM, "n_superstrate": N_F, "n_ridge": N_R,
        "n_gap": N_G, "n_substrate": N_S,
        "polarization": "TE (E along ridges, y)", "incidence": "normal, from superstrate",
        "efficiency": "|amp|^2 Re(kz_m)/kz_inc ; R into superstrate, T into substrate",
        "method": "RCWA (Laurent rule, exact for TE), largest N = %d total orders" % NLIST[-1],
        "N_list": NLIST}}

    # ---------------- case 3 sanity first (validates the matching step)
    print("=" * 78)
    print("CASE 3: sanity limits vs analytic Fresnel / transfer matrix (P = 800 lattice)")
    print("=" * 78)
    c3 = {}
    checks = [
        ("uniform slab n=2.0, h=150", dict(P=800, w=400, h=150, n_ridge=2.0, n_gap=2.0), (N_F, 2.0, N_S, 150)),
        ("uniform slab n=1.33 (gap index), h=150", None, (N_F, N_G, N_S, 150)),
        ("grating P=800 w=400 with h=0", dict(P=800, w=400, h=0.0), (N_F, N_F, N_S, 0.0)),
        ("uniform slab n=2.0, h=0", dict(P=800, w=400, h=0.0, n_ridge=2.0, n_gap=2.0), (N_F, 2.0, N_S, 0.0)),
    ]
    for name, kw, fres in checks:
        Ra, Ta = fresnel_slab(*fres, LAM)
        print(f"\n  {name}")
        print(f"   analytic : R0 = {Ra:.16f}  T0 = {Ta:.16f}  R+T-1 = {Ra + Ta - 1:+.2e}")
        entry = {"analytic": {"R0": Ra, "T0": Ta}}
        if kw is None:
            # 1.33 slab in a P=800 lattice has an exactly-zero layer eigenvalue
            # at m=+-2 (grazing in the layer), so the exp(+-q z) modal basis is
            # degenerate; use a P=300 lattice (only m=0 propagating) instead.
            kw = dict(P=300, w=150, h=150, n_ridge=N_G, n_gap=N_G)
            print("   (run on P=300 lattice: on P=800 the n=1.33 layer has an exactly "
                  "grazing m=+-2 mode, q=0, degenerate exp basis)")
        for N in (21, 161, 641):
            P_, w_, h_ = kw["P"], kw["w"], kw["h"]
            r = solve_rcwa(P_, w_, h_, N, n_ridge=kw.get("n_ridge", N_R), n_gap=kw.get("n_gap", N_G))
            i0 = list(r["m"]).index(0)
            dR, dT = r["R"][i0] - Ra, r["T"][i0] - Ta
            other = r["R"].sum() + r["T"].sum() - r["R"][i0] - r["T"][i0]
            print(f"   RCWA N={N:3d}: R0 = {r['R'][i0]:.16f}  T0 = {r['T'][i0]:.16f}  "
                  f"dR0 = {dR:+.2e}  dT0 = {dT:+.2e}  sum(other orders) = {other:.1e}")
            entry[f"N{N}"] = {"R0": float(r["R"][i0]), "T0": float(r["T"][i0]),
                              "dR0": float(dR), "dT0": float(dT),
                              "other_orders_total": float(other),
                              "R": by_order(r, "R", prop_mask(r)[0]),
                              "T": by_order(r, "T", prop_mask(r)[1])}
        c3[name] = entry
    results["case3_sanity"] = c3

    # ---------------- case 1 and case 2
    for key, (P, w, label) in {"case1": (800, 400, "CASE 1 (diffractive)"),
                               "case2": (300, 150, "CASE 2 (sub-wavelength)")}.items():
        print("\n" + "=" * 78)
        out = {"P": P, "w": w, "h": H}
        rows = convergence(P, w, H, label, out)
        best = rows[-1]
        Rp, Tp = prop_mask(best)
        out["N"] = NLIST[-1]
        out["R"] = by_order(best, "R", Rp)
        out["T"] = by_order(best, "T", Tp)
        for i, mm in enumerate(best["m"]):       # grazing orders: exactly zero power
            if best["graz1"][i]:
                out["R"][int(mm)] = 0.0
            if best["graz3"][i]:
                out["T"][int(mm)] = 0.0
        out["R"] = dict(sorted(out["R"].items()))
        out["T"] = dict(sorted(out["T"].items()))
        out["sum_R"] = float(best["R"].sum())
        out["sum_T"] = float(best["T"].sum())
        out["R_plus_T"] = float(best["R"].sum() + best["T"].sum())
        out["R_plus_T_minus_1"] = out["R_plus_T"] - 1
        out["grazing_orders_superstrate"] = [int(x) for x in best["m"][best["graz1"]]]
        out["grazing_orders_substrate"] = [int(x) for x in best["m"][best["graz3"]]]
        gamp = {int(mm): float(abs(best["Ramp"][i])) for i, mm in enumerate(best["m"]) if best["graz1"][i]}
        if gamp:
            out["grazing_superstrate_field_amplitude_abs"] = gamp
            print(f"\n  grazing superstrate orders carry zero z-power but a finite, z-independent "
                  f"field |r_m| (incident amplitude 1): {gamp}")
        # Richardson estimate from the last three N (h-step halving each time)
        ests = {}
        r1, r2, r3 = rows[-3], rows[-2], rows[-1]
        print("\n  Richardson estimate from N=%d,%d,%d (observed order p, extrapolated value, est. error of N=%d):"
              % (NLIST[-3], NLIST[-2], NLIST[-1], NLIST[-1]))
        for which in ("R", "T"):
            for mm, v3 in by_order(r3, which, prop_mask(r3)[0 if which == "R" else 1]).items():
                v1 = r1[which][list(r1["m"]).index(mm)]
                v2 = r2[which][list(r2["m"]).index(mm)]
                d1, d2 = v2 - v1, v3 - v2
                p = np.log2(abs(d1 / d2)) if d2 != 0 and d1 != 0 else float("nan")
                fac = 2.0 ** p - 1 if np.isfinite(p) and p > 0.5 else float("nan")
                ex = v3 + d2 / fac if np.isfinite(fac) else v3
                err = abs(ex - v3)
                ests[f"{which}_{mm:+d}"] = {"value_Nmax": float(v3), "order_p": float(p),
                                            "extrapolated": float(ex), "est_error_Nmax": float(err)}
                print(f"   {which}_{mm:+d}: N={NLIST[-1]} {v3:.12e}  p={p:5.2f}  extrap {ex:.12e}  err~{err:.1e}")
        out["richardson"] = ests
        # symmetry
        asym = max([abs(out["R"][mm] - out["R"][-mm]) for mm in out["R"]] +
                   [abs(out["T"][mm] - out["T"][-mm]) for mm in out["T"]])
        out["max_abs_asymmetry_m_vs_minus_m"] = asym
        print(f"\n  max |eff(m) - eff(-m)| at N={NLIST[-1]}: {asym:.2e}")

        # angles
        ang = {"superstrate": {}, "substrate": {}}
        print("\n  propagation angles (from z axis) at N = %d:" % NLIST[-1])
        for medium, n, Y, msk, g in (("superstrate", N_F, best["Y1"], Rp, best["graz1"]),
                                     ("substrate", N_S, best["Y3"], Tp, best["graz3"])):
            for i, mm in enumerate(best["m"]):
                if msk[i] or g[i]:
                    s = best["kx"][i] / n
                    cth = (Y[i].real / n)
                    th = np.degrees(np.arcsin(np.clip(s, -1, 1)))
                    ang[medium][int(mm)] = {"theta_deg": float(th), "cos_theta": float(cth),
                                            "sin_theta": float(s), "grazing": bool(g[i])}
                    print(f"   {medium:11s} m={mm:+d}: theta = {th:+10.6f} deg   "
                          f"cos = {cth:.10f}   sin = {s:+.10f}" + ("   <-- GRAZING (Rayleigh anomaly)" if g[i] else ""))
        out["angles"] = ang

        # independent cross-check: exact lamellar eigenmodes
        print("\n  Independent check: exact lamellar eigenmodes (true modal method)")
        mc = {}
        for N in (21, 41, 81, 161):
            t0 = time.time()
            ex = solve_exact_modal(P, w, H, N)
            pm = (np.abs(ex["R"]) > 0) | (np.abs(ex["T"]) > 0)
            d = max(np.max(np.abs(ex["R"] - best["R"][(len(best["m"]) - N) // 2:(len(best["m"]) + N) // 2])),
                    np.max(np.abs(ex["T"] - best["T"][(len(best["m"]) - N) // 2:(len(best["m"]) + N) // 2])))
            print(f"   modal N={N:3d}: R+T-1 = {ex['R'].sum() + ex['T'].sum() - 1:+.2e}   "
                  f"max|eff_modal - eff_RCWA(N={NLIST[-1]})| = {d:.2e}   ({time.time() - t0:.2f} s)")
            mc[str(N)] = {"R": {int(mm): float(v) for mm, v in zip(ex["m"], ex["R"]) if v > 0},
                          "T": {int(mm): float(v) for mm, v in zip(ex["m"], ex["T"]) if v > 0},
                          "max_abs_diff_vs_rcwa_largestN": float(d),
                          "R_plus_T": float(ex["R"].sum() + ex["T"].sum())}
        out["exact_modal_crosscheck"] = mc
        # eigenvalue comparison
        lam_rc = np.sort((-best["ev"]).real)[::-1]
        lam_ex = np.sort(np.concatenate([ex["lam_even"], ex["lam_odd"]]))[::-1]
        k = 12
        print("   layer mode (gamma/k0)^2, exact vs RCWA N=%d (first %d):" % (NLIST[-1], k))
        for j in range(k):
            print(f"     {j:2d}  exact {lam_ex[j]:+.12f}   RCWA {lam_rc[j]:+.12f}   diff {lam_rc[j] - lam_ex[j]:+.2e}")
        out["layer_eigs_exact_vs_rcwa"] = [[float(lam_ex[j]), float(lam_rc[j])] for j in range(k)]

        results[key] = out

    # ---------------- Rayleigh-anomaly sensitivity for case 1
    print("\n" + "=" * 78)
    print("CASE 1 sensitivity: m=+-2 is exactly grazing in the superstrate (2*532/800 = 1.33)")
    sens = {}
    for lam in (531.9, 531.99, 531.999, 532.0, 532.001, 532.01, 532.1):
        r = solve_rcwa(800, 400, H, 321, lam=lam)
        Rp, Tp = prop_mask(r)
        R = by_order(r, "R", Rp)
        T = by_order(r, "T", Tp)
        print(f"  lam={lam:8.3f}: R0={R[0]:.8f} R1={R[1]:.8f} R2={R.get(2, 0.0):.3e} "
              f"T0={T[0]:.8f} T1={T[1]:.8f} T2={T[2]:.8f}  R+T-1={sum(R.values()) + sum(T.values()) - 1:+.1e}")
        sens[str(lam)] = {"R": R, "T": T}
    results["case1_rayleigh_anomaly_sensitivity_N321"] = sens

    path = __file__.rsplit("/", 1)[0] + "/rcwa_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1, sort_keys=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
