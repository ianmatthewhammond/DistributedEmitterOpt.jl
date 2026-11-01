#!/usr/bin/env python3
"""
v2 driver: anomaly-free cases A (P=780), B (P=600), C (P=300), reusing the
validated solver in rcwa_te.py unchanged.  Writes rcwa_results_v2.json.

Field-amplitude convention (same as rcwa_te.py):
  time dependence exp(-i omega t)  (physicist's).  For exp(+i omega t) take the
  complex conjugate of every r_m, t_m below.
  z increases downward; z = 0 is the superstrate/grating interface (top of
  the ridges), z = h the grating/substrate interface; x = 0 at the ridge centre.
  superstrate (z<0): Ey = exp(i k0 n_f z) + sum_m r_m exp(i k_m x - i kz1_m z)
  substrate  (z>h): Ey = sum_m t_m exp(i k_m x + i kz3_m (z - h))
  k_m = 2 pi m / P, kz_m = sqrt(k^2 n^2 - k_m^2) with Re, Im >= 0.
  So r_m is referenced to the plane z = 0 and t_m to the plane z = h, both
  relative to the incident Ey of amplitude 1 (phase 0) at z = 0, x = 0.
  Efficiencies: R_m = |r_m|^2 Re(kz1_m)/(k0 n_f), T_m = |t_m|^2 Re(kz3_m)/(k0 n_f).
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcwa_te as rc  # noqa: E402

LAM, NF, NS, NR, H = rc.LAM, rc.N_F, rc.N_S, rc.N_R, rc.H
NLIST = rc.NLIST
MMAX_REPORT = 4


def fresnel_slab_complex(n1, n2, n3, h, lam):
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    ph = np.exp(1j * 2 * np.pi * n2 * h / lam)
    den = 1 + r12 * r23 * ph * ph
    return (r12 + r23 * ph * ph) / den, t12 * t23 * ph / den   # r at z=0, t at z=h


def cplx(z):
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
            "phase_deg": float(np.degrees(np.angle(z)))}


def fresnel_checks(P):
    print(f"\n  Fresnel limits on the P={P} lattice (complex amplitudes r0 @ z=0, t0 @ z=h)")
    out = {}
    for name, nr, ng, h, ref in (
            ("uniform slab n=2.0 h=150", 2.0, 2.0, 150.0, (NF, 2.0, NS, 150.0)),
            ("grating h=0", NR, rc.N_G, 0.0, (NF, NF, NS, 0.0))):
        ra, ta = fresnel_slab_complex(*ref, LAM)
        Ra, Ta = abs(ra) ** 2, (NS / NF) * abs(ta) ** 2
        e = {}
        for N in (21, 161, 641):
            r = rc.solve_rcwa(P, P / 2, h, N, n_ridge=nr, n_gap=ng)
            i0 = list(r["m"]).index(0)
            other = r["R"].sum() + r["T"].sum() - r["R"][i0] - r["T"][i0]
            e[f"N{N}"] = {"dR0": float(r["R"][i0] - Ra), "dT0": float(r["T"][i0] - Ta),
                          "d_r0": float(abs(r["Ramp"][i0] - ra)),
                          "d_t0": float(abs(r["Tamp"][i0] - ta)),
                          "other_orders_total": float(other)}
            print(f"   {name:26s} N={N:3d}: dR0={e[f'N{N}']['dR0']:+.1e} dT0={e[f'N{N}']['dT0']:+.1e} "
                  f"|dr0|={e[f'N{N}']['d_r0']:.1e} |dt0|={e[f'N{N}']['d_t0']:.1e} other={other:.1e}")
        e["analytic"] = {"R0": Ra, "T0": Ta, "r0": cplx(ra), "t0": cplx(ta)}
        out[name] = e
    return out


def run_case(label, P, w):
    print("\n" + "=" * 78)
    print(f"CASE {label}: P={P}, w={w}, h={H}")
    print("=" * 78)
    out = {"P": P, "w": w, "h": H, "N": NLIST[-1]}
    out["fresnel_checks"] = fresnel_checks(P)

    # Rayleigh-anomaly distances
    print("\n  distance from Rayleigh anomaly  |n^2 - (m lam/P)^2|  (sign: + propagating, - evanescent)")
    ra = {}
    for mm in range(0, MMAX_REPORT + 1):
        s = mm * LAM / P
        dF, dS = NF ** 2 - s * s, NS ** 2 - s * s
        ra[mm] = {"superstrate(and gaps, n=1.33)": float(dF), "substrate": float(dS),
                  "ridge_n2.0": float(NR ** 2 - s * s)}
        print(f"   |m|={mm}: superstrate/gaps {dF:+.6f}   substrate {dS:+.6f}   ridge(n=2) {NR**2 - s*s:+.6f}")
    out["rayleigh_distance_signed_n2_minus_kx2"] = {str(k): v for k, v in ra.items()}
    allabs = [abs(v) for d in ra.values() for k, v in d.items() if not k.startswith("ridge")]
    allabs = [a for a in allabs]  # include m=0 too (trivially far)
    out["min_rayleigh_distance_superstrate_substrate"] = float(min(allabs))
    print(f"   nearest anomaly (superstrate/substrate, any m): {min(allabs):.6f}")

    # convergence
    rows = []
    print("\n  convergence (efficiencies of propagating orders, m >= 0 shown; -m identical)")
    for N in NLIST:
        r = rc.solve_rcwa(P, w, H, N)
        rows.append(r)
        Rp, Tp = rc.prop_mask(r)
        R = rc.by_order(r, "R", Rp)
        T = rc.by_order(r, "T", Tp)
        tot = r["R"].sum() + r["T"].sum()
        s = "  ".join([f"R{k:+d}={v:.10e}" for k, v in R.items() if k >= 0] +
                      [f"T{k:+d}={v:.10e}" for k, v in T.items() if k >= 0])
        print(f"   N={N:3d}: {s}  R+T-1={tot - 1:+.1e}  cond={r['cond']:.1e}")
    out["convergence"] = {str(N): {"R": rc.by_order(r, "R", rc.prop_mask(r)[0]),
                                   "T": rc.by_order(r, "T", rc.prop_mask(r)[1]),
                                   "R_plus_T": float(r["R"].sum() + r["T"].sum())}
                          for N, r in zip(NLIST, rows)}
    best = rows[-1]
    Rp, Tp = rc.prop_mask(best)
    out["R"] = rc.by_order(best, "R", Rp)
    out["T"] = rc.by_order(best, "T", Tp)
    out["sum_R"] = float(best["R"].sum())
    out["sum_T"] = float(best["T"].sum())
    out["R_plus_T_minus_1"] = float(best["R"].sum() + best["T"].sum() - 1)
    out["max_abs_asymmetry_m_vs_minus_m"] = float(max(
        [abs(out["R"][k] - out["R"][-k]) for k in out["R"]] + [abs(out["T"][k] - out["T"][-k]) for k in out["T"]]))

    # Richardson
    r1, r2, r3 = rows[-3], rows[-2], rows[-1]
    rich = {}
    print("\n  Richardson (N=161,321,641): order p and estimated error of N=641")
    for which, msk in (("R", Rp), ("T", Tp)):
        for mm, v3 in rc.by_order(r3, which, msk).items():
            v1 = r1[which][list(r1["m"]).index(mm)]
            v2 = r2[which][list(r2["m"]).index(mm)]
            d1, d2 = v2 - v1, v3 - v2
            p = float(np.log2(abs(d1 / d2))) if d1 and d2 else float("nan")
            ex = v3 + d2 / (2 ** p - 1) if np.isfinite(p) and p > 0.5 else v3
            rich[f"{which}_{mm:+d}"] = {"order_p": p, "extrapolated": float(ex), "est_error_Nmax": float(abs(ex - v3))}
            if mm >= 0:
                print(f"   {which}{mm:+d}: p={p:5.2f}  err~{abs(ex - v3):.1e}")
    out["richardson"] = rich

    # complex amplitudes, with N-convergence and exact-modal cross-check
    m = best["m"]
    sel = np.abs(m) <= MMAX_REPORT
    prev = rows[-2]
    ex = rc.solve_exact_modal  # efficiencies; amplitudes recomputed below
    Nm = 161
    mm_ex = rc.orders(Nm)
    Wm, qm, _, _ = rc.exact_modes(P, w, rc.N_R, rc.N_G, LAM, mm_ex)
    Y1m, _ = rc.kz_norm(NF, mm_ex * LAM / P)
    Y3m, _ = rc.kz_norm(NS, mm_ex * LAM / P)
    Rm, Tm, _ = rc.match_layer(Wm, qm, 2 * np.pi * H / LAM, Y1m, Y3m, mm_ex)
    amps = {}
    print("\n  complex field amplitudes (exp(-i w t); r_m at z=0, t_m at z=h), N=641")
    print("     m   state(sup/sub)        r_m                            t_m              "
          "  |dr|,|dt| vs N=321   |dr|,|dt| vs exact-modal N=161")
    for i in np.nonzero(sel)[0]:
        mi = int(m[i])
        j = list(prev["m"]).index(mi)
        k = list(mm_ex).index(mi)
        st1 = "prop" if Rp[i] else ("graz" if best["graz1"][i] else "evan")
        st3 = "prop" if Tp[i] else ("graz" if best["graz3"][i] else "evan")
        r_, t_ = best["Ramp"][i], best["Tamp"][i]
        dN = (abs(r_ - prev["Ramp"][j]), abs(t_ - prev["Tamp"][j]))
        dM = (abs(r_ - Rm[k]), abs(t_ - Tm[k]))
        amps[str(mi)] = {"r": cplx(r_), "t": cplx(t_), "superstrate": st1, "substrate": st3,
                         "kz_superstrate_over_k0": cplx(best["Y1"][i]),
                         "kz_substrate_over_k0": cplx(best["Y3"][i]),
                         "abs_change_vs_N321": {"r": float(dN[0]), "t": float(dN[1])},
                         "abs_diff_vs_exact_modal_N161": {"r": float(dM[0]), "t": float(dM[1])}}
        print(f"   {mi:+3d}   {st1}/{st3}   {r_.real:+.9e}{r_.imag:+.9e}i  {t_.real:+.9e}{t_.imag:+.9e}i"
              f"   {dN[0]:.1e},{dN[1]:.1e}   {dM[0]:.1e},{dM[1]:.1e}")
    out["amplitudes"] = amps
    out["exact_modal_N161_max_eff_diff"] = float(max(
        np.max(np.abs(np.abs(Rm) ** 2 * Y1m.real / NF - best["R"][(len(m) - Nm) // 2:(len(m) + Nm) // 2])),
        np.max(np.abs(np.abs(Tm) ** 2 * Y3m.real / NF - best["T"][(len(m) - Nm) // 2:(len(m) + Nm) // 2]))))
    print(f"   exact-modal (N=161) vs RCWA (N=641) max efficiency diff: {out['exact_modal_N161_max_eff_diff']:.1e}")

    # angles
    ang = {"superstrate": {}, "substrate": {}}
    print("\n  angles from the z axis")
    for medium, n, Y, msk in (("superstrate", NF, best["Y1"], Rp), ("substrate", NS, best["Y3"], Tp)):
        for i, mi in enumerate(m):
            if msk[i]:
                s = best["kx"][i] / n
                th = float(np.degrees(np.arcsin(s)))
                ang[medium][int(mi)] = {"theta_deg": th, "cos_theta": float(Y[i].real / n), "sin_theta": float(s)}
                print(f"   {medium:11s} m={mi:+d}: theta={th:+11.6f} deg  cos={Y[i].real / n:.10f}")
    out["angles"] = ang

    # final table
    print("\n  converged (N=641), 6 sig figs:")
    for k in sorted(set(out["R"]) | set(out["T"])):
        rs = f"{out['R'][k]:.6g}" if k in out["R"] else "evanescent"
        ts = f"{out['T'][k]:.6g}" if k in out["T"] else "evanescent"
        print(f"   m={k:+d}:  R={rs:>12s}   T={ts:>12s}")
    print(f"   sum R={out['sum_R']:.6g}  sum T={out['sum_T']:.6g}  R+T-1={out['R_plus_T_minus_1']:+.1e}")
    return out


def main():
    res = {"meta": {
        "solver": "rcwa_te.py (unchanged), RCWA with Laurent rule, TE, normal incidence",
        "units": "nm", "lambda0": LAM, "n_superstrate": NF, "n_gap": rc.N_G, "n_ridge": NR,
        "n_substrate": NS, "h": H, "N_list": NLIST,
        "time_convention": "exp(-i omega t); for exp(+i omega t) conjugate all r_m, t_m",
        "geometry": "z positive downward into the structure; z=0 superstrate/grating interface "
                    "(top of ridges); z=h grating/substrate interface; x=0 at ridge centre",
        "phase_reference": "incident Ey = exp(i k0 n_f z), amplitude 1 and phase 0 at (x=0, z=0). "
                           "Reflected Ey = sum r_m exp(i k_m x - i kz1_m z): r_m referenced to z=0. "
                           "Transmitted Ey = sum t_m exp(i k_m x + i kz3_m (z-h)): t_m referenced to z=h. "
                           "k_m = 2 pi m/P; kz on the branch Re>=0, Im>=0.",
        "efficiency": "R_m=|r_m|^2 Re(kz1_m)/(k0 n_f), T_m=|t_m|^2 Re(kz3_m)/(k0 n_f)"}}
    for label, P, w in (("A", 780.0, 390.0), ("B", 600.0, 300.0), ("C", 300.0, 150.0)):
        res[f"case{label}"] = run_case(label, P, w)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rcwa_results_v2.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
