"""Figures for the PML validation page. Usage: python3 make_figures.py results.json [more.json ...] outdir"""
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark theme, thin rules.
NIGHT, PARCH, MUTED, LINE = "#15171e", "#f7f0ec", "#cbbab5", "#59789f"
C_REF, C_PML, C_PML2, C_ABC = PARCH, "#B3AE29", "#7F8330", "#DF8A4D"
C_LEG, C_LEG2 = "#E34F83", "#E9ABAE"
plt.rcParams.update({
    "figure.facecolor": NIGHT, "axes.facecolor": NIGHT, "savefig.facecolor": NIGHT,
    "axes.edgecolor": LINE, "axes.labelcolor": PARCH, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": PARCH, "axes.grid": True, "grid.color": LINE, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "axes.linewidth": 0.6,
    "font.family": "sans-serif",
})

*inputs, out = sys.argv[1:]
os.makedirs(out, exist_ok=True)
D = {}
for f in inputs:
    D.update(json.load(open(f)))


def cplx(v):
    return np.array([x["re"] + 1j * x["im"] for x in v])


def save(fig, name):
    fig.savefig(os.path.join(out, name), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(out, name.replace(".svg", ".png")), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── F1: flat stacks vs exact ────────────────────────────────────────────────
flat = D["test1_flat.jld2"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for ax, (name, r) in zip(axes, flat.items()):
    z = np.array(r["z"]); ex = cplx(r["exact"]); st = r["stack"]
    for lo, hi, lab in ((st["z_min"], st["des_low"], "substrate"), (st["des_low"], st["des_high"], "slab"),
                        (st["des_high"], st["z_max"], "water")):
        ax.axvspan(lo, hi, color=LINE, alpha={"substrate": 0.10, "slab": 0.35, "water": 0.0}[lab], lw=0)
    ax.axvline(st["z_src"], color=MUTED, lw=0.6, ls=":")
    runs = r["runs"]
    ax.plot(z, abs(ex), color=C_REF, lw=2.2, label="exact (transfer matrix)")
    ax.plot(z, abs(cplx(runs["new PML d=400 R=1.0e-8"]["Ey"])), color=C_PML, lw=1.2, ls="--", label="new PML d=400")
    ax.plot(z, abs(cplx(runs["ABC (no PML)"]["Ey"])), color=C_ABC, lw=1.0, ls=":", label="ABC")
    ax.plot(z, abs(cplx(runs["legacy 2023 PML as written"]["Ey"])), color=C_LEG, lw=1.2, label="legacy 2023, as written")
    ax.plot(z, abs(cplx(runs["legacy, sign flipped"]["Ey"])), color=C_LEG2, lw=1.0, ls="-.", label="legacy, sign flipped")
    ax.set_title(name)
    ax.set_xlabel("z (nm)   (source at dotted line; PML lies outside this range)")
    ax.set_ylabel("|Ey|")
axes[0].legend(fontsize=8, loc="upper left")
save(fig, "f1_flat.svg")

# ── F2: lamellar efficiencies vs RCWA ───────────────────────────────────────
lam = D.get("test2_lamellar_h20_o1.jld2") or D[sorted(k for k in D if k.startswith("test2_lamellar"))[0]]
RCWA = {"A": {"R0": 5.68524e-3, "R1": 2.07865e-3, "T0": 0.675388, "T1": 0.153256, "T2": 4.12868e-3},
        "B": {"R0": 9.50397e-3, "R1": 3.73818e-4, "T0": 0.715329, "T1": 0.137210},
        "C": {"R0": 1.30389e-2, "T0": 0.986961}}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [5, 4, 2]})
for ax, case, title in zip(axes, "ABC", ("A · P=780 nm (orders up to 70°)", "B · P=600 nm", "C · P=300 nm (no diffraction)")):
    keys = list(RCWA[case])
    x = np.arange(len(keys))
    for i, (lab, col) in enumerate((("ABC", C_ABC), ("PML d=400", C_PML), ("PML d=800", C_PML2))):
        err = [abs(lam[case][lab][k] / RCWA[case][k] - 1) for k in keys]
        ax.bar(x + (i - 1) * 0.27, err, 0.27, color=col, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 20)
    ax.set_xticks(x, [k.replace("R", "R").replace("T", "T") for k in keys])
    ax.set_title(title)
axes[0].set_ylabel("|FEM / RCWA − 1| per order")
axes[0].legend(fontsize=8, loc="upper right")
save(fig, "f2_lamellar_vs_rcwa.svg")

# ── F3: wrong-way wave per order (lamellar) vs ABC theory ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
theta = {"A": {"backS1": 28.0590, "backS2": 70.1796, "backT1": 30.8519, "backF1": 30.8519},
         "B": {"backS1": 37.6976, "backT1": 41.8103, "backF1": 41.8103}}
lab_of = {"backS1": "silica m=1", "backS2": "silica m=2", "backT1": "water m=1 (above src)",
          "backF1": "water m=1 (below src)"}
for ax, case in zip(axes, "AB"):
    keys = list(theta[case]); x = np.arange(len(keys))
    pred = [abs(1 - np.cos(np.radians(theta[case][k]))) / (1 + np.cos(np.radians(theta[case][k]))) for k in keys]
    ax.bar(x - 0.27, [lam[case]["ABC"][k] for k in keys], 0.27, color=C_ABC, label="ABC, measured")
    ax.scatter(x - 0.27, pred, marker="_", s=400, color=C_REF, zorder=3, label="ABC, theory |1−cosθ|/(1+cosθ)")
    ax.bar(x, [lam[case]["PML d=400"][k] for k in keys], 0.27, color=C_PML, label="PML d=400")
    ax.bar(x + 0.27, [lam[case]["PML d=800"][k] for k in keys], 0.27, color=C_PML2, label="PML d=800")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_xticks(x, [f"{lab_of[k]}\n{theta[case][k]:.1f}°" for k in keys], fontsize=8)
    ax.set_title(f"case {case}: reflected amplitude / outgoing amplitude")
axes[0].legend(fontsize=8, loc="upper left")
save(fig, "f3_backwave.svg")

# ── F4: field slices, case A ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, lab in zip(axes, ("ABC", "PML d=400")):
    r = lam["A"][lab]
    xs = np.array(r["slice_x"]); zs = np.array(r["slice_z"])
    F = np.array([[c["re"] + 1j * c["im"] for c in col] for col in r["slice_Ey"]])  # [z][x]
    full = np.concatenate([F[:, ::-1], F], axis=1)   # mirror unfold: one period
    xf = np.concatenate([-xs[::-1] - r["stack"]["z_min"] * 0 - 390 / 1, xs]) if False else np.linspace(-390, 390, full.shape[1])
    im = ax.pcolormesh(xf, zs, np.real(full), cmap="twilight_shifted", vmin=-0.6, vmax=0.6, shading="auto")
    st = r["stack"]
    for zz in (st["des_low"], st["des_high"]):
        ax.axhline(zz, color=PARCH, lw=0.5, alpha=0.6)
    ax.axhline(st["z_src"], color=MUTED, lw=0.6, ls=":")
    if lab.startswith("PML"):
        ax.axhspan(zs[0], st["z_min"], color=C_PML, alpha=0.12, lw=0)
        ax.axhspan(st["z_max"], zs[-1], color=C_PML, alpha=0.12, lw=0)
    ax.set_title(f"Re Ey, case A, {lab}")
    ax.set_xlabel("x (nm), one period")
axes[0].set_ylabel("z (nm)")
fig.colorbar(im, ax=axes, shrink=0.8)
save(fig, "f4_fieldA.svg")

# ── F5: Raman metric ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.2))
rows = [(f"lamellar {c}", lam[c]) for c in "CBA"]
if "test3_pillars_P300_h25.jld2" in D:
    p = D["test3_pillars_P300_h25.jld2"]
    rows.append(("pillars P=300", {"ABC": p["ABC"], "PML d=400": p["PML d=400 R=1e-8"], "PML d=800": p["PML d=800 R=1e-8"]}))
for k in [k for k in D if k.startswith("test3_pillars_P600")]:
    p = D[k]
    rows.append(("pillars P=600", {"ABC": p["ABC"], "PML d=400": p["PML d=400 R=1e-8"], "PML d=800": p["PML d=800 R=1e-8"]}))
x = np.arange(len(rows))
for i, (lab, col) in enumerate((("ABC", C_ABC), ("PML d=400", C_PML))):
    ax.bar(x + (i - 0.5) * 0.35, [abs(r[lab]["raman"] / r["PML d=800"]["raman"] - 1) for _, r in rows], 0.35, color=col, label=f"{lab} vs PML d=800")
ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
ax.set_ylabel("relative change in ∫|E|⁴ (Raman region)")
ax.legend(fontsize=8)
save(fig, "f5_raman.svg")
print("figures in", out)
