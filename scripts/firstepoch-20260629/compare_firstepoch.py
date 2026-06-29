#!/usr/bin/env python3
"""
Read-only post-processing for the first-epoch (beta=8) OLD-vs-NEW divergence
study (quest tim-130116-82977). Reads the pulled run dirs and emits:
  1. g-vs-iteration overlay (raw + normalized) for all 3 arms.
  2. iter-1 decomposition (raw g, ||grad||, cosine) for OLD vs NEW-nlopt.
Handles partial/missing data so it can be run while jobs are still going.

Usage:
  python3 compare_firstepoch.py --results <pulled_results_dir> --outdir <figdir>
"""
import argparse, csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("arm-old",        "OLD (Emitter3D, NLopt CCSAQ)", "tab:blue"),
    ("arm-new-nlopt",  "NEW :nlopt (DEO, NLopt CCSAQ)", "tab:orange"),
    ("arm-new-ccsaq",  "NEW :standalone_ccsaq (DEO)",   "tab:green"),
]


def read_history(path):
    """Return dict of lists. Handles both schemas (manual arms vs ccsaq)."""
    if not os.path.isfile(path):
        return None
    cols = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
    return cols


def parse_decomp(path):
    d = {}
    if not os.path.isfile(path):
        return d
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                try:
                    d[k] = float(v.strip())
                except ValueError:
                    pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stamp", default="20260629")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hist = {}
    for d, _, _ in ARMS:
        hist[d] = read_history(os.path.join(args.results, d, "history.csv"))

    # ---- Figure 1: normalized g vs iteration --------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for d, label, color in ARMS:
        h = hist.get(d)
        if not h or "iter" not in h or "g" not in h:
            continue
        ax.plot(h["iter"], h["g"], "-o", ms=3, color=color, label=label)
        plotted = True
    ax.set_xlabel("iteration (objective evaluation)")
    ax.set_ylabel("normalized objective  g = g_raw / g_norm")
    ax.set_title("First epoch (beta=8) from uniform-0.5 — normalized g")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    f1 = os.path.join(args.outdir, f"scratch-firstepoch-gnorm-vs-iter-{args.stamp}.png")
    fig.tight_layout(); fig.savefig(f1, dpi=130); plt.close(fig)
    print("wrote", f1)

    # ---- Figure 2: raw g vs iteration (manual arms have it per-eval) ---------
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for d, label, color in ARMS:
        h = hist.get(d)
        if not h or "g_raw" not in h:
            continue
        ax.plot(h["iter"], h["g_raw"], "-o", ms=3, color=color, label=label)
        plotted = True
    ax.set_xlabel("iteration (objective evaluation)")
    ax.set_ylabel("raw objective  g_raw")
    ax.set_title("First epoch (beta=8) from uniform-0.5 — raw g")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    f2 = os.path.join(args.outdir, f"scratch-firstepoch-graw-vs-iter-{args.stamp}.png")
    fig.tight_layout(); fig.savefig(f2, dpi=130); plt.close(fig)
    print("wrote", f2)

    # ---- Figure 3: iter-1 decomposition OLD vs NEW --------------------------
    dec = parse_decomp(os.path.join(args.results, "meshgen", "decomp.txt"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if dec:
        labels = ["OLD", "NEW"]
        graw = [dec.get("g_raw_old", float("nan")), dec.get("g_raw_new", float("nan"))]
        gnrm = [dec.get("||grad_old||", float("nan")), dec.get("||grad_new||", float("nan"))]
        axes[0].bar(labels, graw, color=["tab:blue", "tab:orange"])
        axes[0].set_title("raw g at uniform-0.5 (iter 1)")
        axes[0].set_ylabel("g_raw")
        axes[1].bar(labels, gnrm, color=["tab:blue", "tab:orange"])
        axes[1].set_title("||grad|| at uniform-0.5 (iter 1)")
        txt = (f"g_raw ratio NEW/OLD = {dec.get('g_raw ratio n/o', float('nan')):.4g}\n"
               f"cos(grad_o,grad_n) = {dec.get('cos(grad_o,grad_n)', float('nan')):.6g}\n"
               f"rel||dgrad||/||o|| = {dec.get('rel||dgrad||/||o||', float('nan')):.4g}\n"
               f"g_norm ratio NEW/OLD = {dec.get('g_norm ratio n/o', float('nan')):.4g}")
        fig.text(0.5, -0.02, txt, ha="center", fontsize=9, family="monospace")
    else:
        fig.text(0.5, 0.5, "decomp.txt not found (meshgen job not finished)",
                 ha="center", va="center")
    fig.suptitle("iter-1 decomposition: same mesh + p=0.5 + NLopt -> any diff is ASSEMBLY")
    f3 = os.path.join(args.outdir, f"scratch-firstepoch-iter1-decomp-{args.stamp}.png")
    fig.tight_layout(); fig.savefig(f3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", f3)


if __name__ == "__main__":
    sys.exit(main())
