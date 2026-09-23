#!/usr/bin/env python3
"""Plot the cost-vs-search-time plateau curve from run_plateau.py output.

Produces two-panel figure:
  - left:  log10(FLOPs) vs cotengra wall-clock budget
  - right: contraction width vs budget
for each (memory_model, mode) curve.

Usage
-----
    python plot_plateau.py [--data plateau_data.json] [--out plateau.png]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("plateau_data.json"))
    p.add_argument("--out", type=Path, default=Path("plateau.png"))
    args = p.parse_args()

    rows = json.loads(args.data.read_text())
    rows = [r for r in rows if "error" not in r]
    if not rows:
        raise SystemExit(f"No rows in {args.data}")

    # Group by (memory, mode). Mode = 'amp' (closed) or 'psi_full' (open uncapped) or 'psi_batched' (open with K cap).
    series = defaultdict(list)  # key -> list of (wall_s, log10_flops, width)
    for r in rows:
        mem = r["memory"]
        # closed amp side
        series[(mem, "amp_closed")].append(
            (r["amp_wall_s"], r["amp_log10_flops"], r["amp_width"])
        )
        # open psi full state (uncapped)
        series[(mem, "psi_full")].append(
            (r["psi_wall_s"], r["psi_log10_flops"], r["psi_width"])
        )
        # batched (K-capped) cost — same path as psi_full, just different cost
        series[(mem, "psi_batched_K")].append(
            (r["psi_wall_s"], r["batched_log10_flops"], r["psi_width"])
        )

    # Sort each curve by wall time
    for k in series:
        series[k].sort(key=lambda row: row[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"amp_closed": "C0", "psi_full": "C1", "psi_batched_K": "C2"}
    styles = {"ideal": "--", "GPU-128GB": "-", "GPU-80GB": ":", "ALL-RAM-4TB": "-."}

    for (mem, mode), pts in sorted(series.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys_flops = [p[1] for p in pts]
        ys_w = [p[2] for p in pts]
        c = colors.get(mode, "k")
        ls = styles.get(mem, "-")
        label = f"{mode} | {mem}"
        ax1.plot(xs, ys_flops, marker="o", color=c, linestyle=ls, label=label)
        ax2.plot(xs, ys_w, marker="o", color=c, linestyle=ls, label=label)

    for ax, ylabel, title in (
        (ax1, "log10(FLOPs)",        "Contraction cost vs cotengra wall time"),
        (ax2, "contraction width", "Contraction width vs cotengra wall time"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("cotengra search budget [s] (wall clock)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Cost-vs-search-time plateau experiment\n"
        f"({sum(1 for r in rows)} rows from {args.data.name})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}", flush=True)

    # Also dump CSV alongside for re-plotting in pandas / external tools
    csv_path = args.out.with_suffix(".csv")
    keys = sorted({k for r in rows for k in r.keys()})
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
