#!/usr/bin/env python
"""Collect rcs_tnsa SA-walker outputs into a per-config summary.

Reads results/tnsa/<tag>_s<seed>.out files, takes the best (minimum)
cost per config tag, and converts to the paper's conventions. Units of
the printed "Cost (Log2(FLOPs))" differ by mode in the released driver
(verified against the source and by replaying a mode-1 ordering through
cotengra, which agreed to 3 decimals):

  mode 1: printed c IS log2 of complex multiply-adds (no factor 8):
      log10 complex ops = c * log10(2)
  modes 2/3: main.cpp multiplies the LOG by SCALAR_FACTOR=8 (a print
      units bug; optimization itself is consistent), so the true batch
      cost is 2^(c/8) complex multiply-adds for NSPARSE outputs:
      t_samp = (kappa * N_s * F / NSPARSE) * 8 * 2^(c/8) / (eta * P)

Usage: python parse_tnsa_results.py [--dir results/tnsa] [--f-xeb 2.6e-4]
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

KAPPA, NSAMP, NSPARSE = 10, 1_000_000, 1000
ETA_P = 0.20 * 1.685e18
YEAR = 3.15576e7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tnsa")
    ap.add_argument("--f-xeb", type=float, default=2.6e-4)
    args = ap.parse_args()

    best = defaultdict(lambda: (float("inf"), None))
    n_files = defaultdict(int)
    for f in Path(args.dir).glob("*_s*.out"):
        tag = re.sub(r"_s\d+\.out$", "", f.name)
        n_files[tag] += 1
        m = re.search(r"Cost \(Log2\(FLOPs\)\):\s*([\d.eE+-]+)", f.read_text())
        if m:
            c = float(m.group(1))
            if c < best[tag][0]:
                best[tag] = (c, f.name)

    print(f"{'config':>10s} {'walkers':>8s} {'best log2':>10s}  conversion")
    for tag in sorted(best):
        c, src = best[tag]
        if c == float("inf"):
            print(f"{tag:>10s} {n_files[tag]:8d}   (no parsed cost)")
            continue
        if tag.startswith("m1"):
            log10_complex = c * math.log10(2)
            note = (f"log10 C_amp = {log10_complex:.2f} complex "
                    f"(cotengra converged ref: 22.01)")
        else:
            t = (KAPPA * NSAMP * args.f_xeb / NSPARSE) * 8 * 2.0 ** (c / 8) / ETA_P
            unit = ("s", 1) if t < 3600 else (
                   ("h", 3600) if t < 86400 * 30 else (
                   ("d", 86400) if t < YEAR else ("yr", YEAR)))
            note = (f"t_samp = {t/unit[1]:.2f} {unit[0]} at F={args.f_xeb} "
                    f"(Eq.-3 baseline: 20.0 yr)")
        print(f"{tag:>10s} {n_files[tag]:8d} {c:10.2f}  {note}  [{src}]")


if __name__ == "__main__":
    main()
