"""Step 3: classical sampling cost versus depth (Table I and the headline estimate).

Combines the best-found single-amplitude contraction costs of
``contraction_cost/results/depth_sweep.json`` with the fitted fidelity of
``data/results/fidelity_vs_depth.json`` in the cost model of Eq. (3),

    W = 8 * kappa * N_s * F * C_amp   machine FLOPs,      t = W / (eta * P),

with kappa = 10, N_s = 10^6 samples, Frontier peak P = 1.685e18 FLOPS and efficiency
eta = 0.20.  Both columns assume unlimited working memory.

    python analysis/sampling_cost_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcs.estimators import SECONDS_PER_YEAR, frontier_seconds, sampling_work  # noqa: E402


def human(seconds: float) -> str:
    """Format a duration with a readable unit."""
    for unit, size in (("yr", SECONDS_PER_YEAR), ("d", 86400.0), ("h", 3600.0), ("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"


def main() -> None:
    sweep = json.loads((ROOT / "contraction_cost" / "results" / "depth_sweep.json").read_text())["searches"]
    fit = json.loads((ROOT / "data" / "results" / "fidelity_vs_depth.json").read_text())["fit"]
    fidelity = lambda d: fit["prefactor"] * fit["fidelity_per_cycle"] ** d

    rows = []
    print(f"{'cycles':>6s} {'2q gates':>9s} {'width':>6s} {'log10 C_amp':>12s} {'fitted F':>10s} {'W (FLOPs)':>11s} {'t_samp':>10s}")
    for s in sorted((s for s in sweep if s["search"] == 1), key=lambda s: s["depth"]):
        work = sampling_work(fidelity(s["depth"]), s["log10_camp"])
        t = frontier_seconds(work)
        rows.append({"cycles": s["depth"], "two_qubit_gates": s["cz"], "width": s["width"], "log10_camp": s["log10_camp"],
                     "fitted_fidelity": fidelity(s["depth"]), "work_flops": work, "frontier_seconds": t})
        print(f"{s['depth']:6d} {s['cz']:9d} {s['width']:6.0f} {s['log10_camp']:12.2f} {fidelity(s['depth']):10.2e} {work:11.2e} {human(t):>10s}")
    repeats = {s["depth"]: s["log10_camp"] for s in sweep if s["search"] == 2}
    first = {s["depth"]: s["log10_camp"] for s in sweep if s["search"] == 1}
    print("repeat searches minus first search (decades):", {d: round(repeats[d] - first[d], 2) for d in sorted(repeats)})
    anchor = next(r for r in rows if r["cycles"] == 36)
    print(f"36-cycle anchor: one amplitude = {frontier_seconds(8 * 10 ** anchor['log10_camp']) / 86400:.1f} days of the full machine; "
          f"10^6 samples = {anchor['work_flops']:.2e} FLOPs = {anchor['frontier_seconds'] / SECONDS_PER_YEAR:.0f} Frontier years")
    (ROOT / "data" / "results" / "sampling_cost_table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
