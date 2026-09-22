"""Step 2: fidelity versus depth, exponential fit, anticoncentration check and Fig. 2.

Inputs
    data/counts/mirror_survival.json      mirror benchmark, 4-40 cycles
    data/results/patch_xeb.json           output of ``compute_patch_xeb.py``, 20-40 cycles

For every depth the circuit instances of an estimator are combined with an
inverse-variance weighted mean (3 instances for the mirror benchmark, 15 circuits for the
3- and 4-patch XEB).  The mirror points are fitted with a single exponential
``F(d) = A * f**d`` by unweighted least squares on ``log10 F``.  Error bars are shot noise
only.  The script also tabulates the collision ratio ``2**m * sum p**2 - 1`` of the
patches, which shows where the patch outputs anticoncentrate.

Outputs
    data/results/fidelity_vs_depth.json   points, fit, fitted fidelities, collision ratios
    figures/fig2_fidelity_vs_depth.pdf/.png

    python analysis/fidelity_vs_depth.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcs.estimators import fit_exponential_decay, inverse_variance_mean, mirror_survival  # noqa: E402

#: Mirror fit of our earlier run on the first-generation Nighthawk r1 device (ibm_miami,
#: 62 qubits), shown in Fig. 2 for comparison only: log10 F = R1_INTERCEPT + R1_SLOPE * d.
R1_INTERCEPT, R1_SLOPE = -1.0197, -0.07781
ERROR_BAR_SIGMAS = 5


def pooled_points():
    """Inverse-variance combined fidelity per estimator and depth."""
    points = {"mirror": {}, "3-patch": {}, "4-patch": {}}
    mirror = json.loads((ROOT / "data" / "counts" / "mirror_survival.json").read_text())["depths"]
    for depth, entry in mirror.items():
        p, se = mirror_survival(entry["hits"], entry["shots"])
        points["mirror"][int(depth)] = inverse_variance_mean(p, se)
    records = json.loads((ROOT / "data" / "results" / "patch_xeb.json").read_text())
    groups = defaultdict(list)
    for r in records:
        groups[(r["K"], r["depth"])].append(r)
    for (k, depth), rs in sorted(groups.items()):
        points[f"{k}-patch"][depth] = inverse_variance_mean([r["fidelity"] for r in rs], [r["se"] for r in rs])
    return points, records


def collision_ratios(records):
    """Mean collision ratio ``2**m * sum p**2 - 1`` (the ideal XEB) of the patches per K and depth."""
    acc = defaultdict(list)
    for r in records:
        acc[(r["K"], r["depth"])].extend(r["patch_ideal_xeb"].values())
    return {f"K{k}": {str(d): float(np.mean(v)) for (kk, d), v in sorted(acc.items()) if kk == k} for k in (3, 4)}


def plot(points, prefactor, per_cycle, path_stem):
    """Draw Fig. 2: the three estimators with shot-noise error bars, the mirror fit and the sampled point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = lambda d: prefactor * per_cycle ** d
    dd = np.linspace(4, 40, 300)
    fig, ax = plt.subplots(figsize=(11.3, 7.4))
    h_r1, = ax.plot(dd, 10 ** (R1_INTERCEPT + R1_SLOPE * dd), "--", color="0.5", lw=1.5,
                    label="Nighthawk r1 mirror fit (ibm_miami, 62q)")
    h_fit, = ax.plot(dd, fit(dd), "-", color="black", lw=2, label="mirror fit")
    handles = [h_r1, h_fit]
    for name, fmt, colour, shift, label, kw in (("mirror", "o", "tab:blue", 0.0, "mirror benchmark", dict(ms=7)),
                                                 ("3-patch", "D", "tab:green", 0.25, "3 patch RCS", dict(ms=6)),
                                                 ("4-patch", "x", "tab:red", 0.5, "4 patch RCS", dict(ms=7, mew=2))):
        ds = sorted(points[name])
        handles.append(ax.errorbar(np.array(ds) + shift, [points[name][d][0] for d in ds],
                                   yerr=ERROR_BAR_SIGMAS * np.array([points[name][d][1] for d in ds]),
                                   fmt=fmt, color=colour, capsize=3, lw=1.5, label=label, **kw))
    h_star, = ax.plot([36], [fit(36)], marker="*", ms=22, color="gold", markeredgecolor="black", ls="none", zorder=6,
                      label=f"full circuit, 1,000,000 samples (d=36, 918 CZ): F≈{fit(36):.1e}")
    handles.append(h_star)
    ax.set_yscale("log"); ax.set_ylim(6e-5, 1.05); ax.set_xlim(2.5, 42)
    ax.set_xlabel("Cycles", fontsize=14); ax.set_ylabel(f"Fidelity ± {ERROR_BAR_SIGMAS}σ", fontsize=14)
    ax.grid(True, which="both", color="0.85", lw=0.8); ax.tick_params(labelsize=13)
    legend = ax.legend(handles, [h.get_label() for h in handles], fontsize=11, loc="lower left")
    legend.legend_handles[-1].set_markersize(12)
    fig.suptitle("RCS on ibm_phoenix (Nighthawk r2), 61 qubits", fontsize=15); fig.tight_layout()
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf")); fig.savefig(path_stem.with_suffix(".png"), dpi=110)


def main() -> None:
    points, records = pooled_points()
    depths = sorted(points["mirror"])
    prefactor, per_cycle = fit_exponential_decay(depths, [points["mirror"][d][0] for d in depths])
    fit = lambda d: prefactor * per_cycle ** d
    num_qubits = json.loads((ROOT / "data" / "layout.json").read_text())["num_qubits"]
    out = {"points": {k: {str(d): {"fidelity": v[0], "se": v[1]} for d, v in sorted(p.items())} for k, p in points.items()},
           "fit": {"prefactor": prefactor, "fidelity_per_cycle": per_cycle,
                   "error_per_qubit_per_cycle": -math.log(per_cycle) / num_qubits,
                   "fitted_fidelity": {str(d): fit(d) for d in (20, 24, 28, 32, 36, 40)}},
           "collision_ratio": collision_ratios(records)}
    (ROOT / "data" / "results" / "fidelity_vs_depth.json").write_text(json.dumps(out, indent=1))
    plot(points, prefactor, per_cycle, ROOT / "figures" / "fig2_fidelity_vs_depth")

    print(f"mirror fit: F(d) = {prefactor:.3f} x {per_cycle:.4f}^d; error per qubit per cycle {out['fit']['error_per_qubit_per_cycle']:.2e}")
    print("fitted F at 32, 36, 40 cycles:", ", ".join(f"{fit(d):.2e}" for d in (32, 36, 40)))
    print(f"{'depth':>5s} {'mirror':>22s} {'3-patch':>22s} {'4-patch':>22s}")
    for d in depths:
        cells = [f"{points[k][d][0]:.3e} ± {points[k][d][1]:.1e}" if d in points[k] else "" for k in ("mirror", "3-patch", "4-patch")]
        print(f"{d:5d} " + " ".join(f"{c:>22s}" for c in cells))
    print("collision ratio 2^m sum p^2 - 1 (ideal XEB), mean over patches; 1 for an anticoncentrated patch:")
    for k, label in (("K3", "20/21-qubit patches"), ("K4", "15/16-qubit patches")):
        print(f"  {label}: " + ", ".join(f"d={d}: {v:.3f}" for d, v in out["collision_ratio"][k].items()))


if __name__ == "__main__":
    main()
