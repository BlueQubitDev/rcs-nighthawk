"""Consistency check of the repository: circuits, data, estimator and reference results.

1. every circuit file equals a freshly generated circuit, gate for gate;
2. the shot totals of ``data/counts`` and ``data/samples`` equal ``data/qpu_usage.json``;
3. one patched circuit is re-simulated and its XEB record is compared with
   ``data/results/patch_xeb.json``;
4. the pooled fidelities and the fit recomputed from the records equal
   ``data/results/fidelity_vs_depth.json``.

    python scripts/check_release.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from qiskit import qpy  # noqa: E402

from build_circuits import all_circuits  # noqa: E402
from fidelity_vs_depth import pooled_points  # noqa: E402
from rcs.circuits import gate_signature, split_patched_circuit  # noqa: E402
from rcs.estimators import fit_exponential_decay, patched_xeb  # noqa: E402
from rcs.io import load_shots  # noqa: E402
from rcs.layout import load_layout  # noqa: E402
from rcs.simulate import ideal_probabilities  # noqa: E402


def main() -> None:
    layout = load_layout(ROOT / "data" / "layout.json")
    failures = 0

    bad = 0
    for rel, _, qc in all_circuits(layout):
        with open(ROOT / "data" / "circuits" / (rel + ".qpy"), "rb") as fh:
            bad += gate_signature(qpy.load(fh)[0]) != gate_signature(qc)
    print(f"[{'ok' if not bad else 'FAIL'}] circuits regenerated from layout.json: {bad} differences")
    failures += bad

    usage = json.loads((ROOT / "data" / "qpu_usage.json").read_text())["total_shots"]
    mirror = json.loads((ROOT / "data" / "counts" / "mirror_survival.json").read_text())["depths"]
    n_mirror = sum(int(np.sum(v["shots"])) for v in mirror.values())
    n_patched = sum(len(a) for p in (ROOT / "data" / "counts").glob("patched_*.npz") for a in load_shots(p).values())
    n_samples = len(load_shots(ROOT / "data" / "samples" / "full_d36.npz")["shots"])
    ok = (n_mirror, n_patched, n_samples) == (usage["mirror"], usage["patched"], usage["full_d36_sampling"])
    print(f"[{'ok' if ok else 'FAIL'}] shots: mirror {n_mirror:,}, patched {n_patched:,}, samples {n_samples:,}")
    failures += not ok

    records = json.loads((ROOT / "data" / "results" / "patch_xeb.json").read_text())
    ref = next(r for r in records if (r["K"], r["depth"], r["partition"], r["instance"]) == (3, 36, 0, 0))
    with open(ROOT / "data" / "circuits" / "patched" / "K3" / "d36" / "partition0_instance0.qpy", "rb") as fh:
        subs, index_map = split_patched_circuit(qpy.load(fh)[0], layout.partitions[3][0][1])
    shots = load_shots(ROOT / "data" / "counts" / "patched_K3_d36.npz")["partition0_instance0"]
    new = patched_xeb(shots, {p: ideal_probabilities(subs[p]) for p in subs}, index_map)
    ok = abs(new["fidelity"] / ref["fidelity"] - 1) < 1e-9 and abs(new["se"] / ref["se"] - 1) < 1e-9
    print(f"[{'ok' if ok else 'FAIL'}] re-simulated K=3, 36 cycles, partition 0, instance 0: F = {new['fidelity']:.4e} ± {new['se']:.1e}")
    failures += not ok

    stored = json.loads((ROOT / "data" / "results" / "fidelity_vs_depth.json").read_text())
    points, _ = pooled_points()
    depths = sorted(points["mirror"])
    prefactor, per_cycle = fit_exponential_decay(depths, [points["mirror"][d][0] for d in depths])
    ok = abs(per_cycle / stored["fit"]["fidelity_per_cycle"] - 1) < 1e-12 and all(
        abs(points[k][int(d)][0] / v["fidelity"] - 1) < 1e-12 for k in points for d, v in stored["points"][k].items())
    print(f"[{'ok' if ok else 'FAIL'}] pooled fidelities and fit: {prefactor:.3f} x {per_cycle:.4f}^d, F(36) = {prefactor * per_cycle ** 36:.2e}")
    failures += not ok

    print("release is consistent" if not failures else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
