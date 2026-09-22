"""Step 1: patched XEB fidelity of every patched circuit.

For each of the 180 patched circuits (K = 3, 4; six depths; five partitions; three
instances) the script

1. loads the circuit from ``data/circuits/patched`` and splits it into its patches,
2. simulates every patch exactly (statevector, at most 21 qubits),
3. loads the measured bitstrings from ``data/counts`` and evaluates the ideal-XEB
   normalised product estimator of :func:`rcs.estimators.patched_xeb`.

The result, one record per circuit with the per-patch fidelities and ideal XEB values,
is written to ``data/results/patch_xeb.json``.  Runtime: about 15 minutes on 20 cores.

    python analysis/compute_patch_xeb.py [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import qpy  # noqa: E402

from rcs.circuits import split_patched_circuit  # noqa: E402
from rcs.estimators import patched_xeb  # noqa: E402
from rcs.io import load_shots  # noqa: E402
from rcs.layout import load_layout  # noqa: E402
from rcs.simulate import ideal_probabilities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=None, help="simulator threads (default: all cores)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "patch_xeb.json")
    args = parser.parse_args()

    layout = load_layout(ROOT / "data" / "layout.json")
    records, t0 = [], time.time()
    for k in (3, 4):
        for depth in layout.depths["patched"]:
            shots = load_shots(ROOT / "data" / "counts" / f"patched_K{k}_d{depth}.npz")
            for j, (_, patch_qubits) in enumerate(layout.partitions[k]):
                for i in range(layout.instances):
                    name = f"partition{j}_instance{i}"
                    with open(ROOT / "data" / "circuits" / "patched" / f"K{k}" / f"d{depth}" / f"{name}.qpy", "rb") as fh:
                        circuit = qpy.load(fh)[0]
                    subs, index_map = split_patched_circuit(circuit, patch_qubits)
                    probs = {p: ideal_probabilities(subs[p], threads=args.threads) for p in subs}
                    result = patched_xeb(shots[name], probs, index_map)
                    records.append({"K": k, "depth": depth, "partition": j, "instance": i, **result})
            done = [r for r in records if r["K"] == k and r["depth"] == depth]
            print(f"K={k} depth={depth}: {len(done)} circuits, mean F = {sum(r['fidelity'] for r in done) / len(done):.3e} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
