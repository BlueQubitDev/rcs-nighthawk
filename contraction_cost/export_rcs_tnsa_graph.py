#!/usr/bin/env python
"""Export our circuits into Google's rcs_tnsa .graph/.groups format.

Replicates scripts/circuit_to_tn.py from google-research/rcs_tnsa exactly
(node/edge construction order, fSim group pairing, open-output handling),
so their simulated-annealing optimizer can search our networks under the
same conventions used for their published Table 1. Single-qubit gates are
dropped: they are dim-2 wire tensors that do not affect connectivity,
bond dimensions, or attainable cost, and their released graphs likewise
count only entangler nodes. Every CZ is treated as a member of the fSim
family (CZ = FSim(0, pi)), which is what enables their paired-slice
groups.

Usage:
  python export_rcs_tnsa_graph.py --circuit "data/circuits_q62/q=62,d=32/0.qpy" \
      --out data/q62_tnsa/q62_m32 [--open-output]

Writes <out>.graph and <out>.groups.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_sampling_cost import _load_circuit  # noqa: E402


def export(circ, out_prefix: Path, open_output: bool):
    qindex = {q: i for i, q in enumerate(circ.qubits)}
    n = len(circ.qubits)

    # mirror circuit_to_tn.py: initial nodes 0..n-1 are the |0> tensors
    qubit_to_node = {q: q for q in range(n)}
    next_node = n
    edges = []
    edge_num = 0
    node_qe = defaultdict(list)          # node -> [(qubit, edge), ...]

    for inst in circ.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        gq = sorted(qindex[q] for q in inst.qubits)
        if len(gq) != 2:
            continue                     # drop single-qubit gates
        gate_node = next_node
        next_node += 1
        for q in gq:
            edges.append((qubit_to_node[q], gate_node))
            node_qe[gate_node].append((q, edge_num))
            node_qe[qubit_to_node[q]].append((q, edge_num))
            qubit_to_node[q] = gate_node
            edge_num += 1

    if open_output:
        for q in range(n):
            edges.append((qubit_to_node[q], -1))
            edge_num += 1                # open edges: not in any group

    # fSim pair groups, exactly as in their script
    groups = []
    for node, qe in node_qe.items():
        if node < n or len(qe) != 4:
            continue                     # initial tensors / boundary gates
        (q0, e0), (q1, e1), (q2, e2), (q3, e3) = qe
        if q0 == q2 and q1 == q3:
            groups += [(e0, e3), (e1, e2)]
        elif q0 == q3 and q1 == q2:
            groups += [(e0, e2), (e1, e3)]
        else:
            raise RuntimeError("FSim qubit configuration is inconsistent.")

    all_groups = set(range(edge_num))
    for group in groups:
        for e in group:
            all_groups.discard(e)
    all_groups = sorted((e,) for e in all_groups)
    for group in groups:
        all_groups.append(tuple(sorted(group)))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}.graph", "w") as f:
        for a, b in edges:
            f.write(f"2 {a} {b}\n")
    with open(f"{out_prefix}.groups", "w") as f:
        for group in all_groups:
            f.write(" ".join(str(e) for e in group) + "\n")

    n_pairs = len(groups)
    print(f"{out_prefix}: nodes={next_node} edges={edge_num} "
          f"(open={open_output}), fSim pair groups={n_pairs}, "
          f"singleton groups={len(all_groups)-n_pairs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--open-output", action="store_true")
    args = ap.parse_args()
    circ = _load_circuit(Path(args.circuit))
    export(circ, Path(args.out), args.open_output)


if __name__ == "__main__":
    main()
