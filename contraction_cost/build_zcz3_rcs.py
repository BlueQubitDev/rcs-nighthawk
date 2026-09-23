"""Reconstruct a Zuchongzhi-3.0-like RCS circuit (APPROXIMATE).

The Zuchongzhi 3.0 paper (arXiv:2412.11924) ships no public circuit, so this
rebuilds a circuit with the same *structural* ingredients that drive
tensor-network contraction cost:

  * 105-site lattice as 15 rows x 7 columns; the RCS task uses 83 qubits
    (we keep the 83 most-central sites, drop 22).
  * iSWAP-like two-qubit gates (contraction cost is set by the 4x4 gate shape
    and connectivity, not the specific entries).
  * four disjoint coupler patterns A,B,C,D cycled as **ABCDCDBA** (the paper's
    sequence), for 32 cycles.
  * random single-qubit u3 gates each cycle.

This is a *geometry approximation*: the exact 83-qubit subset and coupler
assignment of the real device are not public, so the resulting cost is an
estimate/upper-bound, not a faithful reproduction of Zuchongzhi 3.0.

Reference: Gao et al., "Establishing a New Benchmark in Quantum Computational
Advantage with 105-qubit Zuchongzhi 3.0 Processor," arXiv:2412.11924.
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple

from qiskit import QuantumCircuit

ZCZ3_SEQUENCE = "ABCDCDBA"  # paper's cycle pattern


def _lattice(nrows: int = 15, ncols: int = 7, n_keep: int = 83):
    coords = [(r, c) for r in range(nrows) for c in range(ncols)]
    if len(coords) > n_keep:
        cr, cc = (nrows - 1) / 2, (ncols - 1) / 2
        coords.sort(key=lambda rc: (abs(rc[0] - cr) + abs(rc[1] - cc)))
        coords = sorted(coords[:n_keep])
    q = {rc: i for i, rc in enumerate(coords)}
    edges_h = [((r, c), (r, c + 1)) for (r, c) in coords if (r, c + 1) in q]
    edges_v = [((r, c), (r + 1, c)) for (r, c) in coords if (r + 1, c) in q]

    def pairs(es):
        return [(q[a], q[b]) for a, b in es]

    # four disjoint matchings A,B,C,D (like Sycamore's ABCD)
    A = pairs([e for e in edges_h if e[0][1] % 2 == 0])
    B = pairs([e for e in edges_h if e[0][1] % 2 == 1])
    C = pairs([e for e in edges_v if e[0][0] % 2 == 0])
    D = pairs([e for e in edges_v if e[0][0] % 2 == 1])
    return coords, {"A": A, "B": B, "C": C, "D": D}


def zcz3_rcs(d: int = 32, seed: int = 0, n_qubits: int = 83,
             nrows: int = 15, ncols: int = 7) -> QuantumCircuit:
    """Build a Zuchongzhi-3.0-like iSWAP RCS at depth d (default 32 cycles)."""
    coords, sets = _lattice(nrows, ncols, n_qubits)
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for cycle in range(d):
        for q_ in range(n_qubits):
            qc.u(rng.random() * math.pi,
                 rng.random() * 2 * math.pi,
                 rng.random() * 2 * math.pi, q_)
        patt = ZCZ3_SEQUENCE[cycle % len(ZCZ3_SEQUENCE)]
        for u, v in sets[patt]:
            qc.iswap(u, v)
    return qc


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    qc = zcz3_rcs(d=d)
    from collections import Counter
    ops = Counter(g.operation.name for g in qc.data)
    n2 = sum(v for k, v in ops.items() if k == "iswap")
    print(f"ZCZ-3.0-like d={d}: n={qc.num_qubits} gates={len(qc.data)} "
          f"iswap={n2} ops={dict(ops)}")
    print(f"  (paper: 83q x 32c, ~{83*32//2} 2q gates region)")
