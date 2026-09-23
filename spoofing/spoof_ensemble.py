#!/usr/bin/env python
"""Ensemble-averaged one-sided (and arm) spoof scores for low-degree modes.

Single-instance exact scores for tiny subsystems are buried in marginal
speckle (chi fluctuates ~ 2^{-(m-a)/2} per instance, ~1e-3 at m=24), so the
operationally relevant quantity -- the ensemble MEAN, which is what survives
at n=62 where the speckle is ~1e-9 -- needs averaging over many circuit
instances.  We generate instances of the experiment's own ensemble (Haar
SU(2) on every qubit each cycle + CZ on the ABCD colour classes extracted
from the reference circuit) restricted to an m-qubit region, and measure
exactly, per instance and depth:

  * singleton one-sided scores  chi_1(v) = 2 <p_sev(v), marg_v(p)> - 1
    for every region vertex whose full-graph degree is <= 3 and whose
    region degree equals it (true pendant/edge modes of the device graph);
  * doubleton scores for adjacent pairs of such vertices;
  * the region's small-arm mode: one-sided and two-sided;
  * the Porter-Thomas collision ratio (anticoncentration check).

Axis convention: local qubit k <-> ndarray axis k (no endianness anywhere);
the arm occupies axes 0..a-1 so the two-sided score is a plain reshape.

Usage:
  python spoof_ensemble.py --circuit "data/circuits/full/d36_logical.qpy" \
      --region-center 3 --region-size 20 --instances 60 \
      --depths 4,8,12,16,20,24,28,32,36,40 --out results/ens_c3_m20.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'contraction_cost'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_sampling_cost import _load_circuit          # noqa: E402
from min_cut import coupler_graph, grow_regions, refine  # noqa: E402


# ------------------------------------------------------------- ensemble def

def colour_classes(circ):
    """The four ABCD coupler classes, as lists of global-index pairs."""
    from qiskit.converters import circuit_to_dag

    qindex = {q: i for i, q in enumerate(circ.qubits)}
    layers = []
    for layer in circuit_to_dag(circ).layers():
        pairs = [tuple(sorted(qindex[q] for q in nd.qargs))
                 for nd in layer["graph"].op_nodes()
                 if nd.op.num_qubits == 2]
        if pairs:
            layers.append(sorted(pairs))
    assert len(layers) >= 4, "reference circuit shallower than one sweep"
    colours = layers[:4]
    for t, lay in enumerate(layers):
        if lay != colours[t % 4]:
            print(f"warning: layer {t} differs from colour {t % 4}")
    return colours


def haar_su2(rng):
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    return (q * (np.diagonal(r) / np.abs(np.diagonal(r)))).astype(np.complex64)


class State:
    """Statevector on axes = local qubits."""

    def __init__(self, m):
        self.m = m
        self.psi = np.zeros((2,) * m, dtype=np.complex64)
        self.psi[(0,) * m] = 1.0

    def u1(self, U, k):
        self.psi = np.moveaxis(np.tensordot(U, self.psi, axes=([1], [k])), 0, k)

    def cz(self, i, j):
        idx = [slice(None)] * self.m
        idx[i] = 1
        idx[j] = 1
        self.psi[tuple(idx)] *= -1

    def probs(self):
        return (np.abs(self.psi) ** 2).astype(np.float64)


def axis_selftest():
    """Marginal of axis k must match that qubit's own 1q state."""
    rng = np.random.default_rng(0)
    st = State(4)
    us = [haar_su2(rng) for _ in range(4)]
    for k, u in enumerate(us):
        st.u1(u, k)
    p = st.probs()
    for k, u in enumerate(us):
        marg = p.sum(axis=tuple(a for a in range(4) if a != k))
        pk = np.abs(u[:, 0]) ** 2
        assert np.allclose(marg, pk, atol=1e-6), f"axis map broken at {k}"


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="data/circuits/full/d36_logical.qpy")
    ap.add_argument("--region-center", type=int, default=None)
    ap.add_argument("--region-size", type=int, default=20)
    ap.add_argument("--arm", default=None,
                    help="JSON list of global qubit indices for the arm mode "
                         "(a true small side of the FULL graph); overrides "
                         "the region-internal arm search")
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--depths", default="4,8,12,16,20,24,28,32,36,40")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    axis_selftest()
    rng = np.random.default_rng(args.seed)
    depths = sorted(int(d) for d in args.depths.split(","))
    ref = _load_circuit(Path(args.circuit))
    G = coupler_graph(ref)
    colours = colour_classes(ref)

    # ---- region: BFS ball around the centre (or min-boundary region)
    if args.region_center is not None:
        order = [args.region_center]
        seen = {args.region_center}
        frontier = [args.region_center]
        while len(order) < args.region_size and frontier:
            nxt = []
            for u in frontier:
                for v in sorted(G[u], key=lambda w: -G.degree(w)):
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            take = min(len(nxt), args.region_size - len(order))
            order += nxt[:take]
            frontier = nxt
        region = sorted(order)
    else:
        import random as pyrandom
        grown = grow_regions(G, reps=30, rng=pyrandom.Random(args.seed))
        region = sorted(grown[args.region_size][1])
    Gr = G.subgraph(region).copy()
    m = len(region)

    # ---- modes
    true_low = [v for v in region
                if G.degree(v) <= 3 and Gr.degree(v) == G.degree(v)]
    pairs = [(u, v) for u in true_low for v in true_low
             if u < v and Gr.has_edge(u, v)]
    import random as pyrandom
    prng = pyrandom.Random(args.seed)
    arm = None
    if args.arm:
        arm = sorted(json.loads(args.arm))
        assert all(v in region for v in arm), "arm must lie inside the region"
        b_full = sum(1 for u, v in G.edges if (u in arm) != (v in arm))
        b_arm = sum(1 for u, v in Gr.edges if (u in arm) != (v in arm))
        assert b_arm == b_full, (
            f"arm boundary truncated by region ({b_arm} != {b_full}); "
            "enlarge the region to contain the arm's full neighbourhood")
    else:
        grown_r = grow_regions(Gr, reps=30, rng=prng)
        cand = [(b, s, side) for s, (b, side) in grown_r.items() if 3 <= s <= 8]
        if cand:
            b_arm, _, side = min(cand)
            b_arm, side = refine(Gr, side, rng=prng)
            arm = sorted(side)
    print(f"region m={m} (centre {args.region_center}): "
          f"{Gr.number_of_edges()} internal couplers")
    print(f"true low-degree modes: {[(v, G.degree(v)) for v in true_low]}")
    print(f"pairs: {pairs};  arm: {arm} (b={b_arm if arm else '-'})")

    # local layout: arm qubits first (axes 0..a-1)
    local_order = (arm or []) + [v for v in region if not arm or v not in arm]
    loc = {v: k for k, v in enumerate(local_order)}
    a = len(arm) if arm else 0
    colours_local = [[(loc[u], loc[v]) for u, v in col
                      if u in loc and v in loc] for col in colours]
    arm_axes = list(range(a))
    comp_axes = list(range(a, m))

    # ---- accumulators: key -> depth -> list of chi
    acc: dict[str, dict[int, list[float]]] = {}

    def push(key, d, val):
        acc.setdefault(key, {}).setdefault(d, []).append(float(val))

    t0 = time.time()
    for r in range(args.instances):
        st = State(m)
        u_single = {k: np.eye(2, dtype=np.complex64) for k in range(m)}
        st_arm = State(a) if arm else None
        st_comp = State(m - a) if arm else None
        pair_states = {pr: np.zeros((2, 2), dtype=np.complex64) for pr in pairs}
        for pr in pairs:
            pair_states[pr][0, 0] = 1.0

        for cyc in range(max(depths)):
            gates = {k: haar_su2(rng) for k in range(m)}
            for k, u in gates.items():
                st.u1(u, k)
                u_single[k] = u @ u_single[k]
            for (u, v) in pairs:
                psi2 = pair_states[(u, v)]
                psi2 = np.einsum("ab,bc->ac", gates[loc[u]], psi2)      # qubit u = axis0
                psi2 = np.einsum("cb,ab->ac", gates[loc[v]], psi2)      # qubit v = axis1
                pair_states[(u, v)] = psi2
            if arm:
                for k, u in gates.items():
                    if k < a:
                        st_arm.u1(u, k)
                    else:
                        st_comp.u1(u, k - a)
            for (i, j) in colours_local[cyc % 4]:
                st.cz(i, j)
                for (u, v) in pairs:
                    if {i, j} == {loc[u], loc[v]}:
                        pair_states[(u, v)][1, 1] *= -1
                if arm:
                    if i < a and j < a:
                        st_arm.cz(i, j)
                    elif i >= a and j >= a:
                        st_comp.cz(i - a, j - a)
            d = cyc + 1
            if d in depths:
                p = st.probs()
                push("pt_ratio", d, p.size * float((p * p).sum()) - 1.0)
                for v in true_low:
                    k = loc[v]
                    marg = p.sum(axis=tuple(x for x in range(m) if x != k))
                    psev = np.abs(u_single[k][:, 0]) ** 2
                    push(f"single_v{v}_b{G.degree(v)}", d,
                         2.0 * float(psev @ marg) - 1.0)
                for (u, v) in pairs:
                    ku, kv = loc[u], loc[v]
                    M = p.sum(axis=tuple(x for x in range(m) if x not in (ku, kv)))
                    if ku > kv:
                        M = M.T                    # ensure axis0 = qubit u
                    p2 = np.abs(pair_states[(u, v)]) ** 2
                    bb = G.degree(u) + G.degree(v) - 2
                    push(f"pair_v{u}v{v}_b{bb}", d,
                         4.0 * float((p2 * M).sum()) - 1.0)
                if arm:
                    P = p.reshape(2 ** a, -1)
                    parm = st_arm.probs().ravel()
                    pcomp = st_comp.probs().ravel()
                    marg_arm = P.sum(axis=1)
                    push(f"arm1_b{b_arm}", d,
                         2.0 ** a * float(parm @ marg_arm) - 1.0)
                    push(f"arm2_b{b_arm}", d,
                         p.size * float(parm @ (P @ pcomp)) - 1.0)
        if (r + 1) % 10 == 0:
            print(f"  instance {r+1}/{args.instances} [{time.time()-t0:.0f}s]")

    # ---- summary
    summary = {}
    for key, per_d in sorted(acc.items()):
        row = {}
        for d, vals in sorted(per_d.items()):
            v = np.array(vals)
            row[d] = {"mean": float(v.mean()),
                      "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                      "n": len(v)}
        summary[key] = row
        line = "  ".join(f"d{d}:{r['mean']:+.2e}({r['sem']:.0e})"
                         for d, r in sorted(row.items()) if d >= 16)
        print(f"{key:>22s}  {line}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"region": region, "centre": args.region_center,
             "true_low_modes": [(v, G.degree(v)) for v in true_low],
             "arm": arm, "instances": args.instances,
             "summary": summary}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
