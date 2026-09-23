#!/usr/bin/env python
"""Adversarial min-cut search on the experiment's coupler graph.

For the Gao-style severing attack, the spoof score decays per cycle as
gamma * b/4 (ABCD schedule: each coupler fires once per 4 cycles), where b
is the number of couplers crossing the cut.  The attacker wants the cut
with the fewest crossing couplers subject to both sides being cheap to
simulate, so we report the best cut found as a function of the larger-side
size L (the attack cost is dominated by simulating the larger side).

Method: randomized greedy region growth (all starts x many random
repetitions, recording the best boundary at every region size) followed by
first-improvement swap refinement, plus Kernighan--Lin bisections for the
balanced point.  Results are best-found values, i.e. upper bounds on the
true min cut (the attacker could in principle do better, not worse).

Usage:
  python min_cut_62q.py --circuit "data/circuits/full/d36_logical.qpy" \
      --reps 200 --out results/q62_min_cuts.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'contraction_cost'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_sampling_cost import _load_circuit  # noqa: E402


def coupler_graph(circuit):
    """Multiplicity-weighted coupler graph from a qiskit circuit."""
    G = nx.Graph()
    qindex = {q: i for i, q in enumerate(circuit.qubits)}
    for inst in circuit.data:
        if inst.operation.num_qubits == 2:
            a, b = (qindex[q] for q in inst.qubits)
            if G.has_edge(a, b):
                G[a][b]["mult"] += 1
            else:
                G.add_edge(a, b, mult=1)
    return G


def boundary_size(G, side):
    return sum(1 for u, v in G.edges if (u in side) != (v in side))


def grow_regions(G, reps, rng):
    """Randomized greedy growth; returns best[side_size] -> (b, frozenset)."""
    nodes = list(G.nodes)
    best = {}
    for start in nodes:
        for _ in range(reps):
            region = {start}
            boundary = set(G[start])
            b = len(list(G.edges(start)))
            while len(region) < len(nodes) - 1 and boundary:
                # candidate scores: boundary change if node joined the region
                scored = []
                for u in boundary:
                    delta = sum(-1 if v in region else +1 for v in G[u])
                    scored.append((delta, rng.random(), u))
                scored.sort()
                # soft greedy: usually take the best, sometimes second-best
                pick = scored[0] if (len(scored) == 1 or rng.random() > 0.15) else scored[1]
                delta, _, u = pick
                region.add(u)
                b += delta
                boundary.discard(u)
                boundary.update(v for v in G[u] if v not in region)
                s = len(region)
                if s not in best or b < best[s][0]:
                    best[s] = (b, frozenset(region))
    return best


def refine(G, side, sweeps=20, rng=None):
    """First-improvement swaps (add one, drop one) at fixed side size."""
    side = set(side)
    b = boundary_size(G, side)
    nodes = set(G.nodes)
    for _ in range(sweeps):
        improved = False
        frontier_in = [u for u in nodes - side if any(v in side for v in G[u])]
        frontier_out = [u for u in side if any(v not in side for v in G[u])]
        if rng:
            rng.shuffle(frontier_in)
            rng.shuffle(frontier_out)
        for u in frontier_in:
            for w in frontier_out:
                if u == w:
                    continue
                trial = (side | {u}) - {w}
                bt = boundary_size(G, trial)
                if bt < b:
                    side, b = trial, bt
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return b, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--kl-seeds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=0.29,
                    help="per-severed-CZ log-XEB damage used in the summary")
    ap.add_argument("--exp-slope", type=float, default=0.1875,
                    help="experiment's measured per-cycle log-fidelity decay")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    circ = _load_circuit(Path(args.circuit))
    G = coupler_graph(circ)
    n = G.number_of_nodes()
    print(f"coupler graph: {n} qubits, {G.number_of_edges()} couplers, "
          f"total 2q gates = {sum(d['mult'] for _,_,d in G.edges(data=True))}")

    best = grow_regions(G, args.reps, rng)

    # balanced bisections via Kernighan-Lin
    for s in range(args.kl_seeds):
        A, B = nx.community.kernighan_lin_bisection(G, seed=s)
        side = min((A, B), key=len)
        b = boundary_size(G, side)
        k = len(side)
        if k not in best or b < best[k][0]:
            best[k] = (b, frozenset(side))

    rows = []
    for s in sorted(best):
        if s < 3 or s > n // 2:
            continue
        b0, side0 = best[s]
        b, side = refine(G, side0, rng=rng)
        L = n - s
        mult = sum(d["mult"] for u, v, d in G.edges(data=True)
                   if (u in side) != (v in side))
        per_cycle = b / 4.0
        slope = args.gamma * per_cycle          # spoof log-XEB decay per cycle
        rows.append({
            "small_side": s, "large_side": L, "cut_couplers": b,
            "severed_2q_gates_total": mult, "severed_per_cycle": per_cycle,
            "spoof_slope_per_cycle": slope,
            "slope_ratio_vs_experiment": slope / args.exp_slope,
            "small_side_qubits": sorted(side),
        })

    print(f"\n  s   L  cut  sev/cyc  spoof-slope  ratio-vs-exp({args.exp_slope})")
    for r in rows:
        print(f" {r['small_side']:3d} {r['large_side']:3d} {r['cut_couplers']:4d}"
              f"  {r['severed_per_cycle']:6.2f}   {r['spoof_slope_per_cycle']:9.3f}"
              f"   {r['slope_ratio_vs_experiment']:6.2f}")

    bal = min((r for r in rows if r["small_side"] >= n // 2 - 4),
              key=lambda r: r["cut_couplers"])
    print(f"\nbest near-balanced cut: {bal['small_side']}+{bal['large_side']}, "
          f"{bal['cut_couplers']} couplers -> {bal['severed_per_cycle']:.2f} severed CZ/cycle")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"circuit": args.circuit, "n_qubits": n,
                   "n_couplers": G.number_of_edges(),
                   "gamma_assumed": args.gamma,
                   "experiment_slope": args.exp_slope, "cuts": rows}
        Path(args.out).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
