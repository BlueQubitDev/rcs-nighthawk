#!/usr/bin/env python
"""Exact severing-attack (Gao-style) XEB score on sub-regions of the q62 circuits.

The attack: partition the target circuit's qubits into (A, B), delete the
two-qubit gates crossing the cut, sample from the product distribution
p_A x p_B, and ask what linear XEB those samples achieve against the
*uncut* target.  For a target on m qubits this score is computed EXACTLY
(no sampling noise) as

    chi_spoof = 2^m * sum_x  [p_A(x_A) p_B(x_B)] * p_target(x)  -  1,

a vector-matrix-vector contraction of the target probability tensor with
the two part distributions.  Direct evaluation needs the target
statevector, so m <= ~28 on a 16 GB laptop and ~34 on a 256 GB node; the
62-qubit attack score is then predicted by transferring the fitted
per-severed-CZ damage rate gamma to the 62-qubit min-cut geometry
(min_cut_62q.py), exactly as in Gao et al.'s own large-n analysis.

Targets are m-qubit restrictions of the experiment's own circuits (gates
internal to a compact region of the coupler graph), so the ensemble,
gate set, and ABCD schedule are the experiment's.  Bonus diagnostics per
target: the Porter-Thomas collision ratio 2^m sum_x p^2 - 1 (-> 1 iff
anticoncentrated) and the exact severed-total 2q-gate count for the cut.

Usage (smoke test):
  python spoof_attack.py --circuit-dir data/circuits_q62 --depths 8,16,24,32 \
      --region-size 20 --cuts balanced --out results/spoof_m20.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'contraction_cost'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_sampling_cost import _load_circuit          # noqa: E402
from min_cut import coupler_graph, grow_regions, refine, boundary_size  # noqa: E402


# ----------------------------------------------------------------- circuits

def restrict_circuit(circ, region_order):
    """Restriction of `circ` to the qubits in region_order (global indices).

    Gates with any support outside the region are dropped (severed).  The
    returned qiskit circuit acts on len(region_order) qubits, with local
    qubit i = region_order[i].
    """
    from qiskit import QuantumCircuit

    pos = {g: i for i, g in enumerate(region_order)}
    qindex = {q: i for i, q in enumerate(circ.qubits)}
    sub = QuantumCircuit(len(region_order))
    kept2q = 0
    for inst in circ.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        gq = [qindex[q] for q in inst.qubits]
        if all(g in pos for g in gq):
            sub.append(inst.operation, [pos[g] for g in gq])
            if len(gq) == 2:
                kept2q += 1
    return sub, kept2q


def probabilities(qc):
    """|psi|^2 of a (measure-free) qiskit circuit, little-endian, float64."""
    from qiskit.quantum_info import Statevector

    psi = Statevector.from_instruction(qc).data
    p = np.abs(psi) ** 2
    del psi
    return p


# ----------------------------------------------------------------- attack

def exact_chis(p_target, p_a, p_b):
    """Exact XEB of three product-attack variants against the uncut target.

    Little-endian: A occupies the low bits, B the high bits.  Returns
      chi_ab    : q = pA x pB      (both severed sides sampled exactly)
      chi_a_uni : q = pA x uniform (only side A simulated -- the cheap attack)
      chi_b_uni : q = uniform x pB
    """
    P = p_target.reshape(p_b.size, p_a.size)      # axis0 = B (high bits)
    marg_a = P.sum(axis=0)                        # target marginal on A
    marg_b = P.sum(axis=1)
    chi_ab = float(p_a.size * p_b.size * (p_b @ (P @ p_a)) - 1.0)
    chi_a_uni = float(p_a.size * (p_a @ marg_a) - 1.0)
    chi_b_uni = float(p_b.size * (p_b @ marg_b) - 1.0)
    return chi_ab, chi_a_uni, chi_b_uni


def collision_ratio(p):
    """2^m sum p^2 - 1: -> 1 for Porter-Thomas (anticoncentration check)."""
    return float(p.size * np.dot(p, p) - 1.0)


def cuts_for_region(Gr, kind, rng):
    """Return list of (label, sideA_set) cuts of the region graph."""
    nodes = list(Gr.nodes)
    out = []
    if kind in ("balanced", "all"):
        best = None
        for s in range(40):
            A, B = nx.community.kernighan_lin_bisection(Gr, seed=s)
            side = min((A, B), key=len)
            b = boundary_size(Gr, side)
            if best is None or b < best[0]:
                best = (b, side)
        b, side = refine(Gr, best[1], rng=rng)
        out.append((f"balanced_b{b}", set(side)))
    if kind in ("min", "all"):
        grown = grow_regions(Gr, reps=30, rng=rng)
        # global best cut over all side sizes 3..m/2 (adversary's choice)
        cand = [(b, s, side) for s, (b, side) in grown.items()
                if 3 <= s <= len(nodes) // 2]
        b, s, side = min(cand)
        b, side = refine(Gr, side, rng=rng)
        out.append((f"mincut_s{len(side)}_b{b}", set(side)))
    if kind in ("arm", "all"):
        # smallest side with the fewest couplers per severed qubit (pendant)
        grown = grow_regions(Gr, reps=30, rng=rng)
        cand = [(b / max(s, 1), b, s, side) for s, (b, side) in grown.items()
                if 2 <= s <= 8]
        if cand:
            _, b, s, side = min(cand)
            b, side = refine(Gr, side, rng=rng)
            out.append((f"arm_s{len(side)}_b{b}", set(side)))
    # dedupe by frozenset
    seen, uniq = set(), []
    for lbl, side in out:
        f = frozenset(side)
        if f not in seen:
            seen.add(f)
            uniq.append((lbl, side))
    return uniq


# ----------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit-dir", default="data/circuits_q62")
    ap.add_argument("--pattern", default="q=62,d={d}/0.qpy")
    ap.add_argument("--depths", default="8,12,16,20,24,28,32")
    ap.add_argument("--region-size", type=int, default=20)
    ap.add_argument("--region", default=None,
                    help="JSON list of global qubit indices (overrides size)")
    ap.add_argument("--cuts", default="all",
                    choices=["balanced", "min", "arm", "all"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    depths = [int(d) for d in args.depths.split(",")]

    # region and its induced coupler graph, chosen on the deepest circuit
    ref = _load_circuit(Path(args.circuit_dir) / args.pattern.format(d=max(depths)))
    G = coupler_graph(ref)
    if args.region:
        region = sorted(json.loads(args.region))
    else:
        grown = grow_regions(G, reps=30, rng=rng)
        region = sorted(grown[args.region_size][1])
    Gr = G.subgraph(region).copy()
    print(f"region: m={len(region)} qubits, {Gr.number_of_edges()} internal couplers")

    cuts = cuts_for_region(Gr, args.cuts, rng)
    for lbl, side in cuts:
        print(f"  cut {lbl}: |A|={len(side)}, couplers={boundary_size(Gr, side)}")

    records = []
    for d in depths:
        circ = _load_circuit(Path(args.circuit_dir) / args.pattern.format(d=d))
        t0 = time.time()
        for lbl, sideA in cuts:
            A = sorted(sideA)
            B = sorted(set(region) - sideA)
            order = A + B                      # local qubits: A = low bits
            tgt, k_t = restrict_circuit(circ, order)
            ca, k_a = restrict_circuit(circ, A)
            cb, k_b = restrict_circuit(circ, B)
            p_t = probabilities(tgt)
            p_a = probabilities(ca)
            p_b = probabilities(cb)
            chi, chi_au, chi_bu = exact_chis(p_t, p_a, p_b)
            pt_ratio = collision_ratio(p_t)
            severed = k_t - k_a - k_b          # 2q gates crossing the cut
            b = boundary_size(Gr, sideA)
            rec = {"depth": d, "cut": lbl, "m": len(region),
                   "cut_couplers": b, "severed_2q_gates": severed,
                   "severed_per_cycle": severed / d,
                   "chi_spoof": chi, "chi_a_uniform": chi_au,
                   "chi_b_uniform": chi_bu, "pt_collision_ratio": pt_ratio}
            records.append(rec)
            print(f"d={d:2d} {lbl:>18s}: chi_spoof={chi: .3e}  "
                  f"chiAuni={chi_au: .2e}  chiBuni={chi_bu: .2e}  "
                  f"sev/cyc={severed/d:.2f}  PT={pt_ratio:.3f}  "
                  f"[{time.time()-t0:.1f}s]")

    # per-cut exponential fits: log chi = c0 - slope*d  (use chi>0 points)
    fits = []
    for lbl in sorted({r["cut"] for r in records}):
        for key in ("chi_spoof", "chi_a_uniform", "chi_b_uniform"):
            pts = [(r["depth"], r[key], r["severed_per_cycle"])
                   for r in records if r["cut"] == lbl and r[key] > 0]
            if len(pts) >= 3:
                ds = np.array([p[0] for p in pts], float)
                ln = np.log([p[1] for p in pts])
                slope, c0 = np.polyfit(ds, ln, 1)
                spc = float(np.mean([p[2] for p in pts]))
                gamma = -slope / spc if spc else float("nan")
                fits.append({"cut": lbl, "score": key, "n_pts": len(pts),
                             "intercept_chi0": float(np.exp(c0)),
                             "slope_per_cycle": float(-slope),
                             "severed_per_cycle": spc,
                             "gamma_per_severed_cz": float(gamma)})
                print(f"fit {lbl}/{key}: chi ~ {np.exp(c0):.2f} * "
                      f"exp(-{-slope:.3f} d)  -> gamma = {gamma:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"region": region, "records": records, "fits": fits}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
