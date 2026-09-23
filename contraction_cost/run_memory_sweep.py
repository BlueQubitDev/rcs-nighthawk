#!/usr/bin/env python3
"""One-search-multi-slice memory-model sweep for a single circuit.

Runs cotengra ONCE per mode (closed amplitude, open |psi>), then slices the
resulting tree to each memory cap post-hoc. This gives the full memory-model
column from a single search — cheaper and more self-consistent than
re-optimising per memory model, and it avoids the walltime blow-up of running
N memory models x 2 modes sequentially.

Columns produced per memory model:
  * C_amp^0(mem)      = closed-amp tree sliced to the cap  (ROBUST: cotengra
                        handles n_slices internally via contraction_cost).
  * C_samp_A(mem)     = 8 * kappa * N_s * F * C_amp^0(mem)     [convention A:
                        independent amplitudes; clean, an upper bound].
  * C_sparse(mem)     = open-|psi> tree sliced + K-cap  [convention B; FLAGGED
                        approximate — the K-cap under output-index slicing is
                        the estimator's known soft spot].
  * C_samp_B(mem)     = 8 * kappa * N_s * F * C_sparse(mem)/K.
  * n_slices, log2_peak per model.

Usage:
  python run_memory_sweep.py --circuit data/q61_circuits/q61_d30.qpy --depth 30 \
      --K 1000 --samples 1000000 --f-xeb 0.002 --opt-time 18000 \
      --parallel 128 --out results/q61_d30_memsweep.json
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Any, Dict, List

import cotengra as ctg
import sampling_cost as sc
from flops_estimator import qiskit_to_quimb

F_EFF = 0.20 * 1.685e18
YR = 3.156e7

# memory caps: (name, log2 max intermediate elements). None = ideal/unlimited.
MEM_MODELS = [
    ("ideal",         None),
    ("secondary-1pb", math.log2(1e15 / 8)),   # ~46.8
    ("allram-32tb",   math.log2(32e12 / 8)),   # ~41.9
    ("node-4tb",      math.log2(4e12 / 8)),     # ~38.9
    ("gpu-128gb",     math.log2(128e9 / 8)),    # ~33.9
]


def _load_circuit(path: Path):
    if path.suffix == ".qpy":
        import qiskit.qpy as qpy
        with open(path, "rb") as f:
            return qpy.load(f)[0]
    from qiskit import qasm2
    return qasm2.load(str(path), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def _opt(opt_time, parallel):
    """Fresh cotengra HyperOptimizer at IDEAL memory (minimize flops)."""
    methods = ["kahypar"] if sc._have_kahypar() else ["greedy"]
    return ctg.HyperOptimizer(methods=methods, minimize="flops",
                              max_time=opt_time, max_repeats=1_000_000,
                              parallel=parallel, progbar=False)


def _n_slices(tree):
    n = 1
    for ix in (getattr(tree, "sliced_inds", {}) or {}):
        n *= int(tree.size_dict[ix])
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", required=True, type=Path)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--f-xeb", type=float, default=2e-3)
    p.add_argument("--opt-time", type=float, default=18000.0)
    p.add_argument("--parallel", default="128")
    p.add_argument("--out", type=Path, default=Path("mem_sweep.json"))
    args = p.parse_args()
    parallel = int(args.parallel) if args.parallel.isdigit() else (
        False if args.parallel.lower() in ("false", "none", "0") else args.parallel)

    qc = _load_circuit(args.circuit).remove_final_measurements(inplace=False)
    qcirc = qiskit_to_quimb(qc)
    N = qcirc.N
    print(f"circuit {args.circuit.name}: n={N} gates={len(qc.data)} depth={args.depth}", flush=True)
    SF, KAPPA, Ns, F = 8.0, 10.0, args.samples, args.f_xeb

    rows: Dict[str, Dict[str, Any]] = {}

    def _write():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([rows[n] for n, _ in MEM_MODELS if n in rows],
                                        indent=2, default=str))

    # ---- ONE search, closed amplitude -> convention A (ROBUST); write first ----
    print("[search] closed amplitude (ideal)...", flush=True)
    t0 = time.time()
    info = qcirc.amplitude_rehearse(b="1" * N, optimize=_opt(args.opt_time, parallel))
    tree_amp = info["tree"]
    print(f"  done {time.time()-t0:.0f}s  ideal width={tree_amp.contraction_width():.0f} "
          f"log10C={tree_amp.contraction_cost(log=10):.2f}", flush=True)
    for name, cap in MEM_MODELS:
        ta = tree_amp if cap is None else tree_amp.slice(target_size=2 ** cap, minimize="flops")
        C_amp0 = float(ta.contraction_cost())
        nsl_a = _n_slices(ta)
        C_samp_A = SF * KAPPA * Ns * F * C_amp0
        rows[name] = dict(
            memory=name, log2_cap=cap, depth=args.depth, n=N, K=args.K, f_xeb=F, n_samples=Ns,
            log10_C_amp0=math.log10(C_amp0) if C_amp0 > 0 else 0,
            width_amp=ta.contraction_width(), n_slices_amp=nsl_a,
            log10_C_samp_A=math.log10(C_samp_A) if C_samp_A > 0 else 0,
            runtime_A_yr=C_samp_A / F_EFF / YR,
        )
        _write()
        print(f"[A {name:>13}] C_amp0=10^{rows[name]['log10_C_amp0']:.2f} "
              f"slices=10^{math.log10(max(nsl_a,1)):.1f} -> samp_A=10^{rows[name]['log10_C_samp_A']:.2f} "
              f"({rows[name]['runtime_A_yr']:.2g}yr)", flush=True)

    # ---- ONE search, open psi -> convention B (FLAGGED approximate) ----
    print("[search] open |psi> (ideal)...", flush=True)
    t0 = time.time()
    psi = qcirc.psi
    tree_psi = psi.contraction_tree(optimize=_opt(args.opt_time, parallel))
    psi_out = set(tree_psi.output)
    print(f"  done {time.time()-t0:.0f}s  ideal width={tree_psi.contraction_width():.0f} "
          f"log10C={tree_psi.contraction_cost(log=10):.2f}", flush=True)
    for name, cap in MEM_MODELS:
        tp = tree_psi if cap is None else tree_psi.slice(target_size=2 ** cap, minimize="flops")
        nsl_p = _n_slices(tp)
        open_now = psi_out - set(getattr(tp, "sliced_inds", {}) or {})
        C_sparse = sc._google_sparse_cost(tp, K=args.K, open_output_inds=open_now)["flops_batched"] * nsl_p
        C_amort = C_sparse / max(args.K, 1)
        C_samp_B = SF * KAPPA * Ns * F * C_amort
        rows[name].update(
            log10_C_sparse=math.log10(C_sparse) if C_sparse > 0 else 0,
            width_psi=tp.contraction_width(), n_slices_psi=nsl_p,
            log10_C_samp_B=math.log10(C_samp_B) if C_samp_B > 0 else 0,
            runtime_B_yr=C_samp_B / F_EFF / YR,
        )
        _write()
        print(f"[B {name:>13}] samp_B=10^{rows[name]['log10_C_samp_B']:.2f} "
              f"({rows[name]['runtime_B_yr']:.2g}yr)  [approx]", flush=True)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
