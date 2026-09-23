#!/usr/bin/env python3
"""Cost-vs-search-time plateau experiment.

For one fixed circuit, run cotengra at a geometric ladder of wall-time
budgets and record the resulting contraction cost. Saves a JSON checkpoint
after every row so the plot script can be run before the sweep finishes.

Usage
-----
    python run_plateau.py [--circuit PATH] [--out plateau_data.json]
                          [--K 1000] [--memory-presets ideal,gpu-128gb]
                          [--budgets 5,15,30,60,180,600,1800]
                          [--seeds 3]

The "best" cotengra optimiser is kahypar (+ optuna for hyperparameter
search) with ``minimize='combo'`` and explicit slicing when a memory model
has a cap. ``--seeds N`` repeats each budget N times with different RNG
seeds so we can plot best/median curves with some noise band.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cotengra as ctg
import quimb.tensor as qtn  # noqa: F401

from qiskit import qasm2

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling_cost as sc  # noqa: E402
from flops_estimator import qiskit_to_quimb  # noqa: E402


DEFAULT_CIRCUIT = (
    "/Users/vmac/Documents/work/EPFL+BQ/ongoing/summit-main/MegaScan/"
    "circuits/vincent_google/swept_circuit_N56_T3_S1_A10_biased_RZZ474_"
    "target=00010010110000000000101000000100010100011100100101110000.qasm"
)


def _load_circuit(path: Path):
    if path.suffix == ".qpy":
        import qiskit.qpy as qpy
        with open(path, "rb") as f:
            return qpy.load(f)[0]
    return qasm2.load(str(path), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def _build_best_optimizer(memory_model: sc.MemoryModel, budget_s: float,
                          seed: int) -> ctg.HyperOptimizer:
    """Best-effort cotengra optimiser for these dense 56-qubit circuits.

    Uses kahypar + optuna (if installed) + minimize='combo' for slicing
    memory regimes, and explicit slicing_opts when the memory model caps
    intermediate size.
    """
    methods = ["kahypar"] if sc._have_kahypar() else ["greedy"]
    minimize = "combo" if memory_model.max_log2_size is not None else "flops"
    kwargs: Dict[str, Any] = dict(
        methods=methods,
        minimize=minimize,
        max_time=budget_s,
        max_repeats=10_000,
        parallel=False,
        progbar=False,
    )
    # NOTE: cotengra's HyperOptimizer does not accept a `seed` kwarg directly;
    # the optuna backend's sampler doesn't seed from create_study() in optuna 4.x.
    # We keep `seed` in the loop only as a row identifier when --seeds > 1.
    # try optuna; cotengra picks it up automatically if installed
    if memory_model.max_log2_size is not None:
        target_size = 2 ** memory_model.max_log2_size
        kwargs["slicing_opts"] = {"target_size": target_size}
        kwargs["slicing_reconf_opts"] = {"target_size": target_size}
    return ctg.HyperOptimizer(**kwargs)


def _stats(tree) -> Dict[str, Any]:
    sliced = getattr(tree, "sliced_inds", {}) or {}
    nsl = 1
    for ix in sliced:
        nsl *= int(tree.size_dict[ix])
    return {
        "log10_flops": float(tree.contraction_cost(log=10)),
        "width": float(tree.contraction_width()),
        "log2_peak": math.log2(float(tree.peak_size())),
        "n_slices": int(nsl),
        "n_sliced_inds": int(len(sliced)),
    }


def _run_one(qc, mm: sc.MemoryModel, budget_s: float, seed: int,
              K: int) -> Dict[str, Any]:
    qcirc = qiskit_to_quimb(qc.remove_final_measurements(inplace=False))
    N = qcirc.N
    out: Dict[str, Any] = {
        "memory": mm.name, "budget_s": budget_s, "seed": seed,
        "K": K, "n_qubits": N,
    }

    # ---- Closed amplitude ----
    t0 = time.time()
    opt = _build_best_optimizer(mm, budget_s, seed)
    info = qcirc.amplitude_rehearse(b="1" * N, optimize=opt)
    tree_amp = info["tree"]
    wall_amp = time.time() - t0
    s_amp = _stats(tree_amp)
    out.update({f"amp_{k}": v for k, v in s_amp.items()})
    out["amp_wall_s"] = wall_amp

    # ---- Open psi (sparse batched, post-hoc K-cap) ----
    t0 = time.time()
    psi = qcirc.psi
    opt = _build_best_optimizer(mm, budget_s, seed)
    tree_psi = psi.contraction_tree(optimize=opt)
    capped = sc._google_sparse_cost(tree_psi, K=K,
                                     open_output_inds=set(tree_psi.output))
    wall_psi = time.time() - t0
    s_psi = _stats(tree_psi)
    out.update({f"psi_{k}": v for k, v in s_psi.items()})
    out["psi_wall_s"] = wall_psi
    out["batched_log10_flops"] = capped["log10_flops_batched"]
    out["amort_log10_flops"] = (
        capped["log10_flops_batched"] - math.log10(max(K, 1))
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", type=Path, default=Path(DEFAULT_CIRCUIT))
    p.add_argument("--out", type=Path, default=Path("plateau_data.json"))
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--memory-presets", default="ideal,gpu-128gb",
                   help="comma-separated preset names")
    p.add_argument("--budgets", default="5,15,30,60,180,600",
                   help="comma-separated wall-time budgets (s)")
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds per (mem, budget); >1 enables variance bands")
    args = p.parse_args()

    print(f"Loading {args.circuit}", flush=True)
    qc = _load_circuit(args.circuit)
    print(f"  n={qc.num_qubits}  gates={len(qc.data)}", flush=True)

    presets = [s.strip() for s in args.memory_presets.split(",") if s.strip()]
    memory_models = [sc.PRESET_MEMORY_MODELS[name] for name in presets]
    budgets = [float(s) for s in args.budgets.split(",")]

    rows: List[Dict[str, Any]] = []
    if args.out.exists():
        try:
            rows = json.loads(args.out.read_text())
            print(f"Resumed from {args.out}: {len(rows)} rows", flush=True)
        except Exception:
            rows = []

    done_keys = {(r["memory"], r["budget_s"], r["seed"]) for r in rows}

    for mm in memory_models:
        for budget in budgets:
            for seed in range(args.seeds):
                key = (mm.name, budget, seed)
                if key in done_keys:
                    print(f"[skip] {key}", flush=True)
                    continue
                print(f"\n[{mm.name} budget={budget}s seed={seed}] running...",
                      flush=True)
                t0 = time.time()
                try:
                    row = _run_one(qc, mm, budget, seed, args.K)
                except Exception as e:
                    print(f"  FAILED: {e.__class__.__name__}: {e}", flush=True)
                    row = {"memory": mm.name, "budget_s": budget, "seed": seed,
                           "error": f"{e.__class__.__name__}: {e}"}
                dt = time.time() - t0
                row["row_wall_s"] = dt
                rows.append(row)
                args.out.write_text(json.dumps(rows, indent=2, default=str))
                if "error" in row:
                    print(f"  -> FAILED in {dt:.1f}s", flush=True)
                else:
                    print(
                        f"  amp:  log10={row['amp_log10_flops']:.2f} "
                        f"w={row['amp_width']:.1f} slc=10^{math.log10(max(row['amp_n_slices'],1)):.1f} "
                        f"({row['amp_wall_s']:.1f}s)",
                        flush=True,
                    )
                    print(
                        f"  psi:  batched log10={row['batched_log10_flops']:.2f} "
                        f"amort log10={row['amort_log10_flops']:.2f} "
                        f"w={row['psi_width']:.1f} ({row['psi_wall_s']:.1f}s)",
                        flush=True,
                    )

    print(f"\nDone. {len(rows)} rows saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
