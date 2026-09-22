#!/usr/bin/env python3
"""Run our cost-model directly on Google's own rcs_tnsa .graph files.

For each .graph in ``data/google_rcs_tnsa/``:
  - Mode 1 (closed amp): add 1-leg projection tensors on every open output
    so the contraction is to a scalar; run cotengra; C_amp^0 = total FLOPs.
  - Mode 3 (sparse-output K-batched): keep open outputs as ``output_inds``,
    run cotengra, walk the tree with the Morvan/rcs_tnsa K-cap formula.

Cost-model arithmetic (8x complex, 10^7 N_prob, F_XEB, Frontier@20%)
is identical to ``sampling_cost.sampling_cost``.

Usage
-----
    python run_rcs_tnsa_graphs.py \\
        --opt-time 600 --K 1000 \\
        --samples 1000000 \\
        --out rcs_tnsa_costs.json
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import cotengra as ctg

import sampling_cost as sc
from load_rcs_tnsa_graph import load_rcs_tnsa_graph

# Per-circuit F_XEB from Morvan Table 1
F_XEB = {
    "google_n53_m20": 2e-3,   # SYC-53
    "google_n67_m32": 1e-3,   # SYC-67 d=32, 10^13 yr
    "google_n70_m24": 2e-3,   # SYC-70 d=24, 50 yr
    "ustc_n56_m20":  6e-4,    # ZCZ-56, 20 min
    "ustc_n60_m24":  3e-4,    # ZCZ-60, 40 days
}
# Morvan Table 1 reference numbers.
# Tuple = (Mode-1 single-amp FLOPs, Mode-3 realistic-memory FLOPs, Frontier-yr).
# The Frontier-yr here is the REALISTIC / limited-memory ("standard") figure.
# Our MemoryModel.ideal() run corresponds instead to Morvan's *unlimited-memory*
# floor (distributed RAM/secondary storage, bandwidth ignored) — see
# MORVAN_UNLIMITED_YR below for the cell our ideal-memory runs should be
# compared against.
MORVAN = {
    "google_n53_m20": (6e17, 2e17,  6.0/(3600*24*365.25)),     # 6 seconds (realistic)
    "google_n67_m32": (2e23, 2e37,  1e13),                     # 10^13 yr (realistic)
    "google_n70_m24": (5e23, 6e25,  50.0),                     # 50 yr (realistic)
    "ustc_n56_m20":   (6e19, 6e19,  20*60.0/(3600*24*365.25)), # 20 min (realistic)
    "ustc_n60_m24":   (1e21, 1e23,  40.0/365.25),              # 40 days (realistic)
}

# Morvan Table 1 *unlimited-memory* sampling floor (Frontier-yr), bandwidth
# ignored — the regime our no-cap MemoryModel.ideal() runs actually match.
#   SYC-67 m=32: realistic 10^13 yr -> all-RAM 6e4 yr -> all-secondary ~12 yr.
#   The 12 yr "all secondary storage, bandwidth ignored" value is the floor.
# For the shallower circuits the realistic figure already ~ the floor (no
# separate fantasy column in the paper); we reuse the realistic number.
MORVAN_UNLIMITED_YR = {
    "google_n53_m20": 6.0/(3600*24*365.25),     # ~ realistic (easy circuit)
    "google_n67_m32": 12.0,                      # all-secondary-storage floor
    "google_n70_m24": 50.0,                      # ~ realistic (24 cyc, no fantasy col)
    "ustc_n56_m20":   20*60.0/(3600*24*365.25),  # ~ realistic
    "ustc_n60_m24":   40.0/365.25,               # ~ realistic
}


def _build_optimizer(opt_time_s: float, memory_log2: float | None,
                     parallel: Any = False) -> Any:
    """kahypar + combo + optional slicing.

    ``parallel`` is passed straight to cotengra's HyperOptimizer:
      * ``False``      -> single-threaded (laptop default).
      * ``"loky"``     -> use all detected cores (one node).
      * ``int``        -> that many worker processes.
    More workers => more contraction-order trials per wall-second => better
    paths, with diminishing returns past the path-quality ceiling.
    """
    methods = ["kahypar"] if sc._have_kahypar() else ["greedy"]
    minimize = "combo" if memory_log2 is not None else "flops"
    kwargs = dict(
        methods=methods,
        minimize=minimize,
        max_time=opt_time_s,
        max_repeats=1_000_000,   # effectively unbounded; max_time is the cap
        parallel=parallel,
        progbar=False,
    )
    if memory_log2 is not None:
        target = 2 ** memory_log2
        kwargs["slicing_opts"] = {"target_size": target}
        kwargs["slicing_reconf_opts"] = {"target_size": target}
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


def _close_outputs(inputs, output_inds, size_dict):
    """Add 1-leg projection tensors so the TN contracts to a scalar."""
    new_inputs = list(inputs)
    for ix in output_inds:
        new_inputs.append((ix,))
    return new_inputs, (), size_dict


def run_one(graph_path: Path, K: int, opt_time_s: float,
            memory_log2: float | None = None,
            parallel: Any = False) -> Dict[str, Any]:
    inputs, output, size_dict, n_nodes = load_rcs_tnsa_graph(graph_path)
    n_qubits = len(output)
    out: Dict[str, Any] = {
        "graph": graph_path.name,
        "n_tensors": n_nodes,
        "n_edges": len(size_dict),
        "n_qubits": n_qubits,
        "K": K,
        "opt_time_s": opt_time_s,
        "memory_log2_cap": memory_log2,
        "parallel": str(parallel),
    }

    # ---- Mode 1: closed amp -----------------------------------------
    inputs_closed, output_closed, sd_closed = _close_outputs(
        inputs, output, size_dict)
    t0 = time.time()
    opt = _build_optimizer(opt_time_s, memory_log2, parallel=parallel)
    tree_amp = ctg.array_contract_tree(
        inputs_closed, output_closed, sd_closed, optimize=opt)
    wall_amp = time.time() - t0
    s_amp = _stats(tree_amp)
    out.update({f"amp_{k}": v for k, v in s_amp.items()})
    out["amp_wall_s"] = wall_amp

    # ---- Mode 3: open psi + K-cap ----------------------------------
    t0 = time.time()
    opt = _build_optimizer(opt_time_s, memory_log2, parallel=parallel)
    tree_psi = ctg.array_contract_tree(
        inputs, output, size_dict, optimize=opt)
    wall_psi = time.time() - t0
    s_psi = _stats(tree_psi)
    out.update({f"psi_{k}": v for k, v in s_psi.items()})
    out["psi_wall_s"] = wall_psi
    # K-cap
    capped = sc._google_sparse_cost(tree_psi, K=K,
                                     open_output_inds=set(tree_psi.output))
    out["batched_log10_flops"] = float(capped["log10_flops_batched"])
    out["amort_log10_flops"] = (
        out["batched_log10_flops"] - math.log10(max(K, 1))
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "google_rcs_tnsa")
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--opt-time", type=float, default=600.0)
    p.add_argument("--memory-log2", type=float, default=None,
                   help="optional memory cap (log2 elements)")
    p.add_argument("--graphs", default="all",
                   help="comma-separated graph stems, or 'all'")
    p.add_argument("--parallel", default="False",
                   help="cotengra parallelism: 'False', 'loky' (all cores), "
                        "or an integer worker count")
    p.add_argument("--out", type=Path, default=Path("rcs_tnsa_costs.json"))
    args = p.parse_args()

    # parse --parallel into False / "loky" / int
    if args.parallel.lower() in ("false", "none", "0"):
        parallel: Any = False
    elif args.parallel.isdigit():
        parallel = int(args.parallel)
    else:
        parallel = args.parallel  # e.g. "loky"

    machine = sc.MachineModel()
    SF = 8.0
    KAPPA = 10.0

    if args.graphs == "all":
        stems = sorted(F_XEB.keys())
    else:
        stems = [s.strip() for s in args.graphs.split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    if args.out.exists():
        try:
            rows = json.loads(args.out.read_text())
            print(f"Resumed: {len(rows)} rows from {args.out}", flush=True)
        except Exception:
            rows = []
    done = {(r["graph"], r.get("memory_log2_cap")) for r in rows}

    for stem in stems:
        graph = args.data_dir / f"{stem}.graph"
        if not graph.exists():
            print(f"[skip] missing {graph}", flush=True)
            continue
        key = (graph.name, args.memory_log2)
        if key in done:
            print(f"[skip] already done {key}", flush=True)
            continue
        print(f"\n=== {stem} ===", flush=True)
        row = run_one(graph, K=args.K, opt_time_s=args.opt_time,
                       memory_log2=args.memory_log2, parallel=parallel)
        # finalise sampling cost
        f_xeb = F_XEB[stem]
        flops_amp0 = 10 ** row["amp_log10_flops"]
        flops_amort = 10 ** row["amort_log10_flops"]
        n_prob = KAPPA * args.samples
        row["f_xeb"] = f_xeb
        row["n_samples"] = args.samples
        row["log10_C_samp_baseline"] = math.log10(SF * n_prob * f_xeb * flops_amp0)
        row["log10_C_samp_batched"]  = math.log10(SF * n_prob * f_xeb * flops_amort)
        row["runtime_baseline_s"]    = (SF * n_prob * f_xeb * flops_amp0) / machine.effective_flops
        row["runtime_batched_s"]     = (SF * n_prob * f_xeb * flops_amort) / machine.effective_flops
        # Compare to Morvan
        morvan_amp, morvan_samp, morvan_yr = MORVAN[stem]
        row["morvan_log10_C_amp0"] = math.log10(morvan_amp)
        row["morvan_log10_C_samp"] = math.log10(morvan_samp)
        row["morvan_runtime_yr"]   = morvan_yr
        row["runtime_baseline_yr"] = row["runtime_baseline_s"] / (3600 * 24 * 365.25)
        row["runtime_batched_yr"]  = row["runtime_batched_s"] / (3600 * 24 * 365.25)
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2, default=str))

        # Print line
        print(
            f"  n={row['n_qubits']:>3}  tensors={row['n_tensors']:>4}  "
            f"edges={row['n_edges']:>4}",
            flush=True,
        )
        print(
            f"  Mode 1 closed amp:    log10 C={row['amp_log10_flops']:.2f}  "
            f"w={row['amp_width']:.1f}  wall={row['amp_wall_s']:.1f}s",
            flush=True,
        )
        print(
            f"  Mode 3 open psi K={args.K}: log10 C_full={row['psi_log10_flops']:.2f}  "
            f"log10 C_batched={row['batched_log10_flops']:.2f}  "
            f"w={row['psi_width']:.1f}  wall={row['psi_wall_s']:.1f}s",
            flush=True,
        )
        print(
            f"  vs Morvan: Mode 1 {row['amp_log10_flops']:.2f} vs {math.log10(morvan_amp):.2f}  "
            f"(× {10**(row['amp_log10_flops']-math.log10(morvan_amp)):.2g})\n"
            f"             runtime B {row['runtime_batched_yr']:.2g} yr vs {morvan_yr:.2g} yr  "
            f"(× {row['runtime_batched_yr']/morvan_yr:.2g})",
            flush=True,
        )

    # Final table
    print("\n\n" + "=" * 110, flush=True)
    print(f"{'circuit':>18} {'n':>4} {'log10 C_amp':>11} {'log10 C_samp':>13} "
          f"{'runtime_yr':>14} | {'Morvan amp':>11} {'Morvan samp':>13} "
          f"{'Morvan_yr':>12}", flush=True)
    print("-" * 110, flush=True)
    for r in rows:
        print(f"{r['graph']:>18} {r['n_qubits']:>4} "
              f"{r['amp_log10_flops']:>11.2f} {r['log10_C_samp_batched']:>13.2f} "
              f"{r['runtime_batched_yr']:>14.2e} | "
              f"{r['morvan_log10_C_amp0']:>11.2f} {r['morvan_log10_C_samp']:>13.2f} "
              f"{r['morvan_runtime_yr']:>12.2e}", flush=True)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
