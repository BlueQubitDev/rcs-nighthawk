#!/usr/bin/env python3
"""CLI driver for the Morvan-style sampling-cost estimator.

Usage
-----
    python estimate_sampling_cost.py \\
        --circuit qasms/q=60,d=20.qpy \\
        --samples 1000000 \\
        --f-xeb 0.002 \\
        --batch-size 1000 \\
        --memory-gb 128 \\
        --opt-time 60 \\
        --depth 20 \\
        --out cost_report.json

Accepts both ``.qasm`` (OpenQASM 3) and ``.qpy`` circuits.  Multiple
``--memory-gb`` values may be passed (or ``--memory-preset ideal,gpu-128gb,
node-4tb``) to sweep memory regimes in a single run; one row of the CSV/JSON
is emitted per (circuit, memory model).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sampling_cost as sc


def _load_circuit(path: Path):
    """Load a qiskit QuantumCircuit from .qasm or .qpy.

    For .qasm we first try OpenQASM-2 with ``LEGACY_CUSTOM_INSTRUCTIONS``
    (so ``u``, ``u3``, ``rzz``, etc. resolve correctly even when the file
    declares ``OPENQASM 2.0``) and fall back to OpenQASM-3 otherwise.
    """
    if path.suffix == ".qpy":
        import qiskit.qpy as qpy
        with open(path, "rb") as f:
            return qpy.load(f)[0]
    if path.suffix in (".qasm", ".qasm3", ".qsm"):
        from qiskit import qasm2
        try:
            return qasm2.load(
                str(path),
                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            )
        except qasm2.QASM2ParseError:
            from qiskit import qasm3
            return qasm3.load(str(path))
    raise ValueError(
        f"Unrecognized circuit file extension: {path.suffix} (expect .qasm or .qpy)"
    )


def _resolve_memory_models(
    presets: Sequence[str],
    memory_gb_list: Sequence[float],
    bytes_per_elem: float,
) -> List[sc.MemoryModel]:
    out: List[sc.MemoryModel] = []
    seen = set()
    for name in presets:
        name = name.strip()
        if not name:
            continue
        if name not in sc.PRESET_MEMORY_MODELS:
            raise ValueError(
                f"Unknown memory preset: {name!r}. Available: "
                f"{sorted(sc.PRESET_MEMORY_MODELS)}"
            )
        mm = sc.PRESET_MEMORY_MODELS[name]
        if mm.name not in seen:
            out.append(mm)
            seen.add(mm.name)
    for gb in memory_gb_list:
        mm = sc.MemoryModel.gpu_gb(float(gb), bytes_per_element=bytes_per_elem)
        if mm.name not in seen:
            out.append(mm)
            seen.add(mm.name)
    if not out:
        out.append(sc.MemoryModel.ideal())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Morvan/rcs_tnsa-style sampling-cost estimator (quimb backend)"
    )
    p.add_argument("--circuit", required=True, type=Path,
                   help="Path to .qasm or .qpy circuit file")
    p.add_argument("--samples", "-N", type=int, default=1_000_000,
                   help="Number of noisy samples to draw (N_s)")
    p.add_argument("--f-xeb", type=float, required=True,
                   help="Target/measured XEB fidelity, e.g. 0.002")
    p.add_argument("--a-fidelity", type=float, default=None,
                   help=("Contraction-level fidelity discount A_fidelity. "
                         "Default: copy --f-xeb (Morvan's frugal-contraction "
                         "truncation is not implemented; this is a baseline)."))
    p.add_argument("--kappa-rej", type=float, default=sc.KAPPA_REJ_DEFAULT,
                   help="Rejection-sampling oversampling factor (default 10)")
    p.add_argument("--batch-size", "-K", type=int, default=1000,
                   help="K, number of sparse output configs per batch")
    p.add_argument("--memory-gb", type=float, nargs="*", default=[],
                   help=("Per-GPU memory cap(s) in GB. Multiple values sweep. "
                         "Combine freely with --memory-preset."))
    p.add_argument("--memory-preset", default="ideal,gpu-128gb",
                   help=("Comma-separated preset names from "
                         + ",".join(sorted(sc.PRESET_MEMORY_MODELS)) + "."))
    p.add_argument("--bytes-per-elem", type=float, default=8.0,
                   help="Bytes/element: 8 for complex64, 16 for complex128")
    p.add_argument("--opt-time", type=float, default=60.0,
                   help="cotengra search time per (circuit, mode), seconds")
    p.add_argument("--max-repeats", type=int, default=256,
                   help="cotengra HyperOptimizer max_repeats")
    p.add_argument("--depth", type=int, default=None,
                   help="Circuit depth/cycle count for the diagnostic table")
    p.add_argument("--peak-flops", type=float, default=1.685e18,
                   help="Machine peak FLOP/s (default Frontier FP32)")
    p.add_argument("--efficiency", type=float, default=0.20,
                   help="Sustained / peak FLOP efficiency")
    p.add_argument("--machine-name", default="Frontier",
                   help="Display name for the machine row")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. .json -> JSON; .csv -> CSV; else both")
    p.add_argument("--parallel", default="False",
                   help="cotengra parallelism: 'False', 'loky' (all cores), "
                        "or an integer worker count")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-row console table")
    args = p.parse_args(argv)

    # parse --parallel into False / "loky" / int
    if args.parallel.lower() in ("false", "none", "0"):
        parallel: Any = False
    elif args.parallel.isdigit():
        parallel = int(args.parallel)
    else:
        parallel = args.parallel  # e.g. "loky"

    presets = [s for s in args.memory_preset.split(",") if s]
    memory_models = _resolve_memory_models(presets, args.memory_gb,
                                            args.bytes_per_elem)
    fid = sc.FidelityModel(
        f_xeb=args.f_xeb,
        n_samples=args.samples,
        kappa_rej=args.kappa_rej,
        a_fidelity=args.a_fidelity,
    )
    machine = sc.MachineModel(
        name=args.machine_name,
        peak_flops=args.peak_flops,
        efficiency=args.efficiency,
    )

    print(f"Loading circuit from: {args.circuit}")
    qc = _load_circuit(args.circuit)
    label = args.circuit.stem
    print(f"  qubits = {qc.num_qubits}, gates = {len(qc.data)}, "
          f"depth = {args.depth}")
    print(f"Memory models: {[m.name for m in memory_models]}")
    print(f"K = {args.batch_size}, N_s = {args.samples}, F_XEB = {args.f_xeb}, "
          f"kappa_rej = {args.kappa_rej}, A_fidelity = {fid.a_eff}")
    print(f"Machine: {machine.name} @ {machine.peak_flops:.3g} FLOP/s × "
          f"{machine.efficiency:.0%} eff")
    print()

    rows: List[Dict[str, Any]] = []
    for mm in memory_models:
        print(f"[{mm.name}] running cotengra (budget {args.opt_time}s × 2)...")
        t0 = time.time()
        rep = sc.sampling_cost(
            qc,
            fidelity=fid,
            K=args.batch_size,
            memory_model=mm,
            machine=machine,
            opt_time_s=args.opt_time,
            max_repeats=args.max_repeats,
            depth=args.depth,
            notes=f"circuit={label}",
            parallel=parallel,
        )
        dt = time.time() - t0
        row = rep.as_dict()
        row["label"] = label
        row["circuit_path"] = str(args.circuit)
        row["wallclock_s"] = dt
        rows.append(row)
        if not args.quiet:
            print(sc.format_report(rep, header=True))
            print()

    if args.out is not None:
        out_path = args.out
        suffix = out_path.suffix.lower()
        if suffix == ".json":
            _write_json(out_path, rows)
        elif suffix == ".csv":
            _write_csv(out_path, rows)
        else:
            _write_json(out_path.with_suffix(".json"), rows)
            _write_csv(out_path.with_suffix(".csv"), rows)
        print(f"Wrote diagnostics: {out_path}")
    return 0


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


if __name__ == "__main__":
    sys.exit(main())
