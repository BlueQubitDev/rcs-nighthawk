"""Run the experiment on IBM Quantum hardware (requires an IBM Quantum account).

Pipeline, as used for the paper
-------------------------------
1. *Placement.*  Score every offset of an 8 x 8 grid on the 12 x 10 lattice with the
   current calibration data and keep the best one (:mod:`rcs.placement`); qubits with
   readout error above 6 % and couplers with CZ error above 2 % are dropped.
2. *Circuits.*  Four-colour the retained couplers, generate the five best patch
   partitions for K = 3 and K = 4, and build the mirror, patched and full circuits
   (:mod:`rcs.circuits`, :mod:`rcs.patching`).
3. *Execution.*  One Qiskit Runtime ``Batch``; depths in ascending order and, at each
   depth, one SamplerV2 job per circuit family submitted back to back (mirror, K = 3,
   K = 4), so that slow drift affects all curves alike.  Measurement twirling on every
   job; gate twirling with 64 randomizations on the mirror jobs only.  The full 36-cycle
   circuit is sampled 10^6 times as ten PUBs of 10^5 shots.
4. *Results.*  Mirror survival counts and measured bitstrings are written in the format
   of the ``data`` folder, ready for the scripts in ``analysis``.

Credentials are read from the environment and are never written to disk:

    export QISKIT_IBM_TOKEN=...          # API key
    export QISKIT_IBM_INSTANCE=...       # instance CRN (optional)
    export QISKIT_IBM_CHANNEL=ibm_quantum_platform   # optional

Usage
-----
    python scripts/run_experiment.py --offline            # build all job payloads from data/layout.json,
                                                          # no network access, nothing is submitted
    python scripts/run_experiment.py --reuse-layout       # connect, reuse data/layout.json, dry run
    python scripts/run_experiment.py --submit --out runs/my_run
                                                          # new placement, submit, wait, save results

``--submit`` consumes QPU time (about 11 minutes for the shot numbers below).  The shot
numbers are the totals per circuit of the released data set.  This script is a cleaned-up
version of the notebook that ran the experiment; only its ``--offline`` path has been
executed after the clean-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcs.circuits import (RCSConfig, build_mirror_circuit, build_rcs_circuit, compile_native_locked,  # noqa: E402
                          compile_native_offline, instance_seed)
from rcs.io import counts_to_shots, save_shots  # noqa: E402
from rcs.layout import Layout, filter_matchings, load_layout, rect_matchings, remove_qubits  # noqa: E402
from rcs.patching import make_patches, make_pseudo_patch_edge_filter  # noqa: E402

BACKEND_NAME = "ibm_phoenix"
LOGICAL_GRID, PHYSICAL_GRID = (8, 8), (12, 10)
THRESH_CZ, THRESH_READOUT, QUBIT_PREFERENCE_TOL = 0.02, 0.06, 0.5
BASE_SEED, INSTANCES, PARTITIONS_PER_K, MIRROR_INPUTS, GATE_TWIRL_RANDOMIZATIONS = 2025, 3, 5, 10, 64
MIRROR_DEPTHS = list(range(4, 20, 2)) + list(range(20, 41, 4))
PATCHED_DEPTHS = list(range(20, 41, 4))
SAMPLED_DEPTH, SAMPLE_PUBS, SAMPLE_SHOTS_PER_PUB = 36, 10, 100_000
#: shots per mirror instance (split evenly over the ten input strings) and per patched circuit
MIRROR_SHOTS = {4: 30_000, 6: 30_000, 8: 30_000, 10: 45_000, 12: 45_000, 14: 45_000, 16: 60_000, 18: 60_000,
                20: 72_000, 24: 96_000, 28: 120_000, 32: 120_000, 36: 240_000, 40: 360_000}
PATCHED_SHOTS = {20: 2_400, 24: 4_800, 28: 12_000, 32: 24_000, 36: 36_000, 40: 54_000}


def connect():
    """Open the Runtime service with credentials from the environment."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        raise SystemExit("set QISKIT_IBM_TOKEN (and optionally QISKIT_IBM_INSTANCE) in the environment")
    service = QiskitRuntimeService(channel=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
                                   token=token, instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    return service, service.backend(BACKEND_NAME, use_fractional_gates=False)


def layout_from_calibration(backend) -> Layout:
    """Steps 1 and 2a: placement, colouring, partitions and mirror input strings."""
    from rcs.placement import find_best_placement, summarise_placement

    placement = find_best_placement(backend=backend, props=backend.properties(),
                                    logical_rows=LOGICAL_GRID[0], logical_cols=LOGICAL_GRID[1],
                                    physical_rows=PHYSICAL_GRID[0], physical_cols=PHYSICAL_GRID[1],
                                    thresh_cz=THRESH_CZ, thresh_readout=THRESH_READOUT,
                                    qubit_preference_tol=QUBIT_PREFERENCE_TOL)
    print(summarise_placement(placement))
    logical_to_physical = [int(p) for p in placement.log_to_phys]
    n = len(logical_to_physical)
    matchings = filter_matchings(remove_qubits(rect_matchings(*LOGICAL_GRID), placement.dropped_logical_grid_indices),
                                 placement.phys_to_log_edges)
    partitions = {k: make_patches(n, placement.phys_to_log_edges, k, seed=BASE_SEED, balance_tol=1, top_n=PARTITIONS_PER_K)
                  for k in (3, 4)}
    rng = np.random.default_rng(seed=BASE_SEED)
    strings = []
    for _ in range(MIRROR_INPUTS):
        bits = np.array(["0"] * n)
        bits[rng.choice(n, size=n // 2, replace=False)] = "1"
        strings.append("".join(bits))
    return Layout(num_qubits=n, logical_to_physical=logical_to_physical, matchings=matchings, schedule=["A", "B", "C", "D"],
                  base_seed=BASE_SEED, partitions=partitions, mirror_input_strings=strings,
                  depths={"mirror": MIRROR_DEPTHS, "patched": PATCHED_DEPTHS, "full_sampled": [SAMPLED_DEPTH]},
                  instances=INSTANCES)


def build_jobs(layout: Layout, compile_fn):
    """Step 2b: the list of jobs ``(label, depth, pubs, shots, sampler_options)`` in submission order."""
    n, m, s, seed = layout.num_qubits, layout.matchings, layout.schedule, layout.base_seed
    mirror_filter = make_pseudo_patch_edge_filter([b for b, _ in layout.partitions[3]], rotate_every=1)
    angles = np.array([[np.pi if c == "1" else 0.0 for c in string] for string in layout.mirror_input_strings])
    measure_twirl = {"enable_measure": True}
    gate_twirl = {"enable_gates": True, "enable_measure": True, "num_randomizations": GATE_TWIRL_RANDOMIZATIONS}

    jobs = []
    for depth in sorted(set(MIRROR_DEPTHS) | set(PATCHED_DEPTHS)):
        if depth in MIRROR_DEPTHS:
            circuits = [build_mirror_circuit(RCSConfig(n, depth // 2, m, s, edge_filter=mirror_filter),
                                             instance_seed(seed, i, "mirror"), i) for i in range(layout.instances)]
            shots_per_input = MIRROR_SHOTS[depth] // MIRROR_INPUTS
            jobs.append(("mirror", depth, [(compile_fn(qc), angles, shots_per_input) for qc in circuits], None, gate_twirl))
        if depth in PATCHED_DEPTHS:
            for k in (3, 4):
                circuits = [build_rcs_circuit(RCSConfig(n, depth, m, s, removed_edges=boundary),
                                              instance_seed(seed, i, "patched"), i)
                            for boundary, _ in layout.partitions[k] for i in range(layout.instances)]
                jobs.append((f"patched_K{k}", depth, [compile_fn(qc) for qc in circuits], PATCHED_SHOTS[depth], measure_twirl))
    full = compile_fn(build_rcs_circuit(RCSConfig(n, SAMPLED_DEPTH, m, s), instance_seed(seed, 0, "full"), 0))
    jobs.append(("full", SAMPLED_DEPTH, [full] * SAMPLE_PUBS, SAMPLE_SHOTS_PER_PUB, measure_twirl))
    return jobs


def submit_and_collect(backend, jobs, layout: Layout, out: Path) -> None:
    """Steps 3 and 4: submit all jobs in one batch, wait, and save the results."""
    from qiskit_ibm_runtime import Batch, SamplerV2

    rep_delay = backend.default_rep_delay
    handles = []
    with Batch(backend=backend) as batch:
        for label, depth, pubs, shots, twirling in jobs:
            sampler = SamplerV2(mode=batch, options={"execution": {"rep_delay": rep_delay}, "twirling": twirling,
                                                     "environment": {"job_tags": [f"rcs {label}", f"d = {depth}"]}})
            handles.append((label, depth, sampler.run(pubs, shots=shots)))
            print(f"submitted {label} d={depth}: {handles[-1][2].job_id()}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps([{"label": l, "depth": d, "job_id": j.job_id()} for l, d, j in handles], indent=1))

    mirror, usage = {}, {}
    for label, depth, job in handles:
        result = job.result()
        usage[f"{label}_d{depth}"] = job.usage()
        if label == "mirror":
            hits = [[pub.data.meas.get_counts(s).get(string[::-1], 0) for s, string in enumerate(layout.mirror_input_strings)]
                    for pub in result]
            mirror[str(depth)] = {"hits": hits, "shots": [[MIRROR_SHOTS[depth] // MIRROR_INPUTS] * MIRROR_INPUTS] * len(hits)}
        elif label == "full":
            merged: dict = {}
            for pub in result:
                for key, c in pub.data.c.get_counts().items():
                    merged[key] = merged.get(key, 0) + c
            save_shots(out / "samples" / f"full_d{depth}.npz", {"shots": counts_to_shots(merged)})
        else:
            arrays = {f"partition{j}_instance{i}": counts_to_shots(result[j * layout.instances + i].data.c.get_counts())
                      for j in range(PARTITIONS_PER_K) for i in range(layout.instances)}
            save_shots(out / "counts" / f"{label}_d{depth}.npz", arrays)
    (out / "counts").mkdir(parents=True, exist_ok=True)
    (out / "counts" / "mirror_survival.json").write_text(json.dumps({"depths": mirror}, indent=1))
    (out / "qpu_usage.json").write_text(json.dumps(usage, indent=1))
    print("results written to", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="no network: stored layout, offline compilation, no submission")
    parser.add_argument("--reuse-layout", action="store_true", help="use data/layout.json instead of a new placement")
    parser.add_argument("--submit", action="store_true", help="actually submit the jobs (consumes QPU time)")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "new_run")
    args = parser.parse_args()
    if args.offline and args.submit:
        raise SystemExit("--offline and --submit exclude each other")

    if args.offline:
        backend = None
        layout = load_layout(ROOT / "data" / "layout.json")
        compile_fn = lambda qc: compile_native_offline(qc, layout.logical_to_physical)
    else:
        _, backend = connect()
        layout = load_layout(ROOT / "data" / "layout.json") if args.reuse_layout else layout_from_calibration(backend)
        compile_fn = lambda qc: compile_native_locked(qc, backend, layout.logical_to_physical)

    jobs = build_jobs(layout, compile_fn)
    total = 0
    for label, depth, pubs, shots, _ in jobs:
        n_shots = sum(p[2] * len(layout.mirror_input_strings) for p in pubs) if label == "mirror" else shots * len(pubs)
        total += n_shots
        first = pubs[0][0] if label == "mirror" else pubs[0]
        print(f"{label:11s} d={depth:2d}: {len(pubs):2d} PUBs, {n_shots:>9,} shots, {first.count_ops().get('cz', 0):4d} CZ in the first circuit")
    print(f"{len(jobs)} jobs, {total:,} shots in total")
    if args.submit:
        submit_and_collect(backend, jobs, layout, args.out)
    else:
        print("dry run: nothing was submitted (use --submit to run on hardware)")


if __name__ == "__main__":
    main()
