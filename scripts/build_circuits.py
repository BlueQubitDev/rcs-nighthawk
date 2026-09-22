"""Regenerate every circuit of the experiment from ``data/layout.json``.

Writes, as QPY and OpenQASM 3,

    data/circuits/mirror/d{depth}_instance{i}         14 depths x 3 instances
    data/circuits/patched/K{K}/d{depth}/partition{j}_instance{i}
                                                      K = 3, 4; 6 depths; 5 partitions x 3 instances
    data/circuits/full/d{depth}_logical               unpatched circuit, depths 4 ... 40
    data/circuits/full/d36_executed                   the sampled circuit compiled to the device

together with ``data/circuits/manifest.json`` (gate counts and SHA-256 of every QPY file).
The circuits are deterministic functions of the layout and the base seed, so the files in
the repository can be reproduced bit for bit; ``--verify`` compares freshly built circuits
with the files on disk instead of writing.

    python scripts/build_circuits.py            # write
    python scripts/build_circuits.py --verify   # compare with the repository files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import qasm3, qpy  # noqa: E402

from rcs.circuits import (RCSConfig, build_mirror_circuit, build_rcs_circuit, compile_native_offline,  # noqa: E402
                          gate_signature, instance_seed)
from rcs.layout import load_layout  # noqa: E402
from rcs.patching import make_pseudo_patch_edge_filter  # noqa: E402

FULL_DEPTHS = list(range(4, 41, 4))


def all_circuits(layout):
    """Yield ``(relative_path_without_suffix, kind, circuit)`` for every circuit."""
    n, m, s, seed = layout.num_qubits, layout.matchings, layout.schedule, layout.base_seed
    mirror_filter = make_pseudo_patch_edge_filter([b for b, _ in layout.partitions[3]], rotate_every=1)
    for d in layout.depths["mirror"]:
        for i in range(layout.instances):
            cfg = RCSConfig(n, d // 2, m, s, edge_filter=mirror_filter)
            yield f"mirror/d{d:02d}_instance{i}", "mirror", build_mirror_circuit(cfg, instance_seed(seed, i, "mirror"), i)
    for k in (3, 4):
        for d in layout.depths["patched"]:
            for j, (boundary, _) in enumerate(layout.partitions[k]):
                for i in range(layout.instances):
                    cfg = RCSConfig(n, d, m, s, removed_edges=boundary)
                    yield (f"patched/K{k}/d{d}/partition{j}_instance{i}", "patched",
                           build_rcs_circuit(cfg, instance_seed(seed, i, "patched"), i))
    for d in FULL_DEPTHS:
        qc = build_rcs_circuit(RCSConfig(n, d, m, s), instance_seed(seed, 0, "full"), 0)
        yield f"full/d{d:02d}_logical", "full", qc
        if d in layout.depths["full_sampled"]:
            yield f"full/d{d:02d}_executed", "full_executed", compile_native_offline(qc, layout.logical_to_physical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="compare with the files on disk instead of writing")
    args = parser.parse_args()

    layout = load_layout(ROOT / "data" / "layout.json")
    out = ROOT / "data" / "circuits"
    manifest, mismatches = [], 0
    for rel, kind, qc in all_circuits(layout):
        path = out / (rel + ".qpy")
        if args.verify:
            with open(path, "rb") as fh:
                stored = qpy.load(fh)[0]
            same = gate_signature(stored) == gate_signature(qc)
            mismatches += not same
            if not same:
                print("DIFFERENT:", rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            qpy.dump(qc, fh)
        path.with_suffix(".qasm").write_text(qasm3.dumps(qc))
        ops = qc.count_ops()
        manifest.append({"file": rel + ".qpy", "qasm3": rel + ".qasm", "kind": kind, "num_qubits": qc.num_qubits,
                         "cz": int(ops.get("cz", 0)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if args.verify:
        print("all circuits identical to the files on disk" if not mismatches else f"{mismatches} circuits differ")
        sys.exit(1 if mismatches else 0)
    (out / "manifest.json").write_text(json.dumps({"format": "QPY (qiskit 1.4, format 13) and OpenQASM 3",
                                                   "count": len(manifest), "circuits": manifest}, indent=1))
    print(f"wrote {len(manifest)} circuits to {out}")


if __name__ == "__main__":
    main()
