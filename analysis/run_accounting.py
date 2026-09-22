"""Step 4: shot and QPU-time accounting of the released data (Appendix D).

Counts the shots in ``data/counts`` and ``data/samples`` and reports the QPU execution
time from ``data/qpu_usage.json``.

    python analysis/run_accounting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcs.io import load_shots  # noqa: E402


def main() -> None:
    mirror = json.loads((ROOT / "data" / "counts" / "mirror_survival.json").read_text())["depths"]
    per_instance = {int(d): int(np.sum(v["shots"][0])) for d, v in mirror.items()}
    mirror_total = sum(int(np.sum(v["shots"])) for v in mirror.values())
    per_circuit, patched_total = {}, 0
    for path in sorted((ROOT / "data" / "counts").glob("patched_K*_d*.npz")):
        shots = load_shots(path)
        depth = int(path.stem.split("_d")[1])
        per_circuit[depth] = len(next(iter(shots.values())))
        patched_total += sum(len(a) for a in shots.values())
    samples = len(load_shots(ROOT / "data" / "samples" / "full_d36.npz")["shots"])
    usage = json.loads((ROOT / "data" / "qpu_usage.json").read_text())

    print(f"mirror  : {mirror_total:>10,} shots; per instance {min(per_instance.values()):,} (4 cycles) to {max(per_instance.values()):,} (40 cycles)")
    print(f"patched : {patched_total:>10,} shots; per circuit  {min(per_circuit.values()):,} (20 cycles) to {max(per_circuit.values()):,} (40 cycles)")
    print(f"samples : {samples:>10,} shots of the full 36-cycle circuit")
    print(f"total   : {mirror_total + patched_total + samples:>10,} shots")
    print(f"QPU time: {usage['total_seconds']} s = {usage['total_seconds'] / 60:.1f} min, of which the sampling run took {usage['seconds']['full_d36_sampling']} s")


if __name__ == "__main__":
    main()
