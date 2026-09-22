"""Single-amplitude contraction cost of the experimental circuits versus depth (Table I).

For every depth the full 61-qubit logical circuit ``data/circuits/full/dNN_logical.qpy``
is converted to a tensor network and the contraction order of the amplitude
``<1...1|U|0...0>`` is searched with :func:`sampling_cost.single_amplitude_cost`
(cotengra ``HyperOptimizer``, methods kahypar + greedy, objective = FLOPs, unlimited
memory) under a depth-scaled wall-clock budget.  ``C_amp`` is the contraction cost of
the best tree found and ``width`` is log2 of its largest intermediate tensor.

The search is stochastic: repeated runs scatter by about 0.1-0.2 decades at the largest
depths, and every value is an upper bound on the optimal cost.  The values quoted in the
paper are stored in ``results/depth_sweep.json``; this script appends to a separate file.

    python depth_sweep.py                          # all depths, paper budgets (about 6.5 h)
    python depth_sweep.py --depths 8 12 --budget 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qiskit import qpy  # noqa: E402

from sampling_cost import MemoryModel, single_amplitude_cost  # noqa: E402

#: Search budget in seconds used for Table I.
PAPER_BUDGET_S = {8: 120, 12: 300, 16: 600, 20: 900, 24: 1800, 28: 3600, 32: 5400, 36: 5400, 40: 5400}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depths", type=int, nargs="*", default=sorted(PAPER_BUDGET_S))
    parser.add_argument("--budget", type=float, default=None, help="seconds per depth (default: budgets of the paper)")
    parser.add_argument("--workers", type=int, default=20, help="parallel search workers (20 in the paper)")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "depth_sweep_rerun.json")
    args = parser.parse_args()

    circuits = HERE.parent / "data" / "circuits" / "full"
    results = json.loads(args.out.read_text()) if args.out.exists() else []
    for depth in args.depths:
        with open(circuits / f"d{depth:02d}_logical.qpy", "rb") as fh:
            qc = qpy.load(fh)[0]
        budget = args.budget if args.budget is not None else PAPER_BUDGET_S[depth]
        t0 = time.time()
        res = single_amplitude_cost(qc, MemoryModel.ideal(), opt_time_s=budget, max_repeats=10_000_000,
                                    parallel=args.workers)
        entry = {"depth": depth, "cz": int(qc.count_ops().get("cz", 0)), "width": res["width"],
                 "log10_camp": res["log10_flops"], "budget_s": budget, "wall_s": time.time() - t0}
        results.append(entry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"depth {depth:2d}: log10 C_amp = {entry['log10_camp']:.2f}, width = {entry['width']:.0f} "
              f"({entry['wall_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
