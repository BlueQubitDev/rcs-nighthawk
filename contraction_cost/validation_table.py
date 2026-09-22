"""Print the validation table of the paper (Appendix E) from ``results/validation``.

The five tensor networks are the ones released with Google's contraction-order optimizer
``rcs_tnsa``; "published" is the single-amplitude cost of Morvan et al., Nature 634, 328
(2024), Table 1, and "ours" the best cost found by ``run_rcs_tnsa_graphs.py`` with a
5-hour, 128-core search per contraction mode.
"""

from __future__ import annotations

import json
from pathlib import Path

NAMES = {"google_n53_m20": "Sycamore-53", "ustc_n56_m20": "Zuchongzhi-56", "ustc_n60_m24": "Zuchongzhi-60",
         "google_n70_m24": "Sycamore-70", "google_n67_m32": "Sycamore-67"}


def main() -> None:
    folder = Path(__file__).resolve().parent / "results" / "validation"
    print(f"{'circuit':15s} {'n':>3s} {'published':>10s} {'ours':>7s} {'delta':>7s}   (log10 C_amp, complex operations)")
    for key, name in NAMES.items():
        r = json.loads((folder / f"rcs_tnsa_costs_{key}.json").read_text())[0]
        pub, ours = r["morvan_log10_C_amp0"], r["amp_log10_flops"]
        print(f"{name:15s} {r['n_qubits']:3d} {pub:10.2f} {ours:7.2f} {ours - pub:+7.2f}")


if __name__ == "__main__":
    main()
