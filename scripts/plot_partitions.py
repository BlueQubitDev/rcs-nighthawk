"""Draw the five patch partitions used for K = 3 and K = 4 (Fig. 1c).

Qubits of a patch share a colour and the boundary couplers removed by the partition are
marked.  Needs only ``data/layout.json``.

    python scripts/plot_partitions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcs.layout import load_layout  # noqa: E402
from rcs.viz import plot_patch_coupling_map  # noqa: E402


def main() -> None:
    layout = load_layout(ROOT / "data" / "layout.json")
    edges = {tuple(e) for e in layout.edges} | {tuple(reversed(e)) for e in layout.edges}
    rows = len(layout.partitions[3])
    fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
    for col, k in enumerate((3, 4)):
        for row, (boundary, patch_qubits) in enumerate(layout.partitions[k]):
            ax = axes[row][col]
            plot_patch_coupling_map(patch_qubits, boundary, edges, np.array(layout.logical_to_physical),
                                    title=f"{k} patches, partition {row} ({len(boundary)} cut couplers)", ax=ax, node_size=200)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    fig.tight_layout()
    out = ROOT / "figures" / "fig1c_patch_partitions"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=80)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
