"""Load Google rcs_tnsa .graph files into cotengra contraction inputs.

Format (one line per edge):
    <bond_dim> <node_a> <node_b>

node_a, node_b are integer tensor IDs. Node -1 denotes an open boundary
(output qubit index) — produced when ``circuit_to_tn.py`` is run with
``--open_output``.

This loader converts a .graph file into the (inputs, output, size_dict) triple
that cotengra's HyperOptimizer takes. Edge indices are auto-named ``e<n>``.
Each non-(-1) edge becomes a shared index between two tensors; each (-1) edge
becomes an open output index (kept in the final tensor).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_rcs_tnsa_graph(path: str | Path):
    """Parse a .graph file and return (inputs, output, size_dict, n_nodes).

    Parameters
    ----------
    path : str | Path
        Path to the .graph file.

    Returns
    -------
    inputs : list of tuple of str
        ``inputs[i]`` = indices on tensor i, in the order edges first encounter
        it. For cotengra this is the "tensor i's leg names".
    output : tuple of str
        Indices kept after full contraction (i.e. those that touch node -1).
    size_dict : dict {str: int}
        Bond dim per index name (always 2 in this format, but kept general).
    n_nodes : int
        Number of non-boundary tensors.
    """
    edges_raw: List[Tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"bad line in {path}: {line!r}")
            bd, a, b = (int(x) for x in parts)
            edges_raw.append((bd, a, b))

    # Assign a unique index name per edge
    node_to_inds: Dict[int, List[str]] = defaultdict(list)
    size_dict: Dict[str, int] = {}
    output_inds: List[str] = []
    for k, (bd, a, b) in enumerate(edges_raw):
        ix = f"e{k}"
        size_dict[ix] = bd
        if a == -1 and b == -1:
            raise ValueError(f"both ends are -1 in edge {k}")
        if a == -1:
            node_to_inds[b].append(ix)
            output_inds.append(ix)
        elif b == -1:
            node_to_inds[a].append(ix)
            output_inds.append(ix)
        else:
            node_to_inds[a].append(ix)
            node_to_inds[b].append(ix)

    n_nodes = max(node_to_inds) + 1
    inputs = [tuple(node_to_inds.get(i, ())) for i in range(n_nodes)]
    return inputs, tuple(output_inds), size_dict, n_nodes


def summary(path: str | Path) -> Dict[str, Any]:
    """Return a one-shot summary of the graph (n_nodes, n_edges, n_open)."""
    inputs, output, size_dict, n_nodes = load_rcs_tnsa_graph(path)
    return {
        "path": str(path),
        "n_tensors": n_nodes,
        "n_edges": len(size_dict),
        "n_open_outputs": len(output),
        "min_legs": min((len(t) for t in inputs if t), default=0),
        "max_legs": max((len(t) for t in inputs if t), default=0),
    }


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        s = summary(path)
        print(s)
