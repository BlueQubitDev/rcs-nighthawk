"""Qubit layout of the experiment and the four-colouring of the square lattice.

Index conventions
-----------------
* *physical* qubit ``p`` of the 120-qubit device sits at row ``p // 10``, column
  ``p % 10`` of the 12 x 10 lattice;
* the experiment uses the 8 x 8 subgrid with rows 1-8 and columns 2-9; three of its
  qubits were dropped by the calibration filters (physical 17, 55 and 62);
* *logical* qubits ``0 .. 60`` number the 61 retained qubits in row-major order and are
  the qubit indices of every circuit and of every measured bitstring;
* a measured bitstring is stored as the integer ``sum_q bit_q * 2**q`` over logical
  qubits ``q``.  This equals ``int(key, 2)`` for a Qiskit counts key, whose rightmost
  character is qubit 0.

Couplers are coloured ``A`` (horizontal, even column), ``B`` (horizontal, odd column),
``C`` (vertical, even row) and ``D`` (vertical, odd row) of the 8 x 8 grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Edge = Tuple[int, int]
Matchings = Dict[str, List[Edge]]


def rect_matchings(rows: int, cols: int) -> Matchings:
    """Four disjoint CZ matchings covering every edge of a ``rows x cols`` grid.

    Grid site ``(r, c)`` has index ``r * cols + c``.
    """
    idx = lambda r, c: r * cols + c
    h_even, h_odd, v_even, v_odd = [], [], [], []
    for r in range(rows):
        for c in range(cols - 1):
            (h_even if c % 2 == 0 else h_odd).append((idx(r, c), idx(r, c + 1)))
    for r in range(rows - 1):
        for c in range(cols):
            (v_even if r % 2 == 0 else v_odd).append((idx(r, c), idx(r + 1, c)))
    return {"A": h_even, "B": h_odd, "C": v_even, "D": v_odd}


def remove_qubits(matchings: Matchings, removed: Iterable[int]) -> Matchings:
    """Delete grid sites from the matchings and renumber the survivors ``0 .. n-1``.

    Edges touching a removed site are dropped; the remaining sites keep their
    row-major order.
    """
    removed_set = set(removed)
    if not removed_set:
        return {k: list(v) for k, v in matchings.items()}
    sites = {q for edges in matchings.values() for e in edges for q in e}
    new_index = {old: new for new, old in enumerate(sorted(sites - removed_set))}
    return {k: [(new_index[a], new_index[b]) for a, b in edges if a not in removed_set and b not in removed_set]
            for k, edges in matchings.items()}


def filter_matchings(matchings: Matchings, usable_edges: Iterable[Edge]) -> Matchings:
    """Keep only the couplers that passed the calibration filters."""
    usable: Set[Edge] = {tuple(sorted(e)) for e in usable_edges}
    return {k: [(a, b) for a, b in edges if tuple(sorted((a, b))) in usable] for k, edges in matchings.items()}


def physical_to_logical(logical_to_physical: Sequence[int]) -> Dict[int, int]:
    """Inverse of the logical -> physical qubit map."""
    return {p: l for l, p in enumerate(logical_to_physical)}


@dataclass
class Layout:
    """Everything that defines the circuits of the experiment.

    Attributes
    ----------
    num_qubits : int
        Number of logical qubits (61).
    logical_to_physical : list of int
        Physical qubit of each logical qubit.
    matchings : dict
        Coupler colours in logical indices (29/22/29/22 couplers).
    schedule : list of str
        Colour order, ``["A", "B", "C", "D"]``.
    base_seed : int
        Base seed of all random circuits (2025).
    partitions : dict
        ``partitions[K]`` is the list of the five ``(boundary_edges, patch_qubits)``
        partitions used for ``K``-patch XEB.
    mirror_input_strings : list of str
        The ten Hamming-weight-30 input strings of the mirror benchmark; character ``q``
        of a string is the prepared state of logical qubit ``q``.
    depths : dict
        Measured depths per circuit family.
    instances : int
        Random circuit instances per depth (and per partition).
    """

    num_qubits: int
    logical_to_physical: List[int]
    matchings: Matchings
    schedule: List[str]
    base_seed: int
    partitions: Dict[int, List[Tuple[Set[Edge], Dict[str, List[int]]]]]
    mirror_input_strings: List[str]
    depths: Dict[str, List[int]]
    instances: int

    @property
    def edges(self) -> List[Edge]:
        """All couplers, sorted."""
        return sorted(tuple(sorted(e)) for v in self.matchings.values() for e in v)


def load_layout(path: str | Path) -> Layout:
    """Read ``data/layout.json``."""
    d = json.loads(Path(path).read_text())
    partitions = {int(k): [({tuple(e) for e in p["boundary_edges"]}, {n: list(q) for n, q in p["patch_qubits"].items()})
                           for p in plist] for k, plist in d["partitions"].items()}
    return Layout(num_qubits=d["num_qubits"], logical_to_physical=d["logical_to_physical"],
                  matchings={k: [tuple(e) for e in v] for k, v in d["matchings"].items()}, schedule=d["schedule"],
                  base_seed=d["base_seed"], partitions=partitions, mirror_input_strings=d["mirror_input_strings"],
                  depths=d["depths"], instances=d["instances"])
