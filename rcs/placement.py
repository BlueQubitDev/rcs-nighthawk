"""Calibration-aware placement of a rectangular logical grid on a rectangular device.

Given a backend's live calibration data, this module searches over every possible
offset placing a ``logical_rows x logical_cols`` grid inside a
``physical_rows x physical_cols`` device, scores each by a fidelity proxy, and
returns the offset that maximises expected fidelity.

The experiment places an 8 x 8 grid on the 12 x 10 lattice of ``ibm_phoenix`` with
``thresh_cz=0.02``, ``thresh_readout=0.06`` and ``qubit_preference_tol=0.1``; see
``scripts/run_experiment.py``.  The resulting layout is stored in ``data/layout.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

Edge = Tuple[int, int]


@dataclass
class PlacementResult:
    """Outcome of a placement search.

    Attributes
    ----------
    log_to_phys : np.ndarray
        Row-major logical -> physical qubit index map of length
        ``logical_rows * logical_cols``.
    phys_to_log_edges : Set[Edge]
        Symmetrised set of *logical* edge pairs ``(la, lb)`` for every good
        physical edge whose endpoints both fall inside the chosen qubit set.
        Both orderings ``(la, lb)`` and ``(lb, la)`` are present, as in a
        symmetrised CouplingMap.
    dr, dc : int
        Row and column offset of the chosen placement inside the physical grid.
    log_F : float
        Score = ``sum_q log(1 - e_readout) + sum_q log(1 - e_sx) +
        sum_{surviving target edges} log(1 - e_cz) - penalty * #missing``.
        Higher is better.
    missing_logical_edges : List[Edge]
        Target-topology edges that have no good physical realisation under
        this placement (i.e. their physical CZ error is above the threshold or
        the edge is absent from the device coupling map).
    total_edges : int
        Number of edges in the target ``logical_rows x logical_cols`` grid.
    candidates : List["PlacementCandidate"]
        Score breakdown for every enumerated placement, ordered by descending
        score. Useful for inspecting how close the runner-up was.
    """

    log_to_phys: np.ndarray
    phys_to_log_edges: Set[Edge]
    dr: int
    dc: int
    log_F: float
    missing_logical_edges: List[Edge]
    total_edges: int
    edge_errors: Dict[Edge, float] = field(default_factory=dict)
    candidates: List["PlacementCandidate"] = field(default_factory=list)
    # Original-grid logical positions (0..LR*LC-1, row-major) that were dropped
    # from this rectangle — either because their readout error exceeded
    # ``thresh_readout`` or because every CZ edge incident to them got pruned.
    # Pass to REMOVE_QUBITS(RECT_MATCHINGS(LR, LC), ...) to compact-renumber
    # the canonical 4-colouring to match the surviving qubits.
    dropped_logical_grid_indices: Set[int] = field(default_factory=set)


@dataclass
class PlacementCandidate:
    """Lightweight summary of a single candidate placement."""

    dr: int
    dc: int
    log_F: float
    num_missing_edges: int


def _rect_target_edges(rows: int, cols: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return the horizontal-then-vertical edge list of an ``rows x cols`` grid.

    Each element is ``((r1, c1), (r2, c2))`` in row-major orientation.
    """
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols - 1):
            edges.append(((r, c), (r, c + 1)))
    for r in range(rows - 1):
        for c in range(cols):
            edges.append(((r, c), (r + 1, c)))
    return edges


def _collect_cz_errors(backend, props, thresh_cz: float) -> Dict[Edge, float]:
    """Return ``{(min, max): error}`` for every device CZ edge below ``thresh_cz``."""
    cz_error: Dict[Edge, float] = {}
    for pair in backend.target["cz"]:
        a, b = int(pair[0]), int(pair[1])
        try:
            e = props.gate_error("cz", [a, b])
        except Exception:
            e = None
        if e is None:
            continue
        e_f = float(e)
        if e_f >= thresh_cz:
            continue
        key = (min(a, b), max(a, b))
        # If both directions of the edge appear separately, keep the smaller error.
        if key not in cz_error or e_f < cz_error[key]:
            cz_error[key] = e_f
    return cz_error


def _collect_qubit_errors(
    backend, props, include_sx_cost: bool
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return ``(readout_err, sx_err)`` dicts keyed by physical qubit index."""
    readout_err: Dict[int, float] = {}
    sx_err: Dict[int, float] = {}
    for q in range(backend.num_qubits):
        try:
            readout_err[q] = float(props.readout_error(q))
        except Exception:
            readout_err[q] = 0.0
        if include_sx_cost:
            try:
                e = props.gate_error("sx", q)
                sx_err[q] = float(e) if e is not None else 0.0
            except Exception:
                sx_err[q] = 0.0
    return readout_err, sx_err


def find_best_placement(
    backend,
    props,
    logical_rows: int,
    logical_cols: int,
    physical_rows: int,
    physical_cols: int,
    thresh_cz: float,
    thresh_readout: Optional[float] = None,
    missing_edge_penalty: Optional[float] = None,
    require_all_edges: bool = False,
    include_sx_cost: bool = True,
    phys_idx_fn: Optional[Callable[[int, int], int]] = None,
    qubit_preference_tol: float = 0.0,
) -> PlacementResult:
    """Search every offset placement of a logical grid and return the best.

    The score for a placement at offset ``(dr, dc)`` is::

        log_F = Σ_q log(1 - e_readout(q))
              + Σ_q log(1 - e_sx(q))           # if include_sx_cost
              + Σ_e log(1 - e_cz(e))           # over surviving target edges
              - penalty * (# missing target edges)

    A "missing" edge is a target-topology edge whose corresponding physical edge
    has CZ error above ``thresh_cz``, is reported as None, or is absent from the
    device's CZ coupling map.

    Parameters
    ----------
    backend : Backend
        Qiskit backend; only ``backend.target["cz"]`` and ``backend.num_qubits``
        are used.
    props : BackendProperties
        Live calibration data used for gate / readout errors.
    logical_rows, logical_cols : int
        Target grid dimensions.
    physical_rows, physical_cols : int
        Physical device dimensions. The default index map assumes
        ``phys_idx(r, c) = r * physical_cols + c``; override with
        ``phys_idx_fn`` for non-rectangular layouts.
    thresh_cz : float
        CZ-error cutoff; edges at or above this are considered missing.
    thresh_readout : Optional[float]
        Reject any placement that contains a qubit with readout error >= this.
        ``None`` disables the filter.
    missing_edge_penalty : Optional[float]
        Cost added to ``-log_F`` for each missing target edge. Defaults to
        ``-log(0.1) ≈ 0.693`` (equivalent to a 50%-error gate). Use a larger
        value to bias the search toward placements with full connectivity.
    require_all_edges : bool
        If True, reject any placement with at least one missing edge.
        Equivalent to ``missing_edge_penalty = +inf``.
    include_sx_cost : bool
        Include single-qubit (SX) gate error in the score. SX errors are
        usually uniform across the device so this rarely changes the ranking,
        but it makes the absolute score meaningful as a per-cycle fidelity
        proxy.
    phys_idx_fn : Optional[Callable[[int, int], int]]
        Override the default rectangular index map. Useful for heavy-hex or
        any device whose qubit numbering doesn't follow ``row*cols + col``.
    qubit_preference_tol : float
        Log-fidelity slack allowed when preferring placements with more
        retained qubits. The search keeps every candidate within
        ``qubit_preference_tol`` of the highest ``log_F`` and picks the one
        with the most kept qubits among them (ties broken by ``log_F``).
        The default 0.0 reproduces the pure max-``log_F`` policy.

    Returns
    -------
    PlacementResult

    Raises
    ------
    ValueError
        If the logical grid doesn't fit in the physical grid, or if no
        candidate placement is valid under the chosen filters.
    """
    if logical_rows > physical_rows or logical_cols > physical_cols:
        raise ValueError(
            f"Logical grid {logical_rows}x{logical_cols} does not fit in "
            f"physical grid {physical_rows}x{physical_cols}."
        )

    if missing_edge_penalty is None:
        missing_edge_penalty = -math.log(0.1)
    if require_all_edges:
        missing_edge_penalty = math.inf

    if phys_idx_fn is None:
        cols = physical_cols

        def phys_idx_fn(r: int, c: int) -> int:
            """Default physical index of device site (r, c): row * columns + column."""
            return r * cols + c

    def logical_idx(r: int, c: int) -> int:
        """Row-major index of logical grid site (r, c)."""
        return r * logical_cols + c

    cz_error = _collect_cz_errors(backend, props, thresh_cz)
    readout_err, sx_err = _collect_qubit_errors(backend, props, include_sx_cost)

    rel_edges = _rect_target_edges(logical_rows, logical_cols)
    total_edges = len(rel_edges)

    n_dr = physical_rows - logical_rows + 1
    n_dc = physical_cols - logical_cols + 1

    # Each snapshot is enough to rebuild a PlacementResult; we defer the
    # final selection so the qubit-count preference can apply.
    snapshots: List[Tuple[int, int, float, Set[int], Set[int],
                          Dict[int, int], List[Edge]]] = []

    for dr in range(n_dr):
        for dc in range(n_dc):
            # Map original-grid logical index -> physical qubit for this rectangle.
            log_to_phys_orig: Dict[int, int] = {
                logical_idx(r, c): phys_idx_fn(r + dr, c + dc)
                for r in range(logical_rows)
                for c in range(logical_cols)
            }

            # Step 1: drop qubits whose readout error fails the threshold.
            if thresh_readout is not None:
                dropped: Set[int] = {
                    li for li, pq in log_to_phys_orig.items()
                    if readout_err.get(pq, 1.0) >= thresh_readout
                }
            else:
                dropped = set()

            # Step 2: drop qubits with no surviving CZ edges (degree 0) in the
            # remaining subset. Single pass suffices — removing a degree-0
            # node can't change anyone else's degree.
            kept_logical = set(log_to_phys_orig) - dropped
            degree: Dict[int, int] = {li: 0 for li in kept_logical}
            for (r1, c1), (r2, c2) in rel_edges:
                la = logical_idx(r1, c1)
                lb = logical_idx(r2, c2)
                if la not in kept_logical or lb not in kept_logical:
                    continue
                pa = log_to_phys_orig[la]
                pb = log_to_phys_orig[lb]
                ekey = (min(pa, pb), max(pa, pb))
                if ekey in cz_error:
                    degree[la] += 1
                    degree[lb] += 1
            isolated = {li for li in kept_logical if degree[li] == 0}
            dropped |= isolated
            kept_logical -= isolated

            if not kept_logical:
                continue

            # Step 3: score the surviving subset. Edges with one endpoint
            # outside ``kept_logical`` are simply not part of the layout, so
            # they don't incur a missing-edge penalty.
            log_F = 0.0
            for li in kept_logical:
                pq = log_to_phys_orig[li]
                p = max(1.0 - readout_err.get(pq, 0.0), 1e-12)
                log_F += math.log(p)
                if include_sx_cost:
                    p = max(1.0 - sx_err.get(pq, 0.0), 1e-12)
                    log_F += math.log(p)

            missing_orig: List[Edge] = []
            invalid = False
            for (r1, c1), (r2, c2) in rel_edges:
                la = logical_idx(r1, c1)
                lb = logical_idx(r2, c2)
                if la not in kept_logical or lb not in kept_logical:
                    continue
                pa = log_to_phys_orig[la]
                pb = log_to_phys_orig[lb]
                ekey = (min(pa, pb), max(pa, pb))
                if ekey in cz_error:
                    p = max(1.0 - cz_error[ekey], 1e-12)
                    log_F += math.log(p)
                else:
                    if math.isinf(missing_edge_penalty):
                        invalid = True
                        break
                    log_F -= missing_edge_penalty
                    missing_orig.append((min(la, lb), max(la, lb)))

            if invalid:
                continue

            snapshots.append(
                (dr, dc, log_F, kept_logical, dropped, log_to_phys_orig, missing_orig)
            )

    if not snapshots:
        raise ValueError(
            f"No valid placement found for {logical_rows}x{logical_cols} grid on "
            f"{physical_rows}x{physical_cols} device "
            f"(thresh_cz={thresh_cz}, thresh_readout={thresh_readout}, "
            f"require_all_edges={require_all_edges})."
        )

    # Pick the snapshot: among candidates whose log_F is within
    # qubit_preference_tol of the maximum, prefer the one with the most kept
    # qubits, breaking ties by log_F.
    max_log_F = max(s[2] for s in snapshots)
    eligible = [s for s in snapshots if s[2] >= max_log_F - qubit_preference_tol]
    chosen = max(eligible, key=lambda s: (len(s[3]), s[2]))
    dr_c, dc_c, log_F_c, kept_logical_c, dropped_c, log_to_phys_orig_c, missing_orig_c = chosen

    # Compact renumbering: surviving original-grid indices get consecutive new
    # indices in [0, |kept|), preserving order.
    sorted_kept = sorted(kept_logical_c)
    old_to_new = {old: new for new, old in enumerate(sorted_kept)}
    new_log_to_phys = np.array([log_to_phys_orig_c[li] for li in sorted_kept])
    kept_phys = {log_to_phys_orig_c[li] for li in kept_logical_c}
    kept_phys_to_orig = {log_to_phys_orig_c[li]: li for li in kept_logical_c}

    ptl_edges: Set[Edge] = set()
    ptl_edge_errors: Dict[Edge, float] = {}
    for (a, b), e_ab in cz_error.items():
        if a in kept_phys and b in kept_phys:
            la = old_to_new[kept_phys_to_orig[a]]
            lb = old_to_new[kept_phys_to_orig[b]]
            ptl_edges.add((la, lb))
            ptl_edges.add((lb, la))
            ptl_edge_errors[(min(la, lb), max(la, lb))] = e_ab

    missing_compact = [
        (min(old_to_new[la], old_to_new[lb]),
         max(old_to_new[la], old_to_new[lb]))
        for (la, lb) in missing_orig_c
    ]

    candidates = [
        PlacementCandidate(dr=s[0], dc=s[1], log_F=s[2],
                           num_missing_edges=len(s[6]))
        for s in snapshots
    ]
    candidates.sort(key=lambda c: c.log_F, reverse=True)

    return PlacementResult(
        log_to_phys=new_log_to_phys,
        phys_to_log_edges=ptl_edges,
        dr=dr_c,
        dc=dc_c,
        log_F=log_F_c,
        missing_logical_edges=missing_compact,
        total_edges=total_edges,
        edge_errors=ptl_edge_errors,
        candidates=candidates,
        dropped_logical_grid_indices=dropped_c,
    )


def summarise_placement(result: PlacementResult) -> str:
    """Return a short human-readable description of a placement result."""
    n_retained = result.total_edges - len(result.missing_logical_edges)
    lines = [
        f"Best placement: dr={result.dr}, dc={result.dc}",
        f"  qubits   : {len(result.log_to_phys)}",
        f"  edges    : {n_retained}/{result.total_edges} retained "
        f"({len(result.missing_logical_edges)} missing)",
        f"  log F    : {result.log_F:.4f}  (expected fidelity proxy: "
        f"{math.exp(result.log_F):.3e})",
    ]
    if result.dropped_logical_grid_indices:
        lines.append(
            f"  dropped  : {len(result.dropped_logical_grid_indices)} qubit(s) "
            f"(bad readout or fully isolated after CZ pruning)"
        )
    if result.candidates and len(result.candidates) > 1:
        runner_up = result.candidates[1]
        lines.append(
            f"  runner-up: dr={runner_up.dr}, dc={runner_up.dc}, "
            f"log F = {runner_up.log_F:.4f} "
            f"(Δ = {result.log_F - runner_up.log_F:+.4f})"
        )
    return "\n".join(lines)
