"""Patch partitions for patched XEB and the edge filter of the pseudo-patched mirror.

Patched circuits
----------------
The qubits are split into ``K`` connected, size-balanced patches and every CZ gate
crossing a patch boundary is removed, so the circuit factorises into ``K`` blocks that
can each be simulated exactly.  :func:`make_patches` searches for partitions with few
boundary couplers by randomised region growing followed by boundary local search; the
experiment uses the five best distinct partitions for ``K = 3`` and ``K = 4``
(``seed=2025``, ``balance_tol=1``, ``top_n=5``).

Pseudo-patched mirror
---------------------
A mirror of the *full* circuit would contain more CZ gates per cycle than the patched
circuits it is compared with.  :func:`make_pseudo_patch_edge_filter` removes, in every
cycle, the boundary couplers of one of the ``K = 3`` partitions and advances to the next
partition each cycle.  The number of removed gates per four-cycle sweep then matches a
3-patch circuit, while every coupler is used again under another partition, so the
mirror circuit stays connected.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Edge = Tuple[int, int]
PatchQubits = Dict[str, List[int]]


def _build_adjacency(num_qubits: int, edges: Iterable[Edge]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_qubits)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def count_cuts(assignment: Sequence[int], edges: Iterable[Edge]) -> int:
    """Number of edges whose end points lie in different patches."""
    return sum(1 for a, b in edges if assignment[a] != assignment[b])


def _is_connected(assignment: Sequence[int], patch: int, adj: List[List[int]]) -> bool:
    members = [q for q in range(len(assignment)) if assignment[q] == patch]
    if len(members) <= 1:
        return True
    seen = {members[0]}
    stack = [members[0]]
    while stack:
        q = stack.pop()
        for nb in adj[q]:
            if nb not in seen and assignment[nb] == patch:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(members)


def _canonical(assignment: Sequence[int]) -> tuple:
    """Relabel patch ids by first appearance so relabelled duplicates coincide."""
    mapping: Dict[int, int] = {}
    out = []
    for a in assignment:
        mapping.setdefault(a, len(mapping))
        out.append(mapping[a])
    return tuple(out)


def optimize_partition(num_qubits: int, edges: Sequence[Edge], k: int, num_restarts: int = 100,
                       max_iters: int = 3000, seed: int = 42, balance_tol: int = 2,
                       top_n: Optional[int] = None):
    """Search for ``k``-way partitions with few cut edges.

    Every restart grows ``k`` regions breadth-first from random seed qubits and then
    moves single boundary qubits between patches while the cut size decreases, keeping
    each patch connected and the patch sizes within the balance window.

    Parameters
    ----------
    num_qubits, edges
        The coupler graph in logical indices.
    k : int
        Number of patches.
    num_restarts, max_iters : int
        Search effort.
    seed : int
        Seed of the search; the partitions of the experiment use 2025.
    balance_tol : int
        Patch sizes lie in ``[n // k - balance_tol // 2, n // k + balance_tol // 2 (+1)]``.
    top_n : int, optional
        If given, return the ``top_n`` distinct partitions with the fewest cuts.

    Returns
    -------
    ``(assignment, metrics)`` for the best partition, or a list of such tuples sorted
    by cut size when ``top_n`` is given.  ``assignment`` maps qubit -> patch id.
    """
    rng = random.Random(seed)
    adj = _build_adjacency(num_qubits, edges)
    target = num_qubits // k
    extra = num_qubits % k
    min_size = max(1, target - balance_tol // 2)
    max_size = target + balance_tol // 2 + (1 if extra > 0 else 0)

    candidates = {}

    for _restart in range(num_restarts):
        seeds = rng.sample(range(num_qubits), k)
        asgn = [-1] * num_qubits
        sizes = [0] * k
        queues = [deque([s]) for s in seeds]
        for i, s in enumerate(seeds):
            asgn[s] = i
            sizes[i] = 1

        rounds_stuck = 0
        while any(a == -1 for a in asgn):
            progress = False
            for pid in range(k):
                if not queues[pid] or sizes[pid] >= max_size:
                    continue
                found = False
                for _ in range(len(queues[pid])):
                    q = queues[pid].popleft()
                    queues[pid].append(q)
                    for nb in adj[q]:
                        if asgn[nb] == -1:
                            asgn[nb] = pid
                            sizes[pid] += 1
                            queues[pid].append(nb)
                            found = True
                            progress = True
                            break
                    if found:
                        break
            if not progress:
                rounds_stuck += 1
                if rounds_stuck > 5:
                    for q in range(num_qubits):
                        if asgn[q] == -1:
                            for nb in adj[q]:
                                if asgn[nb] != -1 and sizes[asgn[nb]] < max_size + 2:
                                    asgn[q] = asgn[nb]
                                    sizes[asgn[nb]] += 1
                                    break
                            if asgn[q] == -1:
                                smallest = min(range(k), key=lambda p: sizes[p])
                                asgn[q] = smallest
                                sizes[smallest] += 1
                    break

        best_cuts = count_cuts(asgn, edges)
        for _ in range(max_iters):
            boundary = [q for q in range(num_qubits) if any(asgn[nb] != asgn[q] for nb in adj[q])]
            improved = False
            rng.shuffle(boundary)
            for q in boundary:
                old = asgn[q]
                if sizes[old] <= min_size:
                    continue
                for new in set(asgn[nb] for nb in adj[q]) - {old}:
                    if sizes[new] >= max_size:
                        continue
                    asgn[q] = new
                    sizes[old] -= 1
                    sizes[new] += 1
                    new_cuts = count_cuts(asgn, edges)
                    if new_cuts < best_cuts and _is_connected(asgn, old, adj):
                        best_cuts = new_cuts
                        improved = True
                        break
                    asgn[q] = old
                    sizes[old] += 1
                    sizes[new] -= 1
                if improved:
                    break
            if not improved:
                break

        sizes_list = [sum(1 for v in asgn if v == p) for p in range(k)]
        balanced = all(min_size <= s <= max_size for s in sizes_list)
        if balanced and all(_is_connected(asgn, p, adj) for p in range(k)):
            key = _canonical(asgn)
            if key not in candidates or best_cuts < candidates[key][0]:
                metrics = {"cuts": best_cuts, "sizes": sizes_list,
                           "edges_per_qubit": round((len(edges) - best_cuts) / num_qubits, 3)}
                candidates[key] = (best_cuts, dict(enumerate(asgn)), metrics)

    ranked = sorted(candidates.values(), key=lambda x: x[0])
    results = [(asgn, metrics) for _, asgn, metrics in ranked]
    if top_n is None:
        return results[0] if results else (None, None)
    return results[:top_n]


def _patch_data(assignment: Dict[int, int], edges: Sequence[Edge]) -> Tuple[Set[Edge], PatchQubits]:
    patch_qubits: PatchQubits = {}
    for q, pid in assignment.items():
        patch_qubits.setdefault(str(pid), []).append(q)
    for name in patch_qubits:
        patch_qubits[name].sort()
    boundary = {(a, b) for (a, b) in edges if assignment[a] != assignment[b]}
    return boundary, patch_qubits


def make_patches(num_qubits: int, good_edges: Iterable[Edge], num_patches: int, seed: int = 0,
                 balance_tol: int = 2, top_n: Optional[int] = None):
    """Patch partitions of the coupler graph.

    Returns ``(boundary_edges, patch_qubits)`` for the best partition, or a list of up
    to ``top_n`` such tuples sorted by the number of boundary edges.  ``boundary_edges``
    is the set of couplers whose CZ gates are removed and ``patch_qubits`` maps a patch
    name to its sorted logical qubits.
    """
    edges = sorted({tuple(sorted(e)) for e in good_edges})
    result = optimize_partition(num_qubits, edges, num_patches, num_restarts=100, max_iters=3000,
                                balance_tol=balance_tol, seed=seed or 42, top_n=top_n)
    if top_n is None:
        assignment, _ = result
        return _patch_data(assignment, edges)
    return [_patch_data(assignment, edges) for assignment, _ in result]


def make_pseudo_patch_edge_filter(boundary_sets: Sequence[Iterable[Edge]], rotate_every: int = 1):
    """Edge filter of the pseudo-patched mirror circuits.

    Parameters
    ----------
    boundary_sets : sequence of edge sets
        Boundary edges of the patch partitions to rotate through (the five ``K = 3``
        partitions in the experiment).
    rotate_every : int
        Number of cycles after which the active partition advances (1 in the experiment).

    Returns
    -------
    callable
        ``edge_filter(cycle, colour, edges)`` for :class:`rcs.circuits.RCSConfig`; it
        drops the boundary edges of the partition that is active in ``cycle``.
    """
    if not boundary_sets:
        raise ValueError("boundary_sets is empty")
    if rotate_every < 1:
        raise ValueError("rotate_every must be >= 1")
    sets = [{tuple(sorted(e)) for e in b} for b in boundary_sets]

    def edge_filter(cycle: int, colour: str, edges: List[Edge]) -> List[Edge]:
        """Drop the boundary couplers of the partition that is active in ``cycle``."""
        cut = sets[(cycle // rotate_every) % len(sets)]
        return [e for e in edges if tuple(sorted(e)) not in cut]

    return edge_filter
