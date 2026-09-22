"""Construction of the random circuits, their patched variants and the mirror circuits.

Circuit family
--------------
One *cycle* is a layer of independent Haar-random SU(2) gates on every qubit followed
by CZ gates on one of the four coupler colours ``A, B, C, D`` of the square lattice.
The colours are applied in the fixed repeating order given by ``schedule``; four
cycles therefore fire every coupler once.  A depth-``d`` circuit applies ``d`` cycles
and measures every qubit in the computational basis.

Single-qubit gates
------------------
Each gate is written as ``RZ(lambda) RX(theta) RZ(phi)`` (applied in the order
``rz(phi)``, ``rx(theta)``, ``rz(lambda)``) with ``phi, lambda ~ U[0, 2 pi)`` and
``cos(theta) ~ U[-1, 1]``, which is Haar measure on SU(2).  The three uniform numbers
come from a stateless SplitMix64 hash of ``(seed, instance, cycle, qubit, k)``, so any
circuit of the experiment can be regenerated exactly without replaying a random stream.

Seeding used in the experiment (base seed 2025)
-----------------------------------------------
* full and patched circuits: instance ``k`` uses ``seed = 2025 + k * 1000003`` and
  instance index ``k``;
* mirror circuits: instance ``k`` uses ``seed = 2025`` and instance index ``k``.

Both conventions are reproduced by :func:`instance_seed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import CouplingMap

Edge = Tuple[int, int]
Matchings = Dict[str, List[Edge]]
EdgeFilter = Callable[[int, str, List[Edge]], List[Edge]]

#: Seed offset between consecutive instances of full and patched circuits.
INSTANCE_SEED_STRIDE = 1000003

#: Native basis used for compilation on Nighthawk (RZ is virtual, RX is not used).
NATIVE_BASIS = ("rz", "sx", "x", "cz", "id", "reset")

_MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------- hashing
def _splitmix64(x: int) -> int:
    """One round of the stateless SplitMix64 mixer on a 64-bit integer."""
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def u01_from_key(*parts: int) -> float:
    """Deterministic uniform number in ``[0, 1)`` from a tuple of integers.

    The parts are folded one by one through SplitMix64 and the top 53 bits of the
    result are mapped to a double.
    """
    x = 0
    for p in parts:
        x = _splitmix64(x ^ (int(p) & _MASK64))
    return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def haar_angles(seed: int, instance: int, cycle: int, qubit: int) -> Tuple[float, float, float]:
    """Euler angles ``(phi, theta, lambda)`` of the Haar-random gate at one site.

    Returns the arguments of ``rz(phi)``, ``rx(theta)``, ``rz(lambda)`` in the order
    in which the three rotations are applied.
    """
    phi = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 1)
    z = 1.0 - 2.0 * u01_from_key(seed, instance, cycle, qubit, 2)
    theta = math.acos(min(1.0, max(-1.0, z)))
    lam = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 3)
    return phi, theta, lam


def instance_seed(base_seed: int, instance: int, kind: str) -> int:
    """Seed used for instance ``instance`` of a circuit family.

    Parameters
    ----------
    base_seed : int
        Base seed of the experiment (2025).
    instance : int
        Instance index, starting at 0.
    kind : {"full", "patched", "mirror"}
        Full and patched circuits offset the seed by ``instance * 1000003``; the
        mirror circuits keep the base seed and differ only through ``instance``.
    """
    if kind in ("full", "patched"):
        return base_seed + instance * INSTANCE_SEED_STRIDE
    if kind == "mirror":
        return base_seed
    raise ValueError(f"unknown circuit kind: {kind!r}")


# --------------------------------------------------------------------------- circuits
@dataclass
class RCSConfig:
    """Parameters of one random circuit.

    Attributes
    ----------
    num_qubits : int
        Number of logical qubits (61 in the experiment).
    cycles : int
        Number of cycles.
    matchings : dict
        Coupler colours ``{"A": [(a, b), ...], ...}`` in logical qubit indices.
    schedule : sequence of str
        Order in which the colours are applied, repeated as needed.
    removed_edges : iterable of edges, optional
        Couplers whose CZ gates are deleted in every cycle.  Passing the boundary
        edges of a patch partition gives the *patched* circuit of that partition.
    edge_filter : callable, optional
        ``edge_filter(cycle, colour, edges) -> edges`` applied after
        ``removed_edges``.  Used for the pseudo-patched mirror circuits, see
        :func:`rcs.patching.make_pseudo_patch_edge_filter`.
    add_barriers : bool
        Insert a barrier after every single-qubit layer and every CZ layer, as in
        the executed circuits.
    """

    num_qubits: int
    cycles: int
    matchings: Matchings
    schedule: Sequence[str] = ("A", "B", "C", "D")
    removed_edges: Optional[Iterable[Edge]] = None
    edge_filter: Optional[EdgeFilter] = None
    add_barriers: bool = True


def build_rcs_circuit(config: RCSConfig, seed: int, instance: int, measure: bool = True) -> QuantumCircuit:
    """Build one random circuit.

    Parameters
    ----------
    config : RCSConfig
        Circuit family parameters.
    seed, instance : int
        Arguments of the angle hash, see :func:`instance_seed`.
    measure : bool
        Append a terminal measurement of qubit ``i`` into classical bit ``i``.

    Returns
    -------
    QuantumCircuit
        Logical circuit on ``config.num_qubits`` qubits with gates ``rz``, ``rx``
        and ``cz``.
    """
    n = config.num_qubits
    removed: Set[Edge] = {tuple(sorted(e)) for e in (config.removed_edges or ())}
    qc = QuantumCircuit(n, n)
    qc.metadata = {"cycles": config.cycles, "schedule": list(config.schedule), "seed": seed, "instance": instance,
                   "removed_edges": sorted(list(e) for e in removed)}

    for cycle in range(config.cycles):
        for q in range(n):
            phi, theta, lam = haar_angles(seed, instance, cycle, q)
            qc.rz(phi, q)
            qc.rx(theta, q)
            qc.rz(lam, q)
        if config.add_barriers:
            qc.barrier()

        colour = config.schedule[cycle % len(config.schedule)]
        edges = [e for e in config.matchings[colour] if e[0] < n and e[1] < n]
        edges = [e for e in edges if tuple(sorted(e)) not in removed]
        if config.edge_filter is not None:
            edges = config.edge_filter(cycle, colour, edges)
        for a, b in edges:
            qc.cz(a, b)
        if config.add_barriers:
            qc.barrier()

    if measure:
        qc.measure(range(n), range(n))
    return qc


def build_mirror_circuit(config: RCSConfig, seed: int, instance: int) -> QuantumCircuit:
    """Build one mirror circuit ``prep(theta) . U . U^dagger`` followed by measurement.

    ``config.cycles`` is the number of cycles of the forward circuit ``U``; the mirror
    sequence ``U U^dagger`` therefore has depth ``2 * config.cycles``.  The preparation
    layer is ``RX(theta[q])`` on every qubit with a free parameter vector ``theta``:
    binding ``theta[q] = pi`` prepares ``|1>`` on qubit ``q`` and ``0`` leaves ``|0>``.
    The survival probability is the probability of measuring the prepared bitstring.
    """
    n = config.num_qubits
    theta = ParameterVector("theta", n)
    prep = QuantumCircuit(n)
    for q in range(n):
        prep.rx(theta[q], q)
    prep.barrier()

    # Build with measurements and strip them again: this also drops the then idle
    # classical register, exactly as in the executed circuits.
    forward = build_rcs_circuit(config, seed, instance, measure=True).remove_final_measurements(inplace=False)
    forward.barrier()
    forward.compose(forward.inverse(), inplace=True)
    qc = prep.compose(forward)
    qc.measure_all()
    qc.metadata = {"forward_cycles": config.cycles, "mirror_depth": 2 * config.cycles, "seed": seed, "instance": instance}
    return qc


def split_patched_circuit(circuit: QuantumCircuit, patch_qubits: Dict[str, Sequence[int]]
                          ) -> Tuple[Dict[str, QuantumCircuit], Dict[str, List[int]]]:
    """Split a patched circuit into its independent patch sub-circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        Patched circuit: no gate acts across two different patches.
    patch_qubits : dict
        ``{patch_name: [logical qubit, ...]}``.

    Returns
    -------
    sub_circuits : dict
        One measurement-free circuit per patch.
    index_map : dict
        ``index_map[patch][j]`` is the logical qubit carried by qubit ``j`` of the
        patch sub-circuit (qubits appear in increasing logical order).
    """
    n = circuit.num_qubits
    names = list(patch_qubits)
    members = {p: {q for q in patch_qubits[p] if q < n} for p in names}
    index_map: Dict[str, List[int]] = {p: [] for p in names}
    subs = {p: QuantumCircuit(len(members[p])) for p in names}

    for inst in circuit.remove_final_measurements(inplace=False).data:
        qidx = [circuit.find_bit(q).index for q in inst.qubits]
        for p in names:
            if inst.operation.name == "barrier":
                subs[p].barrier()
                continue
            if all(q in members[p] for q in qidx):
                for q in qidx:
                    if q not in index_map[p]:
                        index_map[p].append(q)
                subs[p].append(inst.operation, [index_map[p].index(q) for q in qidx])
    return subs, index_map


# --------------------------------------------------------------------------- compilation
def compile_native_locked(qc: QuantumCircuit, backend, physical_qubits: Sequence[int],
                          basis_gates: Sequence[str] = NATIVE_BASIS, optimization_level: int = 1,
                          seed_transpiler: int = 0) -> QuantumCircuit:
    """Transpile to the native gate set with a fixed one-to-one layout and no routing.

    Logical qubit ``i`` is placed on ``physical_qubits[i]``.  Every single-qubit gate
    becomes two SX pulses and virtual RZ rotations.  The function raises if the
    transpiler inserted SWAPs or any gate outside the basis, so the executed circuit is
    guaranteed to be the intended one.
    """
    if len(physical_qubits) != qc.num_qubits:
        raise ValueError(f"{len(physical_qubits)} physical qubits for a {qc.num_qubits}-qubit circuit")
    coupling = backend.coupling_map or CouplingMap(backend.configuration().coupling_map)
    tqc = transpile(qc, backend=backend, basis_gates=list(basis_gates), coupling_map=coupling,
                    initial_layout=list(physical_qubits), optimization_level=optimization_level,
                    seed_transpiler=seed_transpiler, routing_method="none", layout_method="trivial")
    ops = tqc.count_ops()
    if "swap" in ops:
        raise RuntimeError(f"transpilation inserted SWAPs: {dict(ops)}")
    unexpected = set(ops) - set(basis_gates) - {"barrier", "measure"}
    if unexpected:
        raise RuntimeError(f"transpilation introduced non-native gates {unexpected}")
    return tqc


def device_coupling_map(rows: int = 12, cols: int = 10) -> CouplingMap:
    """Bidirectional nearest-neighbour coupling map of the ``rows x cols`` device lattice."""
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    return CouplingMap(edges + [(b, a) for a, b in edges])


def compile_native_offline(qc: QuantumCircuit, physical_qubits: Sequence[int], rows: int = 12, cols: int = 10,
                           basis_gates: Sequence[str] = NATIVE_BASIS, optimization_level: int = 1,
                           seed_transpiler: int = 0) -> QuantumCircuit:
    """Same compilation as :func:`compile_native_locked` without access to the device.

    Uses the square-lattice coupling map of the 120-qubit processor instead of a backend
    object.  For the 32-cycle full circuit the result is gate-for-gate identical to the
    circuit retrieved from the hardware job.
    """
    tqc = transpile(qc, basis_gates=list(basis_gates), coupling_map=device_coupling_map(rows, cols),
                    initial_layout=list(physical_qubits), optimization_level=optimization_level,
                    seed_transpiler=seed_transpiler, routing_method="none", layout_method="trivial")
    ops = tqc.count_ops()
    unexpected = set(ops) - set(basis_gates) - {"barrier", "measure"}
    if "swap" in ops or unexpected:
        raise RuntimeError(f"unexpected gates after transpilation: {dict(ops)}")
    return tqc


def gate_signature(qc: QuantumCircuit, digits: int = 12) -> List[tuple]:
    """Order-preserving list ``(gate, qubits, parameters)`` used to compare circuits."""
    sig = []
    for inst in qc.data:
        params = []
        for p in inst.operation.params:
            try:
                params.append(round(float(p), digits))
            except TypeError:                       # unbound parameter of a mirror circuit
                params.append(str(p))
        sig.append((inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits), tuple(params)))
    return sig
