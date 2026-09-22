"""Exact ideal output probabilities of patch sub-circuits (at most 21 qubits)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit


def ideal_probabilities(circuit: QuantumCircuit, threads: Optional[int] = None) -> np.ndarray:
    """Ideal output distribution of a measurement-free circuit by statevector simulation.

    Entry ``x`` of the returned array is the probability of the bitstring whose bit
    ``j`` is the outcome of circuit qubit ``j`` (Qiskit little-endian ordering).
    Double precision; a 21-qubit patch needs about 32 MB.
    """
    from qiskit_aer import AerSimulator

    options = {"method": "statevector", "precision": "double"}
    if threads:
        options["max_parallel_threads"] = threads
    sim = AerSimulator(**options)
    qc = circuit.copy()
    qc.save_statevector()
    state = np.asarray(sim.run(qc).result().get_statevector())
    return np.abs(state) ** 2
