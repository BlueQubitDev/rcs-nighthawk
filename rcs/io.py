"""Compact storage of measured bitstrings.

A measured bitstring of the 61 logical qubits is stored as one unsigned 64-bit integer
``x = sum_q bit_q * 2**q``.  This is ``int(key, 2)`` for a Qiskit counts key (rightmost
character = qubit 0).  All shots of one circuit form a sorted ``uint64`` array; shot
order carries no information because the hardware results were retrieved as counts.

Files
-----
``data/counts/patched_K{K}_d{depth}.npz``
    arrays ``partition{j}_instance{i}`` with the shots of that patched circuit.
``data/samples/full_d36.npz``
    array ``shots`` with the 10^6 samples of the full 36-cycle circuit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import numpy as np


def key_to_int(key: str) -> int:
    """Qiskit counts key -> integer (qubit 0 is the least significant bit)."""
    return int(key, 2)


def int_to_key(value: int, num_qubits: int) -> str:
    """Integer -> Qiskit counts key of ``num_qubits`` characters."""
    return format(int(value), f"0{num_qubits}b")


def counts_to_shots(counts: Mapping[str, int]) -> np.ndarray:
    """Expand a counts dictionary into a sorted array with one integer per shot."""
    values = np.fromiter((key_to_int(k) for k in counts), dtype=np.uint64, count=len(counts))
    reps = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    return np.sort(np.repeat(values, reps))


def shots_to_counts(shots: np.ndarray, num_qubits: int) -> Dict[str, int]:
    """Inverse of :func:`counts_to_shots`."""
    values, reps = np.unique(np.asarray(shots, dtype=np.uint64), return_counts=True)
    return {int_to_key(v, num_qubits): int(r) for v, r in zip(values, reps)}


def save_shots(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write named shot arrays to a compressed ``.npz`` file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.uint64) for k, v in arrays.items()})


def load_shots(path: str | Path) -> Dict[str, np.ndarray]:
    """Read every shot array of a ``.npz`` file."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def extract_bits(shots: np.ndarray, qubits) -> np.ndarray:
    """Sub-bitstring integers: bit ``j`` of the result is bit ``qubits[j]`` of each shot."""
    shots = np.asarray(shots, dtype=np.uint64)
    out = np.zeros(shots.shape, dtype=np.int64)
    for j, q in enumerate(qubits):
        out |= ((shots >> np.uint64(q)) & np.uint64(1)).astype(np.int64) << j
    return out
