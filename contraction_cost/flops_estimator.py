"""Tensor-network FLOPs estimator for RCS-style circuits.

Builds a quimb tensor network from a qiskit circuit, then uses cotengra to
find a low-FLOPs contraction tree for the single amplitude ``<1^N|U|0^N>``.
Reports ``log10`` FLOPs (cost of the best tree cotengra found within the time
budget) and the contraction width (``log2`` of the largest intermediate
tensor — controls peak memory).

Notes
-----
* The all-ones bitstring is a generic dense projection; TN structure is set by
  the circuit's gate connectivity, not the bitstring choice, so the cost is
  representative of any other bitstring.
* Estimates apply to the *uncompiled* logical circuit. Native-basis
  transpilation adds a small constant overhead that doesn't change the
  asymptotic cost.
* One instance per (config, depth) is enough — TN structure is the same for
  every random-angle instance.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

import cotengra as ctg
import quimb.tensor as qtn


def qiskit_to_quimb(qc):
    """Convert a qiskit circuit to a quimb tensor-network Circuit.

    Supported gates:
        rz, rx, ry, sx, x, h, s, sdg, t, tdg, u, u1, u2, u3,
        cz, cx, cnot, swap, iswap, rzz, rxx, ryy.
    Barriers, measurements, resets, and identities are skipped.
    Extend the dispatch below to support new gates.
    """
    qcirc = qtn.Circuit(N=qc.num_qubits)
    for inst in qc.data:
        op = inst.operation
        name = op.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name in ("barrier", "measure", "reset", "id"):
            continue
        params = [float(p) for p in op.params]
        # --- single-qubit ---
        if name == "rz":
            qcirc.apply_gate("RZ", params[0], qubits[0])
        elif name == "rx":
            qcirc.apply_gate("RX", params[0], qubits[0])
        elif name == "ry":
            qcirc.apply_gate("RY", params[0], qubits[0])
        elif name == "sx":
            qcirc.apply_gate("X_1_2", qubits[0])
        elif name == "x":
            qcirc.apply_gate("X", qubits[0])
        elif name == "y":
            qcirc.apply_gate("Y", qubits[0])
        elif name == "z":
            qcirc.apply_gate("Z", qubits[0])
        elif name == "h":
            qcirc.apply_gate("H", qubits[0])
        elif name == "s":
            qcirc.apply_gate("S", qubits[0])
        elif name == "sdg":
            qcirc.apply_gate("SDG", qubits[0])
        elif name == "t":
            qcirc.apply_gate("T", qubits[0])
        elif name == "tdg":
            qcirc.apply_gate("TDG", qubits[0])
        elif name in ("u", "u3"):
            # u(theta, phi, lambda) — QASM 3 / qiskit-modern 3-parameter
            qcirc.apply_gate("U3", params[0], params[1], params[2], qubits[0])
        elif name == "u2":
            # u2(phi, lambda) == U3(pi/2, phi, lambda)
            qcirc.apply_gate("U3", math.pi / 2, params[0], params[1], qubits[0])
        elif name == "u1":
            # u1(lambda) == RZ(lambda) up to global phase
            qcirc.apply_gate("RZ", params[0], qubits[0])
        # --- two-qubit ---
        elif name == "cz":
            qcirc.apply_gate("CZ", qubits[0], qubits[1])
        elif name in ("cx", "cnot"):
            qcirc.apply_gate("CNOT", qubits[0], qubits[1])
        elif name == "swap":
            qcirc.apply_gate("SWAP", qubits[0], qubits[1])
        elif name == "iswap":
            qcirc.apply_gate("ISWAP", qubits[0], qubits[1])
        elif name == "rzz":
            qcirc.apply_gate("RZZ", params[0], qubits[0], qubits[1])
        elif name == "rxx":
            qcirc.apply_gate("RXX", params[0], qubits[0], qubits[1])
        elif name == "ryy":
            qcirc.apply_gate("RYY", params[0], qubits[0], qubits[1])
        else:
            raise NotImplementedError(
                f"FLOPs estimator: gate '{name}' not handled. "
                "Extend qiskit_to_quimb() to support it."
            )
    return qcirc


def estimate_amplitude_flops(qc, opt_time_s: float = 10, max_repeats: int = 1024) -> Dict[str, float]:
    """Estimate FLOPs to compute ``<1^N|U|0^N>`` for the given qiskit circuit."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qcirc = qiskit_to_quimb(qc_no_meas)
    N = qcirc.N
    opt = ctg.HyperOptimizer(
        methods=["kahypar", "greedy"],
        minimize="flops",
        max_time=opt_time_s,
        max_repeats=max_repeats,
        parallel=False,
        progbar=False,
    )
    info = qcirc.amplitude_rehearse(b="1" * N, optimize=opt)
    tree = info["tree"]
    flops = float(tree.contraction_cost())
    return {
        "flops": flops,
        "log10_flops": math.log10(flops) if flops > 0 else 0.0,
        "width": float(tree.contraction_width()),
        "n_gates": len(qc_no_meas.data),
        "n_qubits": N,
    }


def estimate_circuits_flops(
    circuits: Dict[Any, "qtn.Circuit"],
    opt_time_s: float = 100,
    max_repeats: int = 1024,
    key_label: str = "depth",
) -> Dict[Any, Dict[str, float]]:
    """Run ``estimate_amplitude_flops`` over a ``{key: qc}`` mapping.

    Prints a one-line-per-circuit table to stdout. Wall-clock ≈ ``opt_time_s``
    × ``len(circuits)``. Failures are caught per-circuit and printed; the
    returned dict only contains successful estimates.
    """
    estimates: Dict[Any, Dict[str, float]] = {}
    print("=" * 78)
    print(f"Tensor-network FLOPs estimates for <1^N|U|0^N>  "
          f"(cotengra budget: {opt_time_s}s/circuit)")
    print("=" * 78)
    print(f"{key_label:>6} {'qubits':>7} {'gates':>7} "
          f"{'log10 FLOPs':>12} {'width':>7} {'opt time':>10}")
    print("-" * 78)
    for key, qc in circuits.items():
        t0 = time.time()
        try:
            est = estimate_amplitude_flops(qc, opt_time_s=opt_time_s,
                                           max_repeats=max_repeats)
            dt = time.time() - t0
            estimates[key] = est
            print(f"{key:>6} {est['n_qubits']:>7} "
                  f"{est['n_gates']:>7} {est['log10_flops']:>12.2f} "
                  f"{est['width']:>7.1f} {dt:>9.1f}s")
        except Exception as exc:
            dt = time.time() - t0
            print(f"{key:>6}: FAILED "
                  f"({exc.__class__.__name__}: {exc}) [{dt:.1f}s]")
    print("=" * 78)
    return estimates
