"""Fidelity estimators and the classical-cost model of the paper.

Patched linear XEB (Eqs. (5) and (6) of the paper)
--------------------------------------------------
For patch ``r`` with ``n_r`` qubits and ideal distribution ``p_r`` the per-shot score is
``s_r(x) = 2**n_r * p_r(x_r) - 1``.  Its shot mean divided by its noiseless value
``I_r = 2**n_r * sum_x p_r(x)**2 - 1`` (the *ideal XEB*, called the collision ratio in
Appendix F of the paper) estimates the patch fidelity ``F_r``.  For an anticoncentrated patch ``I_r``
equals ``(2**n_r - 1) / (2**n_r + 1)``; at shallower depth the normalisation removes the
upward bias of the plain linear XEB.  The circuit fidelity is ``F = prod_r F_r`` with a
second-order delta-method standard error that includes the covariance between patches.

Mirror benchmark
----------------
The survival probability (fraction of shots returning the prepared bitstring) estimates
the fidelity of the depth-``d`` sequence ``U U^dagger`` with a binomial standard error.

Combination
-----------
Values of the random circuit instances of one data point are combined by an
inverse-variance weighted mean.  Statistical errors are shot noise only.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .io import extract_bits

# --- classical cost model, Eq. (3) of the paper -------------------------------------
KAPPA_REJECTION = 10.0          #: amplitude evaluations per sample in frugal rejection sampling
MACHINE_FLOPS_PER_COMPLEX_OP = 8.0
FRONTIER_PEAK_FLOPS = 1.685e18  #: theoretical peak of the reference machine
FRONTIER_EFFICIENCY = 0.20      #: assumed sustained efficiency
SECONDS_PER_YEAR = 365.25 * 86400.0


def porter_thomas_limit(num_qubits: int) -> float:
    """Ideal XEB ``(N - 1) / (N + 1)`` of a Haar-random state on ``N = 2**n`` outcomes."""
    dim = 2.0 ** num_qubits
    return (dim - 1.0) / (dim + 1.0)


def ideal_xeb(probs: np.ndarray) -> float:
    """Noiseless linear XEB ``2**n * sum_x p(x)**2 - 1`` of a distribution."""
    probs = np.asarray(probs, dtype=float)
    return float(probs.size * np.dot(probs, probs) - 1.0)


def _delta_method_product_se(means: np.ndarray, cov_of_means: np.ndarray) -> float:
    """Second-order delta-method standard error of ``prod_k means[k]``."""
    k = len(means)
    grad = np.array([np.prod(np.delete(means, i)) for i in range(k)])
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            hess[i, j] = hess[j, i] = np.prod(np.delete(means, [i, j]))
    first = float(grad @ cov_of_means @ grad)
    a = cov_of_means @ hess
    second = 0.5 * float(np.trace(a @ a))
    return math.sqrt(max(first + second, 0.0))


def patched_xeb(shots: np.ndarray, patch_probs: Mapping[str, np.ndarray],
                index_map: Mapping[str, Sequence[int]]) -> Dict[str, object]:
    """Patched XEB fidelity of one circuit.

    Parameters
    ----------
    shots : array of uint64
        Measured bitstrings of the full register, see :mod:`rcs.io`.
    patch_probs : dict
        Ideal distribution of every patch sub-circuit (:func:`rcs.simulate.ideal_probabilities`).
    index_map : dict
        Logical qubits of every patch in sub-circuit order (:func:`rcs.circuits.split_patched_circuit`).

    Returns
    -------
    dict with
        ``fidelity``   product of the normalised patch fidelities,
        ``se``         its shot-noise standard error,
        ``patch_fidelity``, ``patch_ideal_xeb``, ``patch_qubits``  per-patch values,
        ``shots``      number of shots.
    """
    names = list(patch_probs)
    n_shots = int(len(shots))
    if n_shots < 2:
        raise ValueError("need at least two shots")
    norm = {p: ideal_xeb(patch_probs[p]) for p in names}
    scores = np.empty((len(names), n_shots))
    for row, p in enumerate(names):
        probs = np.asarray(patch_probs[p], dtype=float)
        scores[row] = (probs.size * probs[extract_bits(shots, index_map[p])] - 1.0) / norm[p]
    means = scores.mean(axis=1)
    cov = np.atleast_2d(np.cov(scores, bias=True)) / n_shots
    return {"fidelity": float(np.prod(means)), "se": _delta_method_product_se(means, cov),
            "patch_fidelity": {p: float(m) for p, m in zip(names, means)},
            "patch_ideal_xeb": {p: float(norm[p]) for p in names},
            "patch_qubits": {p: len(index_map[p]) for p in names}, "shots": n_shots}


def mirror_survival(hits, shots) -> Tuple[np.ndarray, np.ndarray]:
    """Survival probability and binomial standard error per mirror circuit instance.

    ``hits`` and ``shots`` have shape ``(instances, inputs)``; both are summed over the
    input strings of an instance.
    """
    hits = np.asarray(hits, dtype=float).sum(axis=-1)
    shots = np.asarray(shots, dtype=float).sum(axis=-1)
    p = hits / shots
    return p, np.sqrt(np.maximum(p * (1.0 - p), 1e-12) / shots)


def inverse_variance_mean(values, ses) -> Tuple[float, float]:
    """Inverse-variance weighted mean and its standard error."""
    values, ses = np.asarray(values, dtype=float), np.asarray(ses, dtype=float)
    w = 1.0 / ses ** 2
    return float(np.sum(w * values) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w)))


def fit_exponential_decay(depths, fidelities) -> Tuple[float, float]:
    """Unweighted least-squares fit of ``log10 F = log10 A + d * log10 f``.

    Returns ``(A, f)``: the prefactor and the fidelity per cycle.
    """
    slope, intercept = np.polyfit(np.asarray(depths, dtype=float), np.log10(np.asarray(fidelities, dtype=float)), 1)
    return float(10.0 ** intercept), float(10.0 ** slope)


def sampling_work(fidelity: float, log10_camp: float, num_samples: float = 1e6) -> float:
    """Machine FLOPs ``W = 8 * kappa * N_s * F * C_amp`` of bounded-fidelity sampling."""
    return MACHINE_FLOPS_PER_COMPLEX_OP * KAPPA_REJECTION * num_samples * fidelity * 10.0 ** log10_camp


def frontier_seconds(work: float) -> float:
    """Runtime ``t = W / (eta * P)`` on the reference machine."""
    return work / (FRONTIER_EFFICIENCY * FRONTIER_PEAK_FLOPS)
