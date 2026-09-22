"""Google/Morvan-2024 style sampling-cost estimator built on quimb + cotengra.

Companion module to ``flops_estimator.py``. Where ``flops_estimator.py``
answers "how many FLOPs to compute one fixed amplitude <b|U|0>?", this module
answers "how many FLOPs to draw N_s noisy samples from a depth-d, n-qubit
random circuit, with batched sparse-output contraction under memory caps?"

The key claim of Morvan et al., Nature 634 (2024), Appendix G, is that
sampling cost is **not** ``10 * N_s * F_XEB * C_amp``. Rejection sampling
needs roughly ``kappa_rej * N_s`` approximate probabilities (with
``kappa_rej ~ 10``), but those probabilities are computed in batches that
share contraction intermediates. The per-probability amortized cost is much
smaller than the cost of a single isolated amplitude, *and* memory caps push
the cost back up via slicing.

Cost ingredients we model
-------------------------

1. ``C_amp^0`` — single-amplitude FLOPs, no memory constraint. Closed-bitstring
   contraction tree found by cotengra over <b|U|0>, then ``contraction_cost``.

2. ``C_sparse_batch(K; memory)`` — K-output batched FLOPs. Built from the
   **open**-output TN |psi>=U|0>. At each pairwise contraction step in the
   tree, the local cost is

       2^( sum_{i in involved} log2 bd_i
           - sum_{i in involved ∩ open_outputs} log2 bd_i
           + min( sum_{i in involved ∩ open_outputs} log2 bd_i, log2 K ) )

   summed over all non-leaf nodes of the tree. This is the formula in
   ``rcs_tnsa/src/tensor_network.cpp:140`` (``SlicedLog2FlopsGroupedSlicesSparseOutput``).

3. ``C_prob_amortized = C_sparse_batch(K) / K``.

4. ``N_prob = kappa_rej * N_s`` (default ``kappa_rej = 10``).

5. Fidelity discount, two conventions exposed side-by-side:

   * A (baseline, independent amplitudes):
     ``C_samp_baseline = SCALAR_FACTOR * N_prob * F_XEB * C_amp^0``
   * B (batched sparse-output):
     ``C_samp_batched  = SCALAR_FACTOR * N_prob * A_fidelity * C_prob_amortized``

   where ``A_fidelity`` defaults to ``F_XEB`` (Morvan's frugal-contraction
   fidelity truncation is **not** implemented here — see the technical note).

6. Memory models translate to a single number, ``max_log2_size`` (log2 of the
   largest allowed intermediate, in tensor elements). cotengra's
   ``slicing_opts={"target_size": 2**w}`` enforces it via slicing; the
   resulting number of slices and final cost are returned.

7. ``SCALAR_FACTOR = 8`` machine-FLOPs per single-precision complex FLOP, as
   in ``rcs_tnsa/src/utils.h:23``.

8. Runtime conversion:
       runtime_seconds = C_machine_FLOPs / (efficiency * peak_flops)
   with Frontier defaults ``peak_flops = 1.685e18`` and ``efficiency = 0.20``.

What this estimator does NOT do
-------------------------------

* Schmidt / Pauli-path truncation that drops sub-fidelity contractions
  (Morvan's "frugal contraction"). We expose ``A_fidelity`` as a knob and
  default it to ``F_XEB``; it is a baseline, not a faithful reproduction.
* Memoized exact reuse across slice groups (the
  ``log2_prefactors``/``flopss`` accumulator in
  ``rcs_tnsa/src/tensor_network.cpp:165-251``). We model the **batched
  sparse-output** dimension capping faithfully, but the per-slice memoization
  that further reduces cost on top is not implemented — so for Mode-3-like
  scenarios our estimate is an *upper bound* on the Google cost.
* Secondary-storage bandwidth. We expose a SECONDARY model only as a
  weak-bound flop count; bandwidth is ignored and flagged in the diagnostic.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cotengra as ctg
import quimb.tensor as qtn

# Reuse the Qiskit -> quimb converter from flops_estimator.py
from flops_estimator import qiskit_to_quimb


# --------------------------------------------------------------------------- #
# Constants & hardware
# --------------------------------------------------------------------------- #

#: Real-FLOPs per single-precision complex FLOP. From rcs_tnsa/src/utils.h:23.
SCALAR_FACTOR: int = 8

#: Default rejection-sampling oversampling factor. Morvan: "10^7 approximate
#: probabilities are needed to draw 10^6 samples".
KAPPA_REJ_DEFAULT: float = 10.0


@dataclasses.dataclass(frozen=True)
class MachineModel:
    """Single-precision FLOP rate of the hypothetical sampling supercomputer."""

    name: str = "Frontier"
    peak_flops: float = 1.685e18  # single-precision peak, FP32
    efficiency: float = 0.20      # sustained / peak
    scalar_factor: int = SCALAR_FACTOR

    @property
    def effective_flops(self) -> float:
        return self.efficiency * self.peak_flops

    def runtime_seconds(self, machine_flops: float) -> float:
        return machine_flops / self.effective_flops


# --------------------------------------------------------------------------- #
# Memory model
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class MemoryModel:
    """A contraction memory regime.

    Parameters
    ----------
    name : str
        Human-readable label used in diagnostic tables.
    max_log2_size : Optional[float]
        log2 of the maximum allowed intermediate tensor size, **in tensor
        elements** (not bytes). cotengra slices until every intermediate fits.
        Pass ``None`` for no slicing (ideal / Mode-1 behavior).
    bytes_per_element : float
        Used only by the byte-cap helpers below. complex64 = 8 bytes,
        complex128 = 16 bytes.
    notes : str
        Free-form annotation (e.g. "bandwidth ignored").
    """

    name: str
    max_log2_size: Optional[float]
    bytes_per_element: float = 8.0
    notes: str = ""

    @classmethod
    def ideal(cls) -> "MemoryModel":
        return cls(name="ideal", max_log2_size=None,
                   notes="no slicing; Mode-1 equivalent")

    @classmethod
    def width(cls, max_log2_size: float) -> "MemoryModel":
        return cls(name=f"width<={max_log2_size:g}",
                   max_log2_size=float(max_log2_size),
                   notes="explicit width cap")

    @classmethod
    def gpu_gb(cls, gb: float, bytes_per_element: float = 8.0) -> "MemoryModel":
        max_bytes = gb * (2 ** 30)
        max_elems = max_bytes / bytes_per_element
        w = math.log2(max_elems)
        return cls(name=f"GPU-{gb:g}GB",
                   max_log2_size=w,
                   bytes_per_element=bytes_per_element,
                   notes=f"{gb:g} GB / GPU, {bytes_per_element:g} B/elem")

    @classmethod
    def allram_tb(cls, tb: float, bytes_per_element: float = 8.0) -> "MemoryModel":
        max_bytes = tb * (2 ** 40)
        max_elems = max_bytes / bytes_per_element
        w = math.log2(max_elems)
        return cls(name=f"ALL-RAM-{tb:g}TB",
                   max_log2_size=w,
                   bytes_per_element=bytes_per_element,
                   notes=f"fantasy all-RAM, {tb:g} TB pooled")

    @classmethod
    def secondary(cls, pb: float = 1.0, bytes_per_element: float = 8.0) -> "MemoryModel":
        max_bytes = pb * (2 ** 50)
        max_elems = max_bytes / bytes_per_element
        w = math.log2(max_elems)
        return cls(name=f"SECONDARY-{pb:g}PB",
                   max_log2_size=w,
                   bytes_per_element=bytes_per_element,
                   notes="bandwidth ignored")


PRESET_MEMORY_MODELS: Dict[str, MemoryModel] = {
    "ideal":         MemoryModel.ideal(),
    "gpu-80gb":      MemoryModel.gpu_gb(80.0),
    "gpu-128gb":     MemoryModel.gpu_gb(128.0),
    "node-4tb":      MemoryModel.allram_tb(4.0),
    "allram-32tb":   MemoryModel.allram_tb(32.0),
    "secondary-1pb": MemoryModel.secondary(1.0),
}


# --------------------------------------------------------------------------- #
# Fidelity model
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class FidelityModel:
    """Fidelity / rejection-sampling parameters.

    ``a_fidelity`` is the *contraction-level* fidelity discount applied to
    the batched cost in convention B. It defaults to ``f_xeb`` (no truncation
    over and above what the closed contraction would naturally produce);
    Morvan's frugal-contraction approach effectively replaces it with the
    truncated-fidelity ratio, which we do not implement.
    """

    f_xeb: float
    n_samples: int
    kappa_rej: float = KAPPA_REJ_DEFAULT
    a_fidelity: Optional[float] = None  # default: copy f_xeb

    @property
    def a_eff(self) -> float:
        return self.f_xeb if self.a_fidelity is None else self.a_fidelity

    @property
    def n_prob(self) -> float:
        return self.kappa_rej * self.n_samples


# --------------------------------------------------------------------------- #
# cotengra optimizer factory
# --------------------------------------------------------------------------- #

def _have_kahypar() -> bool:
    try:
        import kahypar  # noqa: F401
        return True
    except Exception:
        return False


def _build_optimizer(
    memory_model: MemoryModel,
    opt_time_s: float,
    max_repeats: int,
    minimize: str,
    parallel: Any = False,
) -> Any:
    """Construct a cotengra optimizer matched to the memory regime.

    Strategy
    --------
    * If ``opt_time_s <= 0`` -> plain ``"greedy"`` (sub-second; recommended for
      smoke tests and very large TNs where HyperOptimizer is too slow).
    * Otherwise, build a HyperOptimizer.  We pick ``kahypar`` when available
      (the only path-search method that consistently beats greedy on the
      ~2000-tensor open-output TNs from Sycamore-scale circuits), and fall
      back to ``greedy`` if not.  ``optuna`` / ``cmaes`` is optional but
      strongly recommended for higher-quality hyperparameter search.
    * Memory-capped runs (``memory_model.max_log2_size`` not None) request
      slicing via ``slicing_opts`` + ``slicing_reconf_opts``, and switch to
      ``minimize="combo"`` (flops + memory) which produces slicing-friendly
      trees with much smaller resulting overhead than ``minimize="flops"``.
    """
    if opt_time_s is not None and opt_time_s <= 0:
        return "greedy"

    methods = ["kahypar", "greedy"] if _have_kahypar() else ["greedy"]
    effective_minimize = minimize
    if memory_model.max_log2_size is not None and minimize == "flops":
        effective_minimize = "combo"

    kwargs: Dict[str, Any] = dict(
        methods=methods,
        minimize=effective_minimize,
        max_time=opt_time_s,
        max_repeats=max_repeats,
        parallel=parallel,
        progbar=False,
    )
    if memory_model.max_log2_size is not None:
        target_size = 2 ** memory_model.max_log2_size
        kwargs["slicing_opts"] = {"target_size": target_size}
        kwargs["slicing_reconf_opts"] = {"target_size": target_size}
    return ctg.HyperOptimizer(**kwargs)


# --------------------------------------------------------------------------- #
# Step 1 — single-amplitude no-memory FLOP proxy (C_amp^0)
# --------------------------------------------------------------------------- #

def single_amplitude_cost(
    qc,
    memory_model: MemoryModel = MemoryModel.ideal(),
    opt_time_s: float = 10.0,
    max_repeats: int = 256,
    bitstring: Optional[str] = None,
    parallel: Any = False,
) -> Dict[str, Any]:
    """Cost of contracting one fixed amplitude ``<b|U|0>`` (Mode-1 style).

    Returns
    -------
    dict with keys:
      flops, log10_flops, width, peak_size, n_slices, n_gates, n_qubits,
      opt_time_s, memory_model, tree.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qcirc = qiskit_to_quimb(qc_no_meas)
    N = qcirc.N
    b = bitstring if bitstring is not None else "1" * N
    opt = _build_optimizer(memory_model, opt_time_s, max_repeats,
                           minimize="flops", parallel=parallel)

    t0 = time.time()
    info = qcirc.amplitude_rehearse(b=b, optimize=opt)
    dt = time.time() - t0
    tree = info["tree"]
    flops = float(tree.contraction_cost())
    width = float(tree.contraction_width())
    peak_size = float(tree.peak_size())
    sliced_inds = getattr(tree, "sliced_inds", {}) or {}
    n_slices = 1
    for ix in sliced_inds:
        n_slices *= int(tree.size_dict[ix])
    return {
        "flops": flops,
        "log10_flops": math.log10(flops) if flops > 0 else 0.0,
        "width": width,
        "peak_size": peak_size,
        "log2_peak_size": math.log2(peak_size) if peak_size > 0 else 0.0,
        "n_slices": n_slices,
        "n_gates": len(qc_no_meas.data),
        "n_qubits": N,
        "opt_time_s": dt,
        "memory_model": memory_model.name,
        "memory_log2_cap": memory_model.max_log2_size,
        "tree": tree,
    }


# --------------------------------------------------------------------------- #
# Step 2 — batched sparse-output cost (Google formula)
# --------------------------------------------------------------------------- #

def _logsumexp2(log2_terms: Iterable[float]) -> float:
    """Stable log2(sum_i 2**log2_terms[i]). Empty -> -inf."""
    terms = list(log2_terms)
    if not terms:
        return float("-inf")
    m = max(terms)
    if m == float("-inf"):
        return m
    s = sum(2.0 ** (t - m) for t in terms)
    return m + math.log2(s)


def _google_sparse_cost(
    tree,
    K: int,
    open_output_inds: Optional[set] = None,
) -> Dict[str, float]:
    """Apply Morvan/rcs_tnsa sparse-output capping to a contraction tree.

    Implements ``tensor_network.cpp:140``:

        for each pairwise contraction (parent p with children l, r):
            involved = legs(l) ∪ legs(r)
            log2_full   = sum_{i in involved} log2 bd_i
            log2_open   = sum_{i in involved ∩ open_outputs} log2 bd_i
            log2_capped = min(log2_open, log2 K)
            log2_step   = log2_full - log2_open + log2_capped
        flops = sum_steps 2**log2_step

    For ``K = 1`` open output indices contribute 0 — sparse-batched
    evaluation of a single configuration. For ``K >> prod(open_bd)`` the cap
    is inactive and the result reduces to the open-state contraction cost.

    Parameters
    ----------
    tree : cotengra.ContractionTree
    K : int
        Batch size (number of sparse output configurations to evaluate).
    open_output_inds : set, optional
        cotengra internal symbol names that correspond to open output edges.
        Defaults to ``set(tree.output)``.
    """
    if open_output_inds is None:
        open_output_inds = set(tree.output)

    size_dict = tree.size_dict
    log2_K = math.log2(max(K, 1))
    log2_terms: List[float] = []

    for p, l, r in tree.traverse():
        # ``traverse()`` yields only internal (parent) contractions, never
        # leaves, so no leaf-guard is needed. NB: node representation differs
        # by cotengra version (frozenset in <=0.7.x, int in >=0.8.0) — do not
        # call ``len(p)`` here; rely on ``get_involved`` instead.
        involved = tree.get_involved(p)  # dict-like: {ix: count}
        if not involved:
            continue
        log2_full = 0.0
        log2_open = 0.0
        for ix in involved:
            lb = math.log2(size_dict[ix])
            log2_full += lb
            if ix in open_output_inds:
                log2_open += lb
        log2_capped = min(log2_open, log2_K)
        log2_step = log2_full - log2_open + log2_capped
        log2_terms.append(log2_step)

    log2_flops = _logsumexp2(log2_terms)
    flops = 2.0 ** log2_flops if log2_flops > float("-inf") else 0.0
    return {
        "flops_batched": flops,
        "log10_flops_batched": (
            log2_flops * math.log10(2) if log2_flops > float("-inf") else 0.0
        ),
        "log2_flops_batched": log2_flops,
    }


def sparse_batched_cost(
    qc,
    K: int,
    memory_model: MemoryModel = MemoryModel.ideal(),
    opt_time_s: float = 30.0,
    max_repeats: int = 256,
    parallel: Any = False,
) -> Dict[str, Any]:
    """C_sparse_batch(K) — cost of K bitstring amplitudes with shared TN.

    Uses the open-output TN ``|psi> = U|0>``, lets cotengra find a
    contraction tree, then applies the Morvan/rcs_tnsa sparse-output
    capping at log2(K). The result is *not* an exact upper bound from a
    K-aware optimizer; cotengra optimizes the open-state cost, and the cap
    is applied post hoc. This is the closest practical analog of Google's
    Mode-2/3 in quimb without re-implementing their custom SA optimizer.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qcirc = qiskit_to_quimb(qc_no_meas)
    N = qcirc.N
    psi = qcirc.psi  # open-output TN

    opt = _build_optimizer(memory_model, opt_time_s, max_repeats,
                           minimize="flops", parallel=parallel)

    t0 = time.time()
    tree = psi.contraction_tree(optimize=opt)
    dt = time.time() - t0

    capped = _google_sparse_cost(tree, K=K, open_output_inds=set(tree.output))
    # Sanity: the "no cap" full-state cost
    full_flops = float(tree.contraction_cost())
    width = float(tree.contraction_width())
    peak_size = float(tree.peak_size())
    sliced_inds = getattr(tree, "sliced_inds", {}) or {}
    n_slices = 1
    for ix in sliced_inds:
        n_slices *= int(tree.size_dict[ix])

    flops_batched = capped["flops_batched"]
    log10_flops_batched = capped["log10_flops_batched"]
    flops_amortized = flops_batched / max(K, 1)
    return {
        "K": int(K),
        "flops_batched": flops_batched,
        "log10_flops_batched": log10_flops_batched,
        "flops_amortized": flops_amortized,
        "log10_flops_amortized": (
            math.log10(flops_amortized) if flops_amortized > 0 else 0.0
        ),
        "flops_full_state": full_flops,
        "log10_flops_full_state": (
            math.log10(full_flops) if full_flops > 0 else 0.0
        ),
        "width": width,
        "peak_size": peak_size,
        "log2_peak_size": math.log2(peak_size) if peak_size > 0 else 0.0,
        "n_slices": n_slices,
        "n_gates": len(qc_no_meas.data),
        "n_qubits": N,
        "opt_time_s": dt,
        "memory_model": memory_model.name,
        "memory_log2_cap": memory_model.max_log2_size,
        "tree": tree,
    }


# --------------------------------------------------------------------------- #
# Step 3-5 — combine into a full sampling-cost estimate
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SamplingCostReport:
    n_qubits: int
    depth: Optional[int]
    n_gates: int
    f_xeb: float
    n_samples: int
    kappa_rej: float
    K: int
    a_fidelity: float
    # raw FLOP counts (single-precision complex)
    flops_amp0: float
    flops_sparse_batch: float
    flops_amortized: float
    # machine FLOPs (× SCALAR_FACTOR × kappa_rej × n_samples × fidelity)
    machine_flops_samp_baseline: float
    machine_flops_samp_batched: float
    # diagnostics
    width_amp0: float
    width_batched: float
    log2_peak_size_amp0: float
    log2_peak_size_batched: float
    n_slices_amp0: int
    n_slices_batched: int
    memory_model: str
    memory_log2_cap: Optional[float]
    machine: str
    peak_flops: float
    efficiency: float
    runtime_baseline_s: float
    runtime_batched_s: float
    ratio_batched_over_baseline: float
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # convenience log10 fields
        for k in (
            "flops_amp0",
            "flops_sparse_batch",
            "flops_amortized",
            "machine_flops_samp_baseline",
            "machine_flops_samp_batched",
        ):
            v = d[k]
            d[f"log10_{k}"] = math.log10(v) if v > 0 else 0.0
        return d


def sampling_cost(
    qc,
    fidelity: FidelityModel,
    K: int,
    memory_model: MemoryModel = MemoryModel.ideal(),
    machine: MachineModel = MachineModel(),
    opt_time_s: float = 30.0,
    max_repeats: int = 256,
    depth: Optional[int] = None,
    notes: str = "",
    parallel: Any = False,
    precomputed_amp: Optional[Dict[str, Any]] = None,
    precomputed_batch: Optional[Dict[str, Any]] = None,
) -> SamplingCostReport:
    """Full Morvan-style sampling-cost estimate for one circuit.

    Combines:
      * C_amp^0 from ``single_amplitude_cost`` (Mode-1 proxy)
      * C_sparse_batch(K) from ``sparse_batched_cost`` (Mode-2/3 proxy)
      * The kappa_rej + N_s + fidelity outer multipliers (paper-text math)
      * The ``SCALAR_FACTOR`` complex-FLOP-to-machine-FLOP conversion
      * Frontier-style runtime conversion

    ``precomputed_amp`` / ``precomputed_batch`` allow re-using cached cotengra
    results (e.g. from path_search_cpu.ipynb) instead of re-optimizing.

    Notes
    -----
    Convention A baseline:
        C_samp_baseline = SCALAR_FACTOR * kappa_rej * N_s * F_XEB * C_amp^0
    Convention B batched:
        C_samp_batched  = SCALAR_FACTOR * kappa_rej * N_s * A_fidelity * (C_sparse_batch(K) / K)
    Both are returned. The ratio ``C_samp_batched / C_samp_baseline`` quantifies
    the amortization gain (or memory blow-up loss if > 1).
    """
    amp = precomputed_amp or single_amplitude_cost(
        qc, memory_model=memory_model, opt_time_s=opt_time_s,
        max_repeats=max_repeats, parallel=parallel,
    )
    batch = precomputed_batch or sparse_batched_cost(
        qc, K=K, memory_model=memory_model, opt_time_s=opt_time_s,
        max_repeats=max_repeats, parallel=parallel,
    )

    a_fid = fidelity.a_eff
    sf = machine.scalar_factor
    n_prob = fidelity.n_prob

    flops_amp0 = amp["flops"]
    flops_batch = batch["flops_batched"]
    flops_amort = batch["flops_amortized"]

    samp_baseline = sf * n_prob * fidelity.f_xeb * flops_amp0
    samp_batched = sf * n_prob * a_fid * flops_amort

    ratio = samp_batched / samp_baseline if samp_baseline > 0 else float("nan")
    return SamplingCostReport(
        n_qubits=amp["n_qubits"],
        depth=depth,
        n_gates=amp["n_gates"],
        f_xeb=fidelity.f_xeb,
        n_samples=fidelity.n_samples,
        kappa_rej=fidelity.kappa_rej,
        K=int(K),
        a_fidelity=a_fid,
        flops_amp0=flops_amp0,
        flops_sparse_batch=flops_batch,
        flops_amortized=flops_amort,
        machine_flops_samp_baseline=samp_baseline,
        machine_flops_samp_batched=samp_batched,
        width_amp0=amp["width"],
        width_batched=batch["width"],
        log2_peak_size_amp0=amp["log2_peak_size"],
        log2_peak_size_batched=batch["log2_peak_size"],
        n_slices_amp0=amp["n_slices"],
        n_slices_batched=batch["n_slices"],
        memory_model=memory_model.name,
        memory_log2_cap=memory_model.max_log2_size,
        machine=machine.name,
        peak_flops=machine.peak_flops,
        efficiency=machine.efficiency,
        runtime_baseline_s=machine.runtime_seconds(samp_baseline),
        runtime_batched_s=machine.runtime_seconds(samp_batched),
        ratio_batched_over_baseline=ratio,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Diagnostic table formatting
# --------------------------------------------------------------------------- #

def _fmt_seconds(s: float) -> str:
    if not math.isfinite(s) or s <= 0:
        return "n/a"
    year = 365.25 * 86400
    if s >= year:
        return f"{s / year:.2f} yr"
    if s >= 86400:
        return f"{s / 86400:.2f} d"
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 60:
        return f"{s / 60:.2f} min"
    if s >= 1:
        return f"{s:.2f} s"
    return f"{s * 1e3:.2f} ms"


def format_report(report: SamplingCostReport, *, header: bool = True) -> str:
    d = report.as_dict()
    lines = []
    if header:
        lines.append("=" * 78)
        lines.append("Sampling-cost report  (Morvan/rcs_tnsa-style cost model)")
        lines.append("=" * 78)
    lines.append(
        f"  circuit         : n={d['n_qubits']}, depth={d['depth']}, "
        f"gates={d['n_gates']}"
    )
    lines.append(
        f"  fidelity        : F_XEB={d['f_xeb']:.4g}, "
        f"A_fidelity={d['a_fidelity']:.4g}, kappa_rej={d['kappa_rej']:g}, "
        f"N_s={d['n_samples']:g}"
    )
    lines.append(
        f"  batch           : K={d['K']}, complex-scalar-factor={SCALAR_FACTOR}"
    )
    lines.append(
        f"  memory          : {d['memory_model']}"
        + (f" (log2 cap = {d['memory_log2_cap']:g})"
           if d['memory_log2_cap'] is not None else "")
    )
    lines.append(
        f"  machine         : {d['machine']}, peak {d['peak_flops']:.3g} FLOP/s, "
        f"eff {d['efficiency']:.0%}"
    )
    lines.append("-" * 78)
    lines.append(
        f"  C_amp^0           = 10^{d['log10_flops_amp0']:6.2f}  "
        f"(width {d['width_amp0']:.1f}, log2 peak {d['log2_peak_size_amp0']:.1f}, "
        f"slices {d['n_slices_amp0']})"
    )
    lines.append(
        f"  C_sparse_batch(K) = 10^{d['log10_flops_sparse_batch']:6.2f}  "
        f"(width {d['width_batched']:.1f}, log2 peak {d['log2_peak_size_batched']:.1f}, "
        f"slices {d['n_slices_batched']})"
    )
    lines.append(
        f"  C_prob_amortized  = 10^{d['log10_flops_amortized']:6.2f}  "
        f"= C_sparse_batch(K) / K"
    )
    lines.append(
        f"  C_samp_baseline   = 10^{d['log10_machine_flops_samp_baseline']:6.2f} "
        f"machine FLOPs  -> {_fmt_seconds(d['runtime_baseline_s'])}"
    )
    lines.append(
        f"  C_samp_batched    = 10^{d['log10_machine_flops_samp_batched']:6.2f} "
        f"machine FLOPs  -> {_fmt_seconds(d['runtime_batched_s'])}"
    )
    lines.append(
        f"  ratio (B/A)       = {d['ratio_batched_over_baseline']:.3g}"
    )
    if d["notes"]:
        lines.append(f"  notes: {d['notes']}")
    if header:
        lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sweep helper
# --------------------------------------------------------------------------- #

def sweep_sampling_cost(
    circuits: Dict[Any, Any],  # {label: qiskit.QuantumCircuit}
    *,
    fidelity: FidelityModel,
    K: int,
    memory_models: Sequence[MemoryModel] = (MemoryModel.ideal(),),
    machine: MachineModel = MachineModel(),
    opt_time_s: float = 30.0,
    max_repeats: int = 256,
    depths: Optional[Dict[Any, int]] = None,
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """Run ``sampling_cost`` over a {label: qc} mapping × memory models.

    Returns a flat list of dicts, one per (label, memory_model). Use
    ``pandas.DataFrame(rows)`` downstream.
    """
    rows: List[Dict[str, Any]] = []
    depths = depths or {}
    for label, qc in circuits.items():
        depth = depths.get(label)
        for mm in memory_models:
            try:
                rep = sampling_cost(
                    qc, fidelity=fidelity, K=K, memory_model=mm,
                    machine=machine, opt_time_s=opt_time_s,
                    max_repeats=max_repeats, depth=depth,
                    notes=f"label={label}",
                )
                row = rep.as_dict()
                row["label"] = label
                rows.append(row)
                if print_table:
                    print(format_report(rep, header=False))
                    print("-" * 78)
            except Exception as e:
                row = {
                    "label": label,
                    "memory_model": mm.name,
                    "error": f"{e.__class__.__name__}: {e}",
                }
                rows.append(row)
                if print_table:
                    print(f"  [{label} | {mm.name}]  FAILED: {row['error']}")
    return rows
