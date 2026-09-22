# Sampling-cost model — technical note

This is the design note for `sampling_cost.py` + `estimate_sampling_cost.py`.
It explains *what* the estimator computes, *why* the naive
`C_samp = 10 · N_s · F_XEB · C_amp` formula is wrong, and *which* ingredients
of Morvan et al., *Nature* **634** (2024), Appendix G we faithfully reproduce
versus only approximate.


## 1. The two distinct tensor-network tasks

The estimator is built around a single conceptual point: the
single-amplitude cost and the sampling cost are answers to different
tensor-network questions.

### Task A — single fixed amplitude

Compute one closed scalar `⟨b | U | 0^N⟩` for one fixed bitstring `b`. The
tensor network is the gate circuit with both ends projected. cotengra finds
a contraction tree that minimises the total cost
`sum_{node} prod_{i ∈ involved(node)} bd_i`.

This is `flops_estimator.py`'s job, and is **Google's Mode 1**
(`rcs_tnsa/src/sa_optimizer.cpp:14-51`). The estimator stores this as

```
C_amp^0(n, d) = single-precision-complex FLOPs to contract one ⟨b|U|0⟩.
```

It is a *proxy* — it ignores memory, ignores slicing, ignores any sharing
across multiple bitstrings.

### Task B — batched sparse outputs

Compute `K` approximate probabilities `|⟨b_k | U | 0⟩|²` for `k = 1..K` with
*shared* contraction intermediates. The tensor network is the open-output
state `|ψ⟩ = U|0⟩` whose output legs are kept as free indices. Cotengra
finds a contraction tree for the open TN. **The cost is then modified by
the Morvan/rcs_tnsa formula**: at each pairwise contraction step, the
contribution of open-output legs to that step's cost is capped at K.

In code (`sampling_cost.py: _google_sparse_cost`):

```
for each parent node p with children l, r:
    involved   = legs(l) ∪ legs(r)
    log2_full  = sum_{i ∈ involved}                  log2 bd_i
    log2_open  = sum_{i ∈ involved ∩ open_outputs}   log2 bd_i
    log2_step  = log2_full − log2_open + min(log2_open, log2 K)
total log2 FLOPs = log2 sum_p 2^log2_step
```

This matches `rcs_tnsa/src/tensor_network.cpp:140` line-for-line.

The result is **Google's Mode 2/3 cost** (modulo the SA-optimizer path
choice and per-slice memoization — see Section 5):

```
C_sparse_batch(K; memory) = batched FLOPs over K sparse outputs.
```

The amortized per-probability cost is then

```
C_prob_amortized = C_sparse_batch(K) / K.
```

For deep RCS circuits where many intermediates are shared, this is
much smaller than `C_amp^0` once `K ≳ 2^width`; for shallow / structured
circuits the gap narrows.


## 2. From per-probability cost to sampling cost

Morvan's rejection-sampling convention: drawing `N_s` noisy samples
requires `N_prob = κ_rej · N_s` approximate probabilities, with
`κ_rej ≈ 10`.  The total sampling FLOPs is

```
                        ┌─ Convention A (baseline / independent amplitudes) ─┐
                        │                                                    │
C_samp_baseline = 8 · κ_rej · N_s · F_XEB · C_amp^0                          │
                                                                             │
                        ┌─ Convention B (batched sparse-output) ─────────────┘
                        │
C_samp_batched  = 8 · κ_rej · N_s · A_fidelity · C_prob_amortized
```

The `8` is the standard complex-FLOPs-to-real-FLOPs conversion
(`rcs_tnsa/src/utils.h:23`: one single-precision complex MAC ≈ 8 machine
FLOPs). Frontier-style runtime:

```
runtime = C_samp / (η · F_peak),    η = 0.20,   F_peak = 1.685 · 10^18 FLOP/s.
```

### Why "≈ C_amp" is wrong

The naive `C_samp = 10 · N_s · F_XEB · C_amp^0` is only convention A,
*assumed to be the relevant model*. It is the right answer in the
fantasy-memory regime where no batching wins are available and amplitudes
are evaluated independently. It is **wrong** for any realistic estimate
because it ignores the much cheaper batched-sparse-output mode that
Google/Morvan exploit. Convention B is the relevant one for matching
Morvan Table 1, and `C_samp_batched / C_samp_baseline` is the
amortization ratio.


## 3. Memory models

`MemoryModel.max_log2_size` is the only knob cotengra needs — it is the
log2 of the largest intermediate tensor permitted (in **elements**, not
bytes). Below it, no slicing; above it, cotengra slices until every
intermediate fits.  Presets:

| name           | log2 elems | translation                                       |
| -------------- | ---------- | ------------------------------------------------- |
| `ideal`        | ∞          | no slicing (Mode 1)                               |
| `gpu-80gb`     | ~33.3      | one H100, complex64                               |
| `gpu-128gb`    | ~34.0      | one H100 NVL / one MI300X, complex64              |
| `node-4tb`     | ~39.0      | fantasy single-node all-RAM, complex64            |
| `allram-32tb`  | ~42.0      | fantasy multi-node all-RAM                        |
| `secondary-1pb`| ~46.7      | secondary storage, *bandwidth ignored*            |

All but `ideal` add cotengra `slicing_opts={"target_size": 2**w}` and
`slicing_reconf_opts` (the latter merges slice groups whose memoization
gain exceeds the marginal slice cost — analogous to the
`log2_prefactors`/`flopss` book-keeping in
`rcs_tnsa/src/tensor_network.cpp:165-251`, but coarser).

A cap of `w` typically forces `n_slices ≈ 2^(width − w)` slices.  The
diagnostic table reports both `width` and `n_slices` so the user can see
how aggressively cotengra had to slice.


## 4. Fidelity discount

Two knobs:

* `F_XEB` — measured/target cross-entropy benchmarking fidelity. Multiplies
  *both* conventions. Plays the role of "how much state purity we need to
  preserve".
* `A_fidelity` — contraction-level fidelity discount applied only in
  convention B (`C_samp_batched`). **Default: `A_fidelity = F_XEB`**.

The reason for keeping `A_fidelity` separate from `F_XEB` is that Morvan's
"frugal contraction" actively *uses* the fidelity budget during the
contraction itself, dropping low-weight tensor paths and obtaining a cost
discount proportional to the surviving fidelity. We do **not** reimplement
that here. The default `A_fidelity = F_XEB` is a baseline that assumes a
budget exists, without modelling how the contractor would spend it.

If you want a placeholder-aggressive truncation you can pass e.g.
`--a-fidelity 0.0001` to model a 50× extra fidelity-based saving on top of
F_XEB. The diagnostic table prints A_fidelity explicitly so it is never
hidden.


## 5. What we faithfully reproduce — and what we do not

The estimator gives a transparent, faithful implementation of three of
the four ingredients in Morvan's cost model.

### Faithful

1. **The `min(log2_local_open, log2 K)` capping of open-output legs**
   (`rcs_tnsa/src/tensor_network.cpp:140`). Same formula, same
   summation strategy.
2. **The `SCALAR_FACTOR = 8` complex-to-real conversion**
   (`rcs_tnsa/src/utils.h:23`). Applied uniformly at the end.
3. **Frontier runtime conversion** (`η · F_peak`, η = 0.20, F_peak = 1.685·10^18).
   These come from the paper text, not the rcs_tnsa C++ codebase; our
   estimator applies them as the final step.
4. **`κ_rej ≈ 10` and `N_prob = κ_rej · N_s`**. Likewise paper text.
5. **Width-/memory-constrained slicing** is delegated to cotengra's
   `slicing_opts` machinery, which implements the same idea as the SA
   optimizer's binary search on slice count vs. width
   (`rcs_tnsa/src/tensor_network.cpp:455-480`).

### Approximate

6. **The contraction-tree path itself.**  cotengra optimises the open-TN
   contraction cost (`prod over involved`) using KaHyPar + greedy. Google's
   SA optimizer optimises the *sparse-output capped* cost — i.e. the path
   is K-aware from the start. Our path is therefore generally a (mild to
   moderate) over-estimate of the optimal K-aware path. The gap is
   smallest when `K ≳ 2^width` (cap is effectively inactive) and largest
   when `K ≪ 2^N` (the path "wastes" effort on output-dim factors that get
   capped away).

7. **Per-slice memoization.**  rcs_tnsa's
   `SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput`
   (`tensor_network.cpp:165-251`) keeps a `log2_prefactors` accumulator
   that lets contractions occurring *before* a slice group's "reach point"
   be paid only `2^prefactor` times rather than `2^total_slices` times.
   cotengra's `slicing_reconf_opts` does a coarser version of the same
   idea but does not split the bookkeeping per slice group.  When the
   cap forces many slices, our cost is therefore an **upper bound** on
   the Mode-3 cost.

### Not modelled

8. **Frugal-contraction fidelity truncation.**  Dropping low-weight
   sub-trees down to a target fidelity is the largest single saving in
   Morvan's cost model. We expose `A_fidelity` as a knob but do not
   implement the truncation. If you set `A_fidelity = F_XEB · ξ` with a
   user-chosen `ξ ≪ 1` you can model the *effect*, but you are not
   reproducing the *mechanism*.

9. **Per-slice secondary-storage I/O bandwidth.**  `SECONDARY-1pb` is a
   FLOP-only fantasy model; the disk-bandwidth bottleneck that any real
   secondary-storage simulator hits is not in the cost. Diagnostic notes
   flag this.


## 6. Behaviour the estimator reproduces qualitatively

Sanity checks that match Morvan Table 1 *trends*:

* **Old / shallow circuits.**  Large amortisation:
  `C_prob_amortized ≪ C_amp^0`, ratio B/A ≪ 1. Verified on n = 8–12
  toy circuits at K = 2^n (≈ 1% of baseline at K = 2^N).
* **Hard / deep circuits at tight memory.**  Slicing increases `n_slices`
  and inflates `C_sparse_batch`. The estimator reports the slice count
  and the per-slice memory; the user can see the blow-up.
* **K-saturation.**  As K → 2^N the capped cost converges to the full
  open-state cost. As K → 1 the capped cost approaches the closed-state
  cost (up to a path-optimisation factor — see "Approximate (6)" above).
* **All-RAM fantasy.**  `C_samp_baseline` matches `8 · κ_rej · N_s · F_XEB · C_amp^0`
  exactly by construction. With `A_fidelity = F_XEB` the ratio B/A is
  determined purely by `C_prob_amortized / C_amp^0`, which is the
  amortisation factor cotengra obtains. That ratio is reported in every
  row.


## 7. Bottom-line answer to the question

**Can our quimb estimator reproduce Morvan Table 1 numbers?**

Partially.

* The structural pieces — `C_amp^0`, the K-capped batched cost, slicing
  under memory caps, the `κ_rej` × `N_s` × fidelity outer multiplier, the
  8× complex factor and the Frontier-runtime conversion — are all
  present and faithful.
* The remaining quantitative gap to Morvan's numbers is dominated by
  three missing ingredients:
  1. A **K-aware contraction-order search** (we apply the cap *post hoc*
     to a path optimised for full open contraction).
  2. **Per-slice-group memoization** of the kind in `rcs_tnsa`'s
     `SlicedMemoized…` path.
  3. **Frugal-contraction fidelity truncation** — the largest single
     saving in Morvan's model.

The estimator's numbers should therefore be interpreted as **upper
bounds on the Morvan Mode-2/3 cost** for any given memory model.

The most impactful follow-up — were one to close the gap — would be (1):
a custom cotengra objective that uses `_google_sparse_cost` as the inner
cost function during path search, instead of the standard
`prod_{involved} bd`.  The other two are mechanically further from the
cotengra API and would warrant a separate implementation effort.


## 8. Reproducibility

```
python estimate_sampling_cost.py \
    --circuit qasms/q=60,d=20.qpy \
    --samples 1000000 \
    --f-xeb 0.002 \
    --batch-size 1000 \
    --memory-preset 'ideal,gpu-128gb,node-4tb,allram-32tb' \
    --opt-time 600 --max-repeats 1024 \
    --depth 20 \
    --out cost_report.json
```

emits one diagnostic row per memory model, writing both `cost_report.json`
and `cost_report.csv`.  Every row carries: `n`, `depth`, `n_gates`,
`F_XEB`, `A_fidelity`, `κ_rej`, `N_s`, `K`, `memory_model`, `memory_log2_cap`,
`C_amp^0`, `C_sparse_batch(K)`, `C_prob_amortized`, `C_samp_baseline`,
`C_samp_batched`, the runtime estimates on Frontier, the width and peak
size of both contractions, the number of slices, and the cotengra
wall-clock spent.
