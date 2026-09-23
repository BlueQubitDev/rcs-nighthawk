# Contraction-cost estimator

Tensor-network estimate of the classical cost of random-circuit sampling, built on
[quimb](https://quimb.readthedocs.io) and [cotengra](https://cotengra.readthedocs.io).
It follows the cost methodology of Morvan et al., Nature 634, 328 (2024).

| quantity | meaning |
|---|---|
| `C_amp` | complex multiply-adds to contract one amplitude `<b|U|0>` with unlimited memory |
| width | log2 of the largest intermediate tensor of the best contraction tree |
| `W = 8 * kappa * N_s * F * C_amp` | machine FLOPs to draw `N_s` samples at fidelity `F` by frugal rejection sampling, `kappa = 10` |
| `t = W / (0.20 * 1.685e18)` | runtime on Frontier at 20 % of its theoretical peak |

## Files

| file | purpose |
|---|---|
| `sampling_cost.py` | library: `single_amplitude_cost`, `sparse_batched_cost`, `sampling_cost`, memory, fidelity and machine models |
| `flops_estimator.py` | Qiskit circuit -> quimb tensor network; quick amplitude-cost estimate |
| `load_rcs_tnsa_graph.py` | parser for Google's `.graph` tensor-network files |
| `depth_sweep.py` | **Table I**: `C_amp` and width of the 61-qubit circuits at 8-40 cycles |
| `estimate_sampling_cost.py` | command-line tool: one circuit -> cost report under several memory models |
| `run_rcs_tnsa_graphs.py` | **Appendix E**: the same search on the five published reference networks |
| `validation_table.py` | prints the Appendix E table from `results/validation` |
| `results/depth_sweep.json` | the values quoted in Table I and the independent repeat searches |
| `results/validation/*.json` | search results on the reference networks (5 h x 128 cores per contraction mode) |
| `data/google_rcs_tnsa/` | the reference networks, see `NOTICE.md` there |
| `docs/sampling_cost_notes.md` | technical note on the cost model |

## Usage

```bash
pip install quimb cotengra kahypar optuna qiskit        # versions in ../requirements.txt

python validation_table.py                                # Appendix E from the stored results
python depth_sweep.py --depths 8 12 --budget 60           # quick look; omit the options for the paper budgets
python estimate_sampling_cost.py --circuit ../data/circuits/full/d36_logical.qpy \
       --samples 1000000 --f-xeb 2.3e-3 --memory-preset ideal --opt-time 600 --depth 36
python run_rcs_tnsa_graphs.py --graphs all --opt-time 600 --parallel 8 --out results/rcs_tnsa_rerun.json
```

## Search budget and reproducibility

Path quality, not the cost formula, dominates the error of such estimates.  Without
`kahypar` the search falls back to greedy and costs come out orders of magnitude too
high.  Table I used wall-clock budgets from 2 minutes (8 cycles) to 1.5 hours (32-40
cycles) on 20 workers; independent 2.5-hour repeats at 32, 36 and 40 cycles differ from
the quoted values by 0.22, 0.14 and 0.06 decades.  The search is stochastic, so a rerun
reproduces the values only within this scatter, and every value is an upper bound on the
optimal contraction cost.  All numbers assume unlimited working memory.

## Search quality

The contraction cost reported by any of these scripts is the cost of the
best contraction tree *found*, so it is an upper bound whose tightness is
set by the search budget. Three tools quantify that:

| script | question |
|---|---|
| `run_plateau.py`, `plot_plateau.py` | how the best-found cost falls with search time, and where it plateaus |
| `run_memory_sweep.py` | how the cost rises when the working memory is capped, from a single GPU up to all of secondary storage |
| `export_rcs_tnsa_graph.py`, `parse_tnsa_results.py` | export a circuit to the `.graph`/`.groups` format of the simulated-annealing optimizer released with Ref. [12], and collect the results of its walkers |

The last pair is the independent cross-check: the same network searched by
a different optimizer. Note that the released driver prints the cost of its
sparse-output modes as `SCALAR_FACTOR * log2(FLOPs)` rather than
`log2(FLOPs) + log2(SCALAR_FACTOR)`; `parse_tnsa_results.py` documents and
undoes this when decoding, and reports single-amplitude (mode 1) costs as
raw complex multiply-adds.

`build_zcz3_rcs.py` reconstructs a geometry-approximate Zuchongzhi-3.0
network for the end-to-end check quoted in Appendix E.

`cluster/` holds the Slurm scripts used for the long searches; set the
allocation and paths for your own site.
