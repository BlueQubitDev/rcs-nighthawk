# Random-circuit sampling on the IBM Nighthawk r2 processor

Code, circuits and measured data for the paper

> *Quantum computational advantage in random-circuit sampling on a 120-qubit
> superconducting quantum computer* (reference to be added).

The experiment runs random circuits on 61 qubits of the 120-qubit square-lattice
processor `ibm_phoenix` through the standard Qiskit Runtime cloud stack.  Circuit fidelity
is estimated from 4 to 40 cycles with a mirror benchmark and from 20 to 40 cycles with
three- and four-patch cross-entropy benchmarking (XEB); the full 36-cycle circuit
(918 CZ gates) is sampled 10^6 times.  The classical sampling cost is estimated from
tensor-network contraction costs.

## Repository layout

```
rcs/                  library: layout, placement, patching, circuits, simulation, estimators, storage, figures
scripts/              build_circuits.py, run_experiment.py, plot_partitions.py, check_release.py
analysis/             compute_patch_xeb.py -> fidelity_vs_depth.py -> sampling_cost_table.py; run_accounting.py
contraction_cost/     tensor-network contraction-cost estimator, Table I sweep, Appendix E validation
spoofing/             severing-attack scores and adversarial cut search (Appendix F)
data/                 layout, circuits, measured bitstrings, samples, reference results
figures/              figures produced by the scripts
```

## Installation

Python 3.11 and the packages of `requirements.txt` (`pip install -r requirements.txt`).
The analysis needs only `qiskit`, `qiskit-aer`, `numpy` and `matplotlib`;
`qiskit-ibm-runtime` is needed only to run on hardware, and `quimb`, `cotengra`,
`kahypar`, `optuna` only for `contraction_cost/`.

## Reproducing the results

All commands are run from the repository root.

| result | command | runtime | output |
|---|---|---|---|
| all 233 circuits, bit for bit | `python scripts/build_circuits.py --verify` | 1 min | confirms `data/circuits` |
| patched XEB of every circuit | `python analysis/compute_patch_xeb.py` | 4 min on 20 cores | `data/results/patch_xeb.json` |
| fidelity vs depth, fit, Fig. 2, anticoncentration | `python analysis/fidelity_vs_depth.py` | seconds | `data/results/fidelity_vs_depth.json`, `figures/fig2_*` |
| Table I and the headline cost | `python analysis/sampling_cost_table.py` | seconds | `data/results/sampling_cost_table.json` |
| shots and QPU time (Appendix D) | `python analysis/run_accounting.py` | seconds | printed |
| contraction costs of Table I | `python contraction_cost/depth_sweep.py` | 6.5 h on 20 cores | `contraction_cost/results/depth_sweep_rerun.json` |
| adversarial cuts of the coupler graph (Appendix F) | `python spoofing/min_cut.py --circuit data/circuits/full/d36_logical.qpy` | 1 min | `spoofing/results/min_cuts.json` |
| severing-attack scores (Appendix F) | `python spoofing/spoof_ensemble.py ...` (see `spoofing/README.md`) | 40 min per region | `spoofing/results/ensemble_*.json` |
| validation table (Appendix E) | `python contraction_cost/validation_table.py` | seconds | printed |
| patch partitions (Fig. 1c) | `python scripts/plot_partitions.py` | seconds | `figures/fig1c_*` |
| consistency check of the whole release | `python scripts/check_release.py` | 1 min | printed |

The reference outputs of all steps are included in `data/results`, so each step can be
run on its own.

### Numbers produced by this repository

| quantity | value |
|---|---|
| mirror fit | F(d) = 0.326 x 0.872^d |
| fitted fidelity at 32, 36, 40 cycles | 4.0e-3, 2.3e-3, 1.3e-3 |
| measured at 36 cycles (mirror, 3-patch, 4-patch) | 2.2e-3, 1.8e-3, 2.0e-3 |
| error per qubit per cycle | 2.3e-3 |
| best-found log10 C_amp at 32, 36, 40 cycles | 19.83, 21.82, 22.03 |
| sampling cost at 36 cycles | 1.2e27 machine FLOPs, about 110 Frontier years |
| collision ratio of 20/21-qubit patches at 20, 32, 40 cycles | 1.08, 1.01, 1.00 |
| total shots, QPU time | 9 055 000, 668 s (19 s for the 10^6 samples) |

## Running on hardware

`scripts/run_experiment.py` performs the calibration-aware placement, builds the circuits,
submits them in one Qiskit Runtime batch and stores the results in the format of `data/`.
Credentials are taken from the environment variables `QISKIT_IBM_TOKEN` and
`QISKIT_IBM_INSTANCE`; nothing is submitted without `--submit`, and `--offline` builds
every job payload without any network access.  The script is a cleaned-up version of the
notebook that ran the experiment; only the offline path was executed after the clean-up.

## Data

See `data/README.md` for formats and conventions.  In short: bitstrings are stored as one
unsigned 64-bit integer per shot with logical qubit `q` in bit `q`; logical qubits
`0 .. 60` map to the physical qubits listed in `data/layout.json`.  The counts of every
circuit are summed over several executions of the identical circuit on the same qubits.  
For the mirror benchmark only the number of shots returning the prepared bitstring was retained.