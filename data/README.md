# Data

| path | content |
|---|---|
| `layout.json` | the 61 logical qubits and their physical qubits, the 102 couplers with their colour A-D, the schedule, base seed, the five patch partitions for K = 3 and K = 4, the ten mirror input strings, placement thresholds and execution options |
| `circuits/` | all 233 circuits as QPY (Qiskit 1.4, format 13) and OpenQASM 3, with `manifest.json` (gate counts, SHA-256) |
| `counts/patched_K{K}_d{depth}.npz` | measured bitstrings of the 15 patched circuits of one data point, arrays `partition{j}_instance{i}` |
| `counts/mirror_survival.json` | mirror benchmark: `hits[i][s]` of `shots[i][s]` executions of instance `i` prepared in input string `s` returned that string |
| `samples/full_d36.npz` | array `shots`: the 10^6 samples of the full 36-cycle circuit |
| `qpu_usage.json` | QPU execution seconds per circuit family and depth, and shot totals |
| `results/` | reference outputs of the analysis scripts |

## Conventions

* **Qubits.**  Physical qubit `p` of the device sits at row `p // 10`, column `p % 10` of
  the 12 x 10 lattice.  The experiment uses rows 1-8 and columns 2-9; physical qubits 17,
  55 and 62 were dropped by the calibration filters.  Logical qubits `0 .. 60` number the
  retained qubits in row-major order.
* **Bitstrings.**  One `uint64` per shot, `x = sum_q bit_q * 2**q` over logical qubits.
  This is `int(key, 2)` of a Qiskit counts key.  `rcs.io.shots_to_counts` converts back.
  Arrays are sorted; the shot order was not retained.
* **Mirror circuits.**  `circuits/mirror/dNN_instanceI` contains a preparation layer
  `RX(theta[q])`; bind `theta[q] = pi` where character `q` of the input string is `1`.
  `NN` is the depth of the sequence `U U^dagger` (forward circuit of `NN / 2` cycles).
* **Patched circuits.**  `circuits/patched/K{K}/d{depth}/partition{j}_instance{i}` is the
  random circuit of instance `i` with the boundary couplers of partition `j` removed.
* **Full circuits.**  `circuits/full/dNN_logical` for 4-40 cycles (used for the contraction
  costs) and `circuits/full/d36_executed`, the sampled circuit compiled to the native
  gates `rz, sx, cz` on the 120-qubit register.