# Running the rcs_tnsa cost estimator on Lonestar6 (TACC)

Scales the contraction-path search from a laptop's single core / 600 s to a
full Lonestar6 node (128 cores) × ~10 h per graph, so cotengra has ~10^4×
more search budget. Expected effect (from the laptop plateau experiment):
closed-amplitude cost drops toward Morvan Table 1 by ~3 orders of magnitude;
asymptotic gap is set by the path-quality ceiling, not by more time.

## Prereqs

- A TACC account with a Lonestar6 allocation (you need the allocation code
  for `#SBATCH -A`).
- This `rcs/` folder copied to `$SCRATCH/rcs` on ls6.

## 1. Copy the code over (from your laptop)

```bash
rsync -av --exclude '*.log' --exclude '*.npz' \
    /Users/vmac/Documents/work/EPFL+BQ/ongoing/rcs/ \
    <user>@ls6.tacc.utexas.edu:'$SCRATCH/rcs/'
```

## 2. One-time setup (on an ls6 login node)

```bash
ssh <user>@ls6.tacc.utexas.edu
cd $SCRATCH/rcs
bash cluster/setup_ls6.sh        # venv + deps + downloads the 5 .graph files
```

## 3. Edit the allocation code

Open `cluster/run_graphs.sbatch` and replace `REPLACE_WITH_ALLOCATION` on the
`#SBATCH -A` line with your TACC allocation.

## 4. Submit

```bash
cd $SCRATCH/rcs
sbatch cluster/run_graphs.sbatch          # job array, one graph per task
squeue -u $USER                           # watch the queue
tail -f logs/rcs_tnsa_*_*.out             # watch progress
```

Knobs (override at submit time):

```bash
# longer search (10h/mode => ~20h/graph; stay under the 48h queue limit):
OPT_TIME=36000 sbatch cluster/run_graphs.sbatch
# fewer cores, different batch size:
NCORE=64 K=10000 sbatch cluster/run_graphs.sbatch
```

## 5. Pull results back (from your laptop)

```bash
scp '<user>@ls6.tacc.utexas.edu:$SCRATCH/rcs/results/*.json' ./cluster_results/
```

Then re-make the comparison table locally:

```bash
python - <<'PY'
import json, math, glob
rows = [r for f in glob.glob('cluster_results/*.json') for r in json.load(open(f))]
for r in sorted(rows, key=lambda r: r['n_qubits']):
    gap = 10**(r['amp_log10_flops'] - r['morvan_log10_C_amp0'])
    print(f"{r['graph']:>22} n={r['n_qubits']:>3} "
          f"C_amp 10^{r['amp_log10_flops']:.2f} vs 10^{r['morvan_log10_C_amp0']:.2f} "
          f"(x{gap:.2g})  runtime {r['runtime_batched_yr']:.2g} vs {r['morvan_runtime_yr']:.2g} yr")
PY
```

## Notes / caveats

- **Each array task uses 1 node.** The array of 5 therefore uses 5 nodes
  concurrently. On the `normal` queue that's a modest request.
- **`--parallel 128`** sets cotengra's worker count. We pin BLAS threads to 1
  (`OMP_NUM_THREADS=1` etc.) so the 128 cotengra workers don't fight numpy.
- **What more time does NOT fix:** the K-aware contraction objective and
  frugal-fidelity truncation are still not implemented (see
  `sampling_cost_notes.md` §5). Those cap how close the *sampling* column can
  get to Morvan regardless of cores or walltime — roughly ×10–100 residual.
- **Memory-capped runs:** add `--memory-log2 34` (≈128 GB complex64) to the
  `srun` line in the sbatch to estimate under a GPU memory cap instead of the
  ideal/no-memory regime. Note the open-output slicing accounting caveat
  (`google_geometry_findings.md`).
