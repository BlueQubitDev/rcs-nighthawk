#!/bin/bash
#SBATCH -J rcs_tnsa            # job name
#SBATCH -o logs/%x_%A_%a.out   # stdout: <name>_<jobid>_<arrayidx>.out
#SBATCH -e logs/%x_%A_%a.err   # stderr
#SBATCH -p normal              # Lonestar6 CPU queue (max 48h)
#SBATCH -N 1                   # 1 node per array task
#SBATCH --ntasks-per-node=1    # 1 task; cotengra forks loky workers within it
#SBATCH --cpus-per-task=128    # give the task all 128 cores (else loky is pinned)
#SBATCH -t 14:00:00            # walltime: ~10-11h compute + margin
#SBATCH --array=0-4            # one array task per Google graph (5 total)
#SBATCH -A <allocation>         # TACC allocation
# #SBATCH --mail-type=END             # uncomment + set --mail-user to get emailed
# #SBATCH --mail-user=you@example.com
#
# Lonestar6 node = 2x AMD EPYC 7763 = 128 physical cores, 256 GB RAM.
# Each array task runs ONE graph with 128-way parallel cotengra search.
#
# Submit:   sbatch cluster/run_graphs.sbatch
# Watch:    squeue -u $USER ;  tail -f logs/rcs_tnsa_*_*.out
# Pull back: scp 'ls6:$SCRATCH/rcs/results/*.json' .

set -euo pipefail

# ---- knobs ----------------------------------------------------------------
OPT_TIME="${OPT_TIME:-18000}"      # cotengra budget per mode [s]; 18000 = 5h
                                   # (Mode-1 + Mode-3 => ~10h total per graph)
K="${K:-1000}"                     # sparse-output batch size
SAMPLES="${SAMPLES:-1000000}"      # N_s
NCORE="${NCORE:-128}"              # cotengra worker processes (= cores/node)
VENV="${VENV:-$WORK/rcs-venv}"     # big venv on $WORK (HOME quota too small)
WORKDIR="${WORKDIR:-$WORK/rcs}"

# ---- map array index -> graph stem ---------------------------------------
STEMS=(google_n53_m20 ustc_n56_m20 ustc_n60_m24 google_n70_m24 google_n67_m32)
STEM="${STEMS[$SLURM_ARRAY_TASK_ID]}"

cd "$WORKDIR"
mkdir -p logs results

echo "================================================================"
echo " Lonestar6 rcs_tnsa cost run"
echo "   array idx : $SLURM_ARRAY_TASK_ID -> $STEM"
echo "   node      : $(hostname)"
echo "   cores     : $NCORE   opt-time/mode : ${OPT_TIME}s"
echo "   K=$K  N_s=$SAMPLES"
echo "================================================================"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
unset PYTHONPATH

# keep numpy/BLAS from oversubscribing — cotengra parallelism is process-level
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
# HOME quota is tiny on ls6 — keep numba's JIT cache on $WORK
export NUMBA_CACHE_DIR="$WORK/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# run python directly (no srun) so loky workers see all $NCORE cores on the node
python -u run_rcs_tnsa_graphs.py \
    --graphs "$STEM" \
    --opt-time "$OPT_TIME" \
    --K "$K" \
    --samples "$SAMPLES" \
    --parallel "$NCORE" \
    --out "results/rcs_tnsa_costs_${STEM}.json"

echo "==> done $STEM -> results/rcs_tnsa_costs_${STEM}.json"
