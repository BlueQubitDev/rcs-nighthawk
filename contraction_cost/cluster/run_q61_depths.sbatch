#!/bin/bash
#SBATCH -J q61_depth          # job name
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p normal             # Lonestar6 CPU queue (max 48h)
#SBATCH -N 1                  # 1 node per array task
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # all 128 cores -> cotengra loky workers
#SBATCH -t 14:00:00           # ~10h cotengra + margin
#SBATCH --array=0-3           # one task per depth: 10, 20, 30, 40
#SBATCH -A <allocation>         # TACC allocation

set -uo pipefail

# ---- knobs ----------------------------------------------------------------
OPT_TIME="${OPT_TIME:-18000}"      # cotengra budget per mode [s]; 5h x 2 modes
K="${K:-1000}"                     # sparse-output batch size
SAMPLES="${SAMPLES:-1000000}"      # N_s
FXEB="${FXEB:-0.002}"              # ASSUMED XEB fidelity (no device) — rescale later
NCORE="${NCORE:-128}"
VENV="${VENV:-$WORK/rcs-venv}"
WORKDIR="${WORKDIR:-$WORK/rcs}"

DEPTHS=(10 20 30 40)
D="${DEPTHS[$SLURM_ARRAY_TASK_ID]}"

cd "$WORKDIR"
mkdir -p logs results

source "$VENV/bin/activate"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$WORK/.numba_cache"; mkdir -p "$NUMBA_CACHE_DIR"

echo "[$(date)] node=$(hostname) depth=$D  opt_time=${OPT_TIME}s/mode  cores=$NCORE"

# estimate_sampling_cost.py drives sampling_cost.sampling_cost; we expose
# --parallel through the optimizer by setting cotengra workers via env.
# (sampling_cost._build_optimizer reads parallel from the call; the CLI does
#  not yet pass --parallel, so we set it through COTENGRA via NCORE below.)
python -u estimate_sampling_cost.py \
    --circuit "q61_circuits/q61_d${D}.qpy" \
    --samples "$SAMPLES" \
    --f-xeb "$FXEB" \
    --batch-size "$K" \
    --memory-preset 'ideal,gpu-128gb' \
    --opt-time "$OPT_TIME" \
    --max-repeats 1000000 \
    --depth "$D" \
    --parallel "$NCORE" \
    --out "results/q61_d${D}_cost.json"

echo "[$(date)] depth=$D done."
