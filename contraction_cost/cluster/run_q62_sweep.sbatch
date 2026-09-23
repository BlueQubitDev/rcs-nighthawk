#!/bin/bash
#SBATCH -J q62_sweep
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 14:00:00
#SBATCH --array=0-9          # one task per depth (4,8,...,40)
#SBATCH -A <allocation>         # TACC allocation
#----------------------------------------------------------------
# q=62 depth sweep, IDEAL memory only (one memory model => the loop finishes
# and JSON is written; avoids the 2-model walltime bug from the q61 run).
# Budget + parallelism scale with depth (trivial depths run serial+short to
# dodge the loky deadlock and save SUs; deep depths get the full search).
#----------------------------------------------------------------
set -uo pipefail
source "$WORK/rcs-venv/bin/activate"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$WORK/.numba_cache"; mkdir -p "$NUMBA_CACHE_DIR"
cd "$WORK/rcs"; mkdir -p logs results

DEPTHS=(4 8 12 16 20 24 28 32 36 40)
D="${DEPTHS[$SLURM_ARRAY_TASK_ID]}"

# depth-scaled search budget + parallelism
if   [ "$D" -le 12 ]; then OPT=300;   PAR=False   # trivial: short, serial (loky-safe)
elif [ "$D" -le 24 ]; then OPT=3600;  PAR=128     # moderate: 1h/mode
else                       OPT=18000; PAR=128      # deep: 5h/mode
fi

echo "[$(date)] node=$(hostname) q=62 depth=$D opt=${OPT}s parallel=$PAR"
python -u estimate_sampling_cost.py \
    --circuit "q62_circuits/q62_d${D}.qpy" \
    --f-xeb 2e-3 --samples 1000000 --batch-size 1000 \
    --memory-preset ideal \
    --opt-time "$OPT" --parallel "$PAR" \
    --depth "$D" \
    --out "results/q62_d${D}_cost.json"
echo "[$(date)] q=62 depth=$D done."
