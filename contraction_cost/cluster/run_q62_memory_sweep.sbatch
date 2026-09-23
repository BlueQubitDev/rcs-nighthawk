#!/bin/bash
#SBATCH -J q62_memsweep
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -A <allocation>         # TACC allocation
#----------------------------------------------------------------
# Eq.(3)-side memory sensitivity for q62 d=32: one long path search,
# then slicing to every memory preset (ideal, gpu-80gb, node-4tb,
# allram-32tb, secondary-1pb). Complements the rcs_tnsa mode-3 runs:
# together they give cost-vs-memory for both the single-amplitude and
# the batched task on our circuit.
#----------------------------------------------------------------
set -uo pipefail
source "$WORK/rcs-venv/bin/activate"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$WORK/.numba_cache"; mkdir -p "$NUMBA_CACHE_DIR"
cd "$WORK/rcs"; mkdir -p logs results

python -u run_memory_sweep.py \
    --circuit "q62_circuits/q62_d32.qpy" \
    --depth 32 \
    --K 1000 --samples 1000000 --f-xeb 2.6e-4 \
    --opt-time 18000 --parallel 128 \
    --out results/q62_d32_memsweep.json
echo "[$(date)] memory sweep done."
