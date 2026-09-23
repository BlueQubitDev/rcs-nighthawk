#!/bin/bash
#SBATCH -J zcz3_est
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00           # 83q x 32c is the hardest; give it the max
#SBATCH -A <allocation>         # TACC allocation
#----------------------------------------------------------------
# Zuchongzhi-3.0-like RCS (83 qubits x 32 cycles, iSWAP, ABCDCDBA).
# RECONSTRUCTION — geometry-approximate; this is an upper-bound estimate,
# not a faithful reproduction (paper ships no circuit).
# Ideal memory (unlimited-memory floor, comparable to the paper's 762.2PB row).
#----------------------------------------------------------------
set -uo pipefail
source "$WORK/rcs-venv/bin/activate"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$WORK/.numba_cache"; mkdir -p "$NUMBA_CACHE_DIR"
cd "$WORK/rcs"; mkdir -p logs results zcz3_circuits

# build the reconstructed circuit on the node (deterministic seed)
python -u - <<'PYBUILD'
from build_zcz3_rcs import zcz3_rcs
import qiskit.qpy as qpy
qc = zcz3_rcs(d=32, seed=2025)
with open("zcz3_circuits/zcz3_83q_32c.qpy", "wb") as f:
    qpy.dump(qc, f)
print("built zcz3_83q_32c.qpy:", qc.num_qubits, "qubits,", len(qc.data), "gates")
PYBUILD

# F=2.5e-4 to match the paper's reported task fidelity
echo "[$(date)] node=$(hostname) ZCZ-3.0 reconstruction 83q x 32c"
python -u estimate_sampling_cost.py \
    --circuit "zcz3_circuits/zcz3_83q_32c.qpy" \
    --f-xeb 2.5e-4 --samples 1000000 --batch-size 1000 \
    --memory-preset ideal \
    --opt-time 32400 --parallel 128 \
    --depth 32 \
    --out "results/zcz3_83q_32c_cost.json"
echo "[$(date)] ZCZ-3.0 done."
