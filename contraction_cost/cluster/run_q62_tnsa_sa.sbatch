#!/bin/bash
#SBATCH -J q62_tnsa_sa
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 10:00:00
#SBATCH --array=0-5
#SBATCH -A <allocation>         # TACC allocation
#----------------------------------------------------------------
# K-aware batched-contraction search of our q62 d=32 network using
# Google's rcs_tnsa simulated-annealing optimizer (the code behind the
# sampling column of Morvan et al. Table 1). One array task per config;
# each node runs 128 independent single-threaded walkers (their
# recommended usage), 4e6 SA steps each (~7 h at the measured
# ~10 min / 1e5 steps on this graph). Take the best over walkers.
#
# Configs:
#   0: mode 1, closed graph            -> single-amplitude anchor (vs 10^22.01)
#   1: mode 2, open, width 33          -> per-slice tensors ~ 80 GB GPU class
#   2: mode 2, open, width 42          -> per-slice tensors ~ 32 TB RAM class
#   3: mode 2, open, width 74          -> effectively unlimited width (batched floor)
#   4: mode 3, open, log2 footprint 42 -> total memory 32 TB (all-RAM class)
#   5: mode 3, open, log2 footprint 49 -> total memory ~4.5 PB (secondary class)
# Sparse configurations per contraction: 1000 (their examples; our K=1000).
#----------------------------------------------------------------
set -uo pipefail
export OMP_NUM_THREADS=1
cd "$WORK/rcs"; mkdir -p logs results/tnsa

SRC="$WORK/rcs/external/google-research/rcs_tnsa/src"
[ -x "$SRC/optimize" ] || (cd "$SRC" && make optimize -j)

G="data/q62_tnsa/q62_m32.graph"
GO="data/q62_tnsa/q62_m32_open.graph"
GR="data/q62_tnsa/q62_m32_open.groups"
STEPS=4000000; T0=1e5; T1=1e-5; NSPARSE=1000

seed_of () { echo $(( (SLURM_ARRAY_TASK_ID + 1) * 1000 + $1 )); }

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=m1_closed
     for w in $(seq 1 128); do s=$(seed_of "$w")
       "$SRC/optimize" 1 "$G" "results/tnsa/${TAG}_s${s}.out" "$s" $STEPS $T0 $T1 &
     done; wait ;;
  1|2|3) case "$SLURM_ARRAY_TASK_ID" in 1) W=33;; 2) W=42;; 3) W=74;; esac
     TAG="m2_w${W}"
     for w in $(seq 1 128); do s=$(seed_of "$w")
       "$SRC/optimize" 2 "$GO" "$GR" "results/tnsa/${TAG}_s${s}.out" "$s" "$W" $NSPARSE $STEPS $T0 $T1 &
     done; wait ;;
  4|5) case "$SLURM_ARRAY_TASK_ID" in 4) M=42;; 5) M=49;; esac
     TAG="m3_f${M}"
     for w in $(seq 1 128); do s=$(seed_of "$w")
       "$SRC/optimize" 3 "$GO" "$GR" "results/tnsa/${TAG}_s${s}.out" "$s" "$M" $NSPARSE $STEPS $T0 $T1 &
     done; wait ;;
esac
echo "[$(date)] config $SLURM_ARRAY_TASK_ID done."
