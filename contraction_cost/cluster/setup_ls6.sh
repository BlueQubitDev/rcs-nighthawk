#!/bin/bash
# One-time setup on TACC Lonestar6 (ls6).
#
#   ssh <user>@ls6.tacc.utexas.edu
#   cd $SCRATCH/rcs            # where you rsync'd the rcs/ folder
#   bash cluster/setup_ls6.sh
#
# Creates a venv in $SCRATCH (HOME quota on TACC is tiny), installs the
# tensor-network stack, and downloads the 5 Google rcs_tnsa .graph files so
# the cluster never needs your laptop.
#
# Run this on a LOGIN node (it only pip-installs + curls; no heavy compute).
set -euo pipefail

# --- locations -------------------------------------------------------------
: "${WORK:?WORK not set — are you on a TACC node?}"
VENV="${VENV:-$WORK/rcs-venv}"     # big venv on $WORK (HOME quota too small)
DATA_DIR="${DATA_DIR:-$(pwd)/rcs_tnsa_data}"

echo "==> Loading python module"
module load python3 2>/dev/null || module load python 2>/dev/null || true
python3 --version

echo "==> Creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

echo "==> Installing tensor-network stack"
# kahypar + optuna are the two that materially improve path quality.
pip install \
    "quimb>=1.11" \
    "cotengra>=0.7" \
    "kahypar>=1.3" \
    "optuna>=3" \
    "joblib>=1.3" \
    "qiskit>=1.0" \
    "qiskit-quimb" \
    numpy
# joblib supplies loky, cotengra's parallel backend (needed for --parallel).

echo "==> Verifying imports"
python - <<'PY'
import quimb, cotengra, kahypar, optuna, joblib
from joblib.externals.loky import get_reusable_executor  # loky backend check
print("quimb", quimb.__version__, "| cotengra", cotengra.__version__,
      "| kahypar OK | optuna", optuna.__version__, "| joblib", joblib.__version__,
      "| loky OK")
PY

echo "==> Downloading Google rcs_tnsa .graph files into $DATA_DIR"
mkdir -p "$DATA_DIR"
BASE="https://raw.githubusercontent.com/google-research/google-research/master/rcs_tnsa/data"
for stem in google_n53_m20 google_n67_m32 google_n70_m24 ustc_n56_m20 ustc_n60_m24; do
    for ext in graph groups; do
        if [ ! -f "$DATA_DIR/$stem.$ext" ]; then
            curl -fsSL "$BASE/$stem.$ext" -o "$DATA_DIR/$stem.$ext"
            echo "    fetched $stem.$ext"
        fi
    done
done

echo
echo "==> Setup complete."
echo "    venv:     $VENV"
echo "    data:     $DATA_DIR"
echo "    next:     sbatch cluster/run_graphs.sbatch"
