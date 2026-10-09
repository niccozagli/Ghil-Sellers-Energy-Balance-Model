#!/bin/bash
# One array task = one μ experiment. Submit with submit_plasim_extraction.sh,
# which supplies REPO, LIST, OUTPUT_ROOT and WORKERS, the array range, the
# CPU count and the log paths.
#SBATCH --job-name=plasim_extract
#SBATCH --time=03:30:00
#SBATCH --partition=short
#SBATCH --account=complexp
#SBATCH --qos=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G

set -euo pipefail

: "${REPO:?submit with REPO}"
: "${LIST:?submit with LIST}"
: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?run as an array task}"
WORKERS=${WORKERS:-4}

EXPERIMENT=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$LIST")
[[ -n "$EXPERIMENT" && -d "$EXPERIMENT" ]] || { echo "ERROR: no experiment for task $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2; }
NAME=$(basename "$EXPERIMENT")

# HDF5 file locking is unreliable on Lustre scratch.
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "$(date -Is) task $SLURM_ARRAY_TASK_ID: $NAME on $(hostname)"
"$REPO/.venv/bin/python" "$REPO/scripts/extract_plasim_raw_maps.py" \
    --experiment-dir "$EXPERIMENT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --report-path "$OUTPUT_ROOT/logs/report_${NAME}.json"
echo "$(date -Is) done: $NAME"
