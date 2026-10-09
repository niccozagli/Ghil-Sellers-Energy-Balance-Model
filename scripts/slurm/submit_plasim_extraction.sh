#!/bin/bash
# Submit one raw-map extraction task per μ experiment as a Slurm array.
#
# Usage (from a login node, inside the repository clone):
#   scripts/slurm/submit_plasim_extraction.sh                 # every CONTROL_*_MU_* run
#   MU_MIN=1230 MU_MAX=1245 scripts/slurm/submit_plasim_extraction.sh
#   ONLY="1232p5 1233p75" scripts/slurm/submit_plasim_extraction.sh
#   DRY_RUN=1 scripts/slurm/submit_plasim_extraction.sh       # list the tasks, submit nothing
#
# Each task builds, resumes, or appends one archive, so the script can be re-run
# after new model years appear or after a task hits its time limit.

set -euo pipefail
shopt -s nullglob

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd -P)}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-/home/n/nz68/plasim-workspace/experiments}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/complexp/nz68/plasim-workspace/extracted}
WORKERS=${WORKERS:-4}
MAX_PARALLEL=${MAX_PARALLEL:-4}
MU_MIN=${MU_MIN:-}
MU_MAX=${MU_MAX:-}
ONLY=${ONLY:-}
DRY_RUN=${DRY_RUN:-0}

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$REPO/.venv/bin/python" ]] || die "No environment at $REPO/.venv; run 'uv sync' in $REPO on a login node first"
[[ -d "$EXPERIMENTS_ROOT" ]] || die "Experiments root not found: $EXPERIMENTS_ROOT"
# Resolve the root (often a symlink to scratch) but keep each experiment's own
# name: the archive is named after the experiment directory.
EXPERIMENTS_ROOT=$(cd "$EXPERIMENTS_ROOT" && pwd -P)

mu_of() {  # CONTROL_..._MU_1232p5 -> 1232.5
    local name=${1##*_MU_}
    echo "${name/p/.}"
}

in_range() {
    local mu=$1
    [[ -z "$MU_MIN" ]] || awk -v a="$mu" -v b="$MU_MIN" 'BEGIN{exit !(a >= b)}' || return 1
    [[ -z "$MU_MAX" ]] || awk -v a="$mu" -v b="$MU_MAX" 'BEGIN{exit !(a <= b)}' || return 1
}

selected=()
for dir in "$EXPERIMENTS_ROOT"/CONTROL_*_MU_*; do
    [[ -d "$dir/output/spinup" ]] || { echo "skip (no output/spinup): $dir" >&2; continue; }
    label=${dir##*_MU_}
    if [[ -n "$ONLY" ]]; then
        [[ " $ONLY " == *" $label "* ]] || continue
    else
        in_range "$(mu_of "$dir")" || continue
    fi
    selected+=( "$dir" )
done
(( ${#selected[@]} > 0 )) || die "No matching experiments under $EXPERIMENTS_ROOT"

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
LIST="$LOG_DIR/experiments_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${selected[@]}" > "$LIST"
echo "Experiments (${#selected[@]}), listed in $LIST:"
cat "$LIST"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY_RUN=1: nothing submitted."
    exit 0
fi

sbatch \
    --array="0-$(( ${#selected[@]} - 1 ))%${MAX_PARALLEL}" \
    --cpus-per-task="$(( WORKERS + 1 ))" \
    --output="$LOG_DIR/slurm-%x-%A_%a.out" \
    --error="$LOG_DIR/slurm-%x-%A_%a.err" \
    --export="ALL,REPO=$REPO,LIST=$LIST,OUTPUT_ROOT=$OUTPUT_ROOT,WORKERS=$WORKERS" \
    "$REPO/scripts/slurm/extract_plasim_raw_maps.slurm"
