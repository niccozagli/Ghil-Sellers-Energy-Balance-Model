#!/usr/bin/env bash
#SBATCH --job-name=plasim-zonal-t
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=cluster/plasim-zonal-t-%j.out
#SBATCH --error=cluster/plasim-zonal-t-%j.err

# Submit from the repository root with:
# sbatch cluster/extract_plasim_zonal_temperature.sbatch

set -euo pipefail

# Slurm may start the job in its spool directory, so use the directory from
# which sbatch was run as the repository root.
repository_root="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root.}"
cd "$repository_root"

# Change these two values for another PLASIM experiment.
co2_ppm=360
solar_irradiance_w_m2=1365

# This directory is an input only. The Python script opens its NetCDF files
# read-only and writes only to the output directory below.
input_dir="/data/complexp/fr120/PLASIM-1.0/plasim/run/output"
output_dir="$repository_root/data"
venv_python="$repository_root/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
    echo "No project virtual environment found. Run 'uv sync' on the login node first." >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "Input directory (read-only): $input_dir"
echo "Output directory: $output_dir"

"$venv_python" cluster/extract_plasim_zonal_temperature.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --co2-ppm "$co2_ppm" \
    --solar-irradiance-w-m2 "$solar_irradiance_w_m2"
