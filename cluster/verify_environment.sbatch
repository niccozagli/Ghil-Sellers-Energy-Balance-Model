#!/usr/bin/env bash
#SBATCH --job-name=gsebm-verify
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=cluster/gsebm-verify-%j.out
#SBATCH --error=cluster/gsebm-verify-%j.err

# Verify that the uv-managed GSEBM environment works on a Slurm compute node.
# Submit from the repository root with: sbatch cluster/verify_environment.sbatch

set -euo pipefail

# Slurm records the directory from which sbatch was invoked.  Do not derive
# this from BASH_SOURCE: Slurm may execute relative script paths from its
# spool directory instead of the submission directory.
repository_root="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root.}"
cd "$repository_root"

# uv's installer places its executable here.  Batch shells may not source
# ~/.bashrc, so make the location explicit.
export PATH="$HOME/.local/bin:$PATH"
# Cluster cache and project files can be on separate filesystems.
export UV_LINK_MODE=copy
venv_python="$repository_root/.venv/bin/python"

echo "Host: $(hostname)"
echo "Working directory: $PWD"
uv --version
echo "Virtual-environment interpreter: $venv_python"

if [[ ! -x "$venv_python" ]]; then
    echo "No project virtual environment found. Run 'uv sync' on the login node first." >&2
    exit 1
fi

"$venv_python" --version
"$venv_python" -c "import gsebm; print('gsebm import works')"
"$venv_python" -m unittest discover -s tests
