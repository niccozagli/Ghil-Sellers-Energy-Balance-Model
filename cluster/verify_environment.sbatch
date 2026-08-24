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

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

# uv's installer places its executable here.  Batch shells may not source
# ~/.bashrc, so make the location explicit.
export PATH="$HOME/.local/bin:$PATH"
# Cluster cache and project files can be on separate filesystems.
export UV_LINK_MODE=copy
export PYTHONPATH=src

echo "Host: $(hostname)"
echo "Working directory: $PWD"
uv --version
uv run python --version
uv run python -c "import gsebm; print('gsebm import works')"
uv run python -m unittest discover -s tests
