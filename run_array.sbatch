#!/bin/bash
# Template: run `python sweep.py` to generate a runnable array script with the
# correct array range, manifest path, project path, and log directory.
#SBATCH --job-name=cv_grid
#SBATCH --array=__ARRAY_RANGE__
#SBATCH --output=__LOG_DIR__/%x_%A_%a.out
#SBATCH --error=__LOG_DIR__/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="$PROJECT_ROOT/__MANIFEST_PATH__"
cd "$PROJECT_ROOT"

# Activate the repository environment when present. On clusters that use
# modules or conda, add the site-specific setup below this comment.
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

python run_sweep_task.py "$MANIFEST" "$SLURM_ARRAY_TASK_ID"
