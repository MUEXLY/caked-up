#!/bin/bash
#SBATCH --job-name=calib_sequential
#SBATCH --output=logs/sequential_%A_%a.txt
#SBATCH --error=logs/sequential_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Create the logs directory before submitting this job.
source venv/bin/activate  # or conda activate your_env

python3 sequential_run.py sequential_config.json --prior-index "$SLURM_ARRAY_TASK_ID"
