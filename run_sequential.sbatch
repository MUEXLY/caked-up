#!/bin/bash
#SBATCH --job-name=calib_sequential
#SBATCH --output=logs/sequential_%j.txt
#SBATCH --error=logs/sequential_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Create the logs directory before submitting this job.
source venv/bin/activate  # or conda activate your_env

python3 sequential_run.py sequential_config.json
