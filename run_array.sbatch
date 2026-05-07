#!/bin/bash
#SBATCH --job-name=calib_sweep
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-35
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# activate environment
source venv/bin/activate  # or
# conda activate your_env  # or venv activation

# config index from Slurm
IDX=$SLURM_ARRAY_TASK_ID

CONFIG=configs/config_$(printf "%04d" $IDX).json

echo "Running config: $CONFIG"

python3 main.py $CONFIG