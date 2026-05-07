#!/bin/bash
#SBATCH --job-name=calib_fuid_single
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# activate environment
source venv/bin/activate

CONFIG=config.json

echo "Running config: $CONFIG"

python3 main.py $CONFIG