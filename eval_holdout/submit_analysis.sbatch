#!/bin/bash
#SBATCH --job-name=holdout-analysis
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G

# activate environment
source ../venv/bin/activate

python3 evaluate_uncertainty.py