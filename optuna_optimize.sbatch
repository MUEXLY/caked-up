#!/bin/bash
#SBATCH --job-name=calib_fuid_optuna_optimize
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# activate environment
source venv/bin/activate

echo "Running optuna optimizer using optuna_hyperparameters/config_optuna.json"

python3 optuna_hyperparameters/optuna_hyperparam.py