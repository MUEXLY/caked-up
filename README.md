# CAKED-UP


[![arXiv](https://img.shields.io/badge/arXiv-2605.30492-b31b1b.svg)](https://arxiv.org/abs/2605.30492)
[![Citation](https://img.shields.io/badge/Citation-CFF-brightgreen.svg)](./CITATION.cff)



**Calibration Addressing Kappa Embedded Discrepancy with Uncertainty Propagation (CAKED-UP)** is a Bayesian calibration framework for inexact computer models. The method combines Gaussian Process (GP) emulation, Bayesian inference, and orthogonal discrepancy projections to quantify:

- Calibration parameter uncertainty
- Model-form inadequacy
- Embedded input discrepancies
- Predictive uncertainty propagation

CAKED-UP is designed to streamline calibration workflows for simulation-based scientific and engineering models while supporting scalable workflows on both local machines and SLURM-based HPC systems.

---

# Features

- Bayesian calibration of computational models
- Gaussian Process emulators for simulations and discrepancies
- Orthogonalization between parameter discrepancy (`\kappa`) and model discrepancy (`\delta`)
- Cross-validation utilities
- Hyperparameter grid search with SLURM arrays
- Optuna-based hyperparameter optimization
- Automated post-processing and sensitivity analysis
- HPC-ready execution scripts

---

# Repository Structure

```text
caked-up/
├── main.py
├── config.json
├── requirements.txt
├── sweep.py
├── post_process_sweep.py
├── submit_single.sbatch
├── run_array.sbatch
├── modelData/
├── observationData/
├── sensitivity_analysis/
├── optuna_hyperparameters/
└── ...
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd caked-up
```

---

## 2. Create a Python Virtual Environment

It is recommended to use a dedicated virtual environment.

```bash
python3 -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# HPC Setup (SLURM)

If running on a SLURM-based cluster:

## Allocate an Interactive Node

```bash
salloc ...
```

## Load Anaconda

```bash
module load anaconda3
```

Then proceed with the virtual environment setup described above.

---

# Configuration File

The `config.json` file provides the primary interface for configuring a calibration study.

It includes settings for:

## Calibration Settings

- Number of MCMC samples (`N_mcmc`)
- GP hyperparameter priors
- Orthogonalization settings
- Noise assumptions
- Sampling controls
- Embedded discrepancy configuration

## Input Settings

- Simulation/model data paths
- Observation/experimental data paths

## Cross-Validation Settings

- Known `\kappa` (input discrepancy) forms
- Known `\delta` (model discrepancy) forms
- Validation controls

## Output Settings

- Diagnostic plots
- Saved posterior data
- Trained emulator models
- Post-processing outputs

---

# Input Data Format

CAKED-UP expects column-wise text files with one sample per row.

Example directory structure:

```text
modelData/
├── appDomain.txt
├── modelPredictions.txt
└── thetaVals.txt

observationData/
├── appDomain.txt
└── observationData.txt
```

## Model Data

| File | Description |
|---|---|
| `appDomain.txt` | Physical input parameters |
| `modelPredictions.txt` | Simulation/model outputs |
| `thetaVals.txt` | Calibration parameter values |

## Observation Data

| File | Description |
|---|---|
| `appDomain.txt` | Experimental physical inputs |
| `observationData.txt` | Experimental measurements |

Example datasets are included in the repository.

---

# Running a Calibration

After configuring `config.json` and preparing the input datasets:

```bash
python3 main.py config.json
```

---

# Running on SLURM

To submit a single calibration case:

```bash
sbatch submit_single.sbatch
```

---

# Hyperparameter Grid Search

CAKED-UP includes a grid search workflow for studying the effects of GP shrinkage priors and discrepancy hyperparameters.

## Step 1 — Define the Parameter Grid

Modify `sweep.py`:

```python
param_grid = {
    "kappa_var": [0.01, 0.05, 0.1],
    "kappa_ell": [0.2, 0.5],
    "eta_var": [0.005, 0.01],
    "eta_ell": [0.5, 1.0],
}
```

## Step 2 — Generate Configuration Files

```bash
python3 sweep.py
```

This creates a `configs/` directory containing all generated calibration cases.

## Step 3 — Submit the SLURM Array

```bash
sbatch run_array.sbatch
```

## Step 4 — Post-Process Results

```bash
python3 post_process_sweep.py
```

## Step 5 — Run Sensitivity Analysis

```bash
python3 sensitivity_analysis/sensitivity_analysis.py
```

---

# Optuna Hyperparameter Optimization

For automated hyperparameter tuning, CAKED-UP supports optimization using the Optuna framework.

## Configure the Optimization

Edit:

```text
optuna_hyperparameters/config_optuna.json
```

## Run the Optimization

```bash
python3 optuna_hyperparameters/optuna_hyperparam.py
```

This workflow calls the main calibration routine internally and assumes cross-validation is enabled in the global `config.json`.

---

# Methodology Overview

The CAKED-UP framework:

1. Constructs GP emulators for simulation outputs
2. Represents embedded parameter discrepancies using `\kappa`
3. Represents model-form discrepancies using `\delta`
4. Orthogonalizes discrepancy spaces to reduce identifiability issues
5. Performs Bayesian inference using MCMC sampling
6. Propagates uncertainty through posterior predictive distributions

The orthogonalization procedure includes:
- Projection between discrepancy subspaces
- Jacobian evaluation across the application domain
- PCA-based dimensionality reduction and decorrelation

---

# Output

Typical outputs include:

- Posterior parameter distributions
- GP emulator diagnostics
- Predictive uncertainty bands
- Orthogonal basis diagnostics
- Cross-validation statistics
- Sensitivity analysis figures
- Hyperparameter sweep summaries

---

# Recommended Workflow

1. Prepare simulation and observation datasets
2. Configure `config.json`
3. Run a baseline calibration
4. Validate with cross-validation studies
5. Perform hyperparameter sweeps or Optuna optimization
6. Analyze posterior predictions and discrepancy structure

---

# Future Development

Planned improvements include:

- Expanded Optuna automation utilities
- Improved visualization tools
- Additional discrepancy kernels
- Multi-fidelity model support
- Enhanced HPC scalability
- Improved documentation and tutorials

---
# Citation

If you use CAKED-UP in your research, please cite the associated methodology paper:

```bibtex
@article{myhill2026cakedup,
  title={Shrinkage-Constrained Functional Calibration for Complex Computer Models},
  author={Myhill, Liam and Martinez, Enrique and Russcher, Sez},
  journal={arXiv preprint arXiv:2605.30492},
  year={2026},
  url={https://arxiv.org/abs/2605.30492}
}
```

A machine-readable citation is also provided via the repository's `CITATION.cff` file. GitHub will automatically expose a **"Cite this repository"** button in the repository sidebar when the file is present.

## Related Publication

**Liam Myhill, Enrique Martinez, and Sez Russcher.**

*Shrinkage-Constrained Functional Calibration for Complex Computer Models.*

arXiv:2605.30492 (2026)

https://arxiv.org/abs/2605.30492

---

# License

Add your preferred license information here (MIT, BSD, GPL, etc.).

---

# Contact

For questions, issues, or contributions, please open an issue or pull request on the repository.
