# integratedDelta_calibration

Scripts for calibrating simulation data to experimental observations using an embedded discrepancy Gaussian Process (GP) emulator.

---

## Overview

This repository implements a Bayesian calibration framework that:

- Builds a GP emulator of simulator outputs  
- Embeds a model discrepancy term  
- Calibrates simulation parameters to observational data using MCMC  

The workflow is driven by `main.py` and configured via `config.json`.

---

## Environment Setup

This project uses a Python virtual environment.

### 1. Create a virtual environment

    python3 -m venv venv

### 2. Activate the environment

**macOS/Linux**
    
    source venv/bin/activate

**Windows**
    
    venv\Scripts\activate

### 3. Install required packages

    pip install -r requirements.txt

Core dependencies include:

- numpy  
- scipy  
- matplotlib  
- pandas  
- scikit-learn  

---

## Running the Calibration

To execute the integrated delta calibration routine:

    python3 main.py

All runtime behavior is controlled through `config.json`.

---

## Repository Structure

    integratedDelta_calibration/
    │
    ├── main.py
    ├── config.json
    ├── requirements.txt
    │
    ├── modelData/
    │   ├── appDomain.txt
    │   ├── modelPredictions.txt
    │   └── thetaVals.txt
    │
    └── observationData/
        ├── appDomain.txt
        └── observationData.txt

---

## Input Data Format

Input data is organized into two subdirectories:

### `/modelData`
Simulator outputs to be emulated and calibrated.

- `appDomain.txt` — Application domain locations  
- `modelPredictions.txt` — Simulator outputs  
- `thetaVals.txt` — Parameter values corresponding to simulations  

### `/observationData`
Experimental data to be matched.

- `appDomain.txt` — Observation domain locations  
- `observationData.txt` — Measured responses  

### Formatting Requirements

All `.txt` files must:

- Be column-formatted  
- Include a header row labeling each column  
- Use the delimiter specified in `config.json`  

The provided template files illustrate formatting for MD (observational) and DDD (simulator) predictions of  
$\tau^{CRSS}$.

---

## Configuration

All calibration settings are controlled via `config.json`.

### `calibration_settings`
Controls MCMC behavior:

- `N_mcmc` — Total MCMC iterations  
- `N_burn` — Burn-in samples  
- `N_samp_post` — Posterior samples retained  

### `input_settings`

- Input file delimiter  

### `output_settings`

- Controls which plots and diagnostics are generated  
- `figure_options.prior_posterior_density` adds a prior/posterior density comparison
  for each calibration parameter. The legend distinguishes the prior
  `theta` from the posterior `theta* = theta^0 + delta_theta`.
- `delta_posterior_physical` plots physical delta_theta relative to the fixed
  `theta^0` baseline. Its shaded prior band is transformed to
  `theta - theta^0`, so the range is centered in the same zero-based
  coordinate system as delta_theta.
- `figure_options.emulator_sensitivity_pca` adds
  `figures/emulator_sensitivity_pca.png`, a PCA biplot of local emulator
  sensitivities to each calibration parameter. Theta labels are read in order
  from `label_settings.theta_labels`.
- `figure_options.emulator_sensitivity_boxplot` adds
  `figures/emulator_sensitivity_boxplot.png`, a ranked box-and-whisker plot of
  absolute local sensitivities. Parameters are ordered by median influence;
  white diamonds mark the mean sensitivity.
- `figure_options.holdout_sensitivity` evaluates normalized RMSE on repeated random
  holdout subsets. Results are written to
  `results/holdout_sensitivity_metrics.json`; fractions and repetitions are
  configured under `output_settings.validation_options`.
- `figure_options.kappa_hyperprior_sensitivity` runs repeated paired
  calibrations after scaling one delta_theta GP hyperprior median at a time. Configure
  per-delta_theta log-normal hyperpriors under
  `validation_options.kappa_hyperpriors`, using `rho` for correlation length
  and `sigma` for GP amplitude. The fields `log_mu` and `log_sigma` are
  defined on the natural logarithm of each quantity. For example:
  `"rho": {"log_mu": -1.0, "log_sigma": 1.0}` and
  `"sigma": {"log_mu": -1.0, "log_sigma": 1.0}`.
  The sampler converts the amplitude prior to the corresponding variance
  prior internally.
  The validation writes
  `results/kappa_hyperprior_sensitivity_metrics.json` and
  `figures/kappa_hyperprior_sensitivity.png`, which ranks posterior delta_theta
  changes for the rho and sigma perturbations. The validation is disabled by
  default and uses `kappa_hyperprior_factors`, `kappa_hyperprior_repeats`,
  `kappa_hyperprior_n_mcmc`, and `kappa_hyperprior_burn_in` for its repeated
  runs.
- Holdout selection protects all observations at the minimum or maximum
  application-domain values. `figure_options.holdout_predictions` adds a grid
  figure with one row per holdout trial and one column for the full-domain
  posterior predictive plus one column per calibration parameter. Held-out
  points are highlighted in every panel.
- `figure_options.holdout_nrmse_sensitivity` adds
  `figures/holdout_nrmse_sensitivity.png`, which aggregates every holdout trial
  at each withheld fraction and plots mean NRMSE with +/- 1 SD, alongside the
  full-dataset NRMSE baseline. When holdout cases are available, the figure
  also includes a posterior inset from one trial at the largest holdout
  fraction, with selected observations labeled `Holdout`.
- When `figure_options.holdout_sensitivity` is `true`, each holdout trial is
  refit using only retained observations and scored only on withheld interior
  observations. `validation_options.holdout_n_mcmc` and
  `holdout_burn_in` control the additional trial sampler cost. When it is
  `false`, calibration is performed once and no holdout refits are run.

### `label_settings`

- `observation_set_label` controls the label used for the observation data
  in generated figures.
- `simulation_set_label` controls the label used for simulator data and
  predictive-fit titles.
- LaTeX labels must use JSON-escaped backslashes, for example:
  `"theta1": "\\ell^{c}"` and `"kappa1": "\\delta_{\\theta}"`. Labels beginning
  with `\` are rendered as Matplotlib mathtext in parameter and discrepancy
  figure annotations.
- The external-validation figure uses `application_domain_labels` and
  `observation_labels` for the left panel axes, and renders configured theta
  labels in the parameter subplots.
- `figure_options.data_priors` writes `figures/rawData_theta_parameters.png`
  using a fixed three-column subplot layout. Its theta titles, colorbars,
  domain x-axis, and observation y-axis use the configured label settings.

### External validation

`output_settings.validation_options.external_validation` enables an opt-in
validation pass over a separate CSV. The CSV is read after calibration and is
never included in emulator training or MCMC. Configure its column mapping and
any unit conversion with `x_column`, `y_column`, `parameter_columns`, and
`y_scale`. The pass writes `figures/external_validation.png` and
`results/external_validation_predictions.csv`, including the learned
discrepancy correction evaluated at each external application-domain point.
The external-validation figure also includes the original simulation and
observation datasets; external validation points use the configured simulation
set label. Each calibration parameter is shown in its own panel and y-axis.
The parameter panels use the calibration input `theta_*` values as prior
points. Corrected external-validation points use consistent marker shapes for
each unique external application-domain location in both the prediction and
parameter panels, with a compact shared legend above the panels using the
configured application-domain label.

### `results_path` & `results_options`

*(In development)*  
Controls output of serialized model objects for downstream analysis.

---

## Notes

- The virtual environment directory (`venv/`) should not be committed to the repository.
- Only `requirements.txt` is required to reproduce the environment.

## Citation

If you use this repository or the associated methodology in your research, please cite the following paper.

**Bayesian Model Calibration with Integrated Discrepancy: Addressing Inexact Dislocation Dynamics Models**
Liam Myhill, Enrique Martinez Saez, Sez Russcher (2026)

[![arXiv](https://img.shields.io/badge/arXiv-2603.11960-b31b1b.svg)](https://arxiv.org/abs/2603.11960)
[![DOI](https://img.shields.io/badge/DOI-coming%20soon-blue)]()

Paper: https://arxiv.org/abs/2603.11960

```bibtex
@misc{myhill2026bayesianmodelcalibrationintegrated,
  title   = {Bayesian Model Calibration with Integrated Discrepancy: Addressing Inexact Dislocation Dynamics Models},
  author  = {Myhill, Liam and Martinez Saez, Enrique and Russcher, Sez},
  year    = {2026},
  eprint  = {2603.11960},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url     = {https://arxiv.org/abs/2603.11960}
}
```

---

### Repository Citation

This repository includes a `CITATION.cff` file so that GitHub can generate a **"Cite this repository"** button automatically.
