# Cross-validation grid search

The grid-search workflow generates Slurm array configs for Study 1 and Study 2,
runs one calibration per array task, and ranks each shrinkage-method combination
by its cross-validation NRMSE.

## Configure and generate

Edit `grid_search_config.json`, then run from the repository root:

```bash
python sweep.py
```

The generator reads the number of calibrated input fields from each study's
`modelData/thetaVals.txt`. It takes the Cartesian product of the listed
per-field kappa prior candidates and `delta_eta_gp_grid`. The current candidate
list covers GP, lasso, fused lasso, and spike-slab; horseshoe is omitted.

All generated configs receive the same physical-constraints block. By default,
that block is copied from `config.json`. To override it for the entire sweep,
replace `shared_physical_constraints` in `grid_search_config.json` with one
complete physical-constraints object. Do not set study-specific physical
constraints when comparing candidates.

Study 1 uses the cross-validation truth from the existing project config
(constant theta `[1.5, 1.75]` and cosine additive discrepancy). Study 2 follows
the existing Study 2 config convention (theta fields `[sin(x), cos(x)]`, with
no known additive discrepancy). These truth settings are explicit in the sweep
config and should be changed there if the study definitions change.

The grid file includes two candidate hyperprior distributions per kappa family
and two delta_eta GP hyperprior distributions. Increase or reduce the arrays in
that file to control the grid size. For the current two-parameter studies this
creates 128 configs per study (256 total). The generated files are written to
`configs/grid_search/`; run outputs go under `results/cv_grid_search/`.

## Submit on Slurm

After generation, submit the generated array script:

```bash
sbatch configs/grid_search/run_array.generated.sbatch
```

The array range and concurrency limit are generated from the manifest. The
template `run_array.sbatch` activates the repository `venv` when present. Add
cluster-specific module or environment setup to that template before
generating the array script. The script requests 4 hours, one CPU, and 16 GB;
adjust the Slurm resource directives for the target cluster and calibration
size.

## Rank the results

After the array completes:

```bash
python post_processing/post_process_sweep.py
```

The default ranking metric is `net_nrmse` (the sum of y, theta, and—when
configured—delta NRMSE). Ranking is ascending within each study and each
per-field shrinkage-method combination. The script writes:

- `results/cv_grid_search/postprocessed/ranked_runs.csv`
- `results/cv_grid_search/postprocessed/best_by_method.csv`
- one chart of each study's best NRMSE per method combination

Use `--metric y_nrmse` if ranking should reflect only observable prediction
error. The component NRMSEs and shared-constraint fingerprint are retained in
the CSV for interpretation and fairness checks.
