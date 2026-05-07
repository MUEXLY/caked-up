import os
import json
import itertools

# ---- output directory ----
config_dir = "configs"
os.makedirs(config_dir, exist_ok=True)

# ---- parameter grid ----
param_grid = {
    "kappa_var": [0.01, 0.05, 0.1],
    "kappa_ell": [0.2, 0.5],
    "eta_var": [0.005, 0.01],
    "eta_ell": [0.5, 1.0],
}

# ---- base config ----
with open("config.json", "r") as f:
    base_config = json.load(f)

# ---- generate combinations ----
keys = list(param_grid.keys())
values = list(param_grid.values())

configs = list(itertools.product(*values))

print(f"Generating {len(configs)} configs...")

for i, combo in enumerate(configs):
    config = base_config.copy()

    param_dict = dict(zip(keys, combo))

    # ---- modify config ----
    config["calibration_settings"]["kappa_prior_vals"]["var"] = param_dict["kappa_var"]
    config["calibration_settings"]["kappa_prior_vals"]["ell"] = param_dict["kappa_ell"]

    config["calibration_settings"]["delta_eta_prior_vals"]["var"] = param_dict["eta_var"]
    config["calibration_settings"]["delta_eta_prior_vals"]["ell"] = param_dict["eta_ell"]

    # ---- create readable run name ----
    run_name = (
        f"kvar{param_dict['kappa_var']}_"
        f"kell{param_dict['kappa_ell']}_"
        f"evar{param_dict['eta_var']}_"
        f"eell{param_dict['eta_ell']}"
    )

    results_path = f"results/sweep/{run_name}"

    # ---- inject into config ----
    config["output_settings"]["results_path"] = results_path
    config["output_settings"]["figures_path"] = f"{results_path}/figures"

    # ---- save ----
    fname = f"{config_dir}/config_{i:04d}.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=2)

print("Done.")