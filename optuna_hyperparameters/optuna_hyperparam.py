import optuna
import numpy as np
import json
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt
import copy


# ============================================================
# Diagnostics
# ============================================================
def save_optuna_diagnostics(study, output_dir):

    trials = [t for t in study.trials if t.value is not None]
    losses = [t.value for t in trials]

    if len(losses) == 0:
        return

    idx = np.arange(len(losses))

    plt.figure()
    plt.plot(idx, losses, marker='o', linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.title("Optuna Optimization Progress")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "optuna_loss_trajectory.png")

    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Config utilities
# ============================================================
def load_configs():
    with open("config.json") as f:
        main_config = json.load(f)

    with open("optuna_hyperparameters/config_optuna.json") as f:
        optuna_config = json.load(f)

    return main_config, optuna_config


def sample_params(trial, param_settings):
    params = {}

    for name, spec in param_settings.items():

        if spec["type"] == "float":
            if spec.get("log", False) and (spec["low"] <= 0 or spec["high"] <= 0):
                exp_val = trial.suggest_float(f"{name}_log10", spec["low"], spec["high"])
                params[name] = 10 ** exp_val
            else:
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )

        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])

        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])

        else:
            raise ValueError(f"Unsupported type: {spec['type']} for {name}")

    return params


# ============================================================
# Config builder (single source of truth)
# ============================================================
def build_trial_config(base_config, optuna_config, params, tmp_dir):

    config = copy.deepcopy(base_config)

    # -----------------------------
    # calibration parameters
    # -----------------------------
    config["calibration_settings"]["kappa_prior_vals"]["ell"] = params["ell_kappa"]
    config["calibration_settings"]["kappa_prior_vals"]["var"] = params["var_kappa"]

    config["calibration_settings"]["delta_eta_prior_vals"]["ell"] = params["ell_delta"]
    config["calibration_settings"]["delta_eta_prior_vals"]["var"] = params["var_delta"]

    config["calibration_settings"]["sigma2_prior_val"] = params["sigma2"]

    config["calibration_settings"]["mh_scale_kappa_prior"] = params["mh_scale_kappa"]

    # -----------------------------
    # mcmc settings
    # -----------------------------
    config["calibration_settings"]["N_mcmc"] = optuna_config["mcmc_settings"]["num_iterations"]
    config["calibration_settings"]["burn_in"] = optuna_config["mcmc_settings"]["burn_in"]
    config["calibration_settings"]["mh_scale_kappa_prior"] = optuna_config["mcmc_settings"]["mh_scale_kappa_prior"]
    config["calibration_settings"]["posterior_predictive_samples"] = optuna_config["mcmc_settings"]["num_posterior_predictive_samples"]

    # -----------------------------
    # input paths
    # -----------------------------
    config["input_settings"]["model_data_path"] = optuna_config["data_settings"]["model_data_path"]
    config["input_settings"]["obs_data_path"] = optuna_config["data_settings"]["observation_data_path"]
    config["input_settings"]["input_delimiter"] = optuna_config["data_settings"]["file_delimiter"]

    # -----------------------------
    # output 
    # -----------------------------
    config["output_settings"]["results_path"] = tmp_dir
    config["output_settings"]["figures_path"] = os.path.join(tmp_dir, "figures")

    os.makedirs(config["output_settings"]["figures_path"], exist_ok=True)

    return config


# ============================================================
# Run main safely
# ============================================================
def run_main_and_get_loss(config, tmp_dir):

    config_path = os.path.join(tmp_dir, "config.json")

    # run subprocess
    result = subprocess.run(
        ["python3", "main.py", config_path],
        capture_output=True,
        text=True
    )

    # --------------------------------------------------------
    # ALWAYS write log (even on crash)
    # --------------------------------------------------------
    log_file = os.path.join(tmp_dir, "main_run.log")

    with open(log_file, "w") as f:
        f.write("===== STDOUT =====\n")
        f.write(result.stdout or "")

        f.write("\n\n===== STDERR =====\n")
        f.write(result.stderr or "")

        f.write(f"\n\n===== RETURN CODE ===== {result.returncode}\n")

    # --------------------------------------------------------
    # failure case
    # --------------------------------------------------------
    if result.returncode != 0:
        return float("inf"), None

    # --------------------------------------------------------
    # read loss
    # --------------------------------------------------------
    loss_file = os.path.join(tmp_dir, "cross_validation_losses.json")

    if not os.path.exists(loss_file):
        return float("inf"), None

    with open(loss_file, "r") as f:
        loss_data = json.load(f)

    loss = loss_data.get("net_loss", float("inf"))

    return loss, loss_data


# ============================================================
# Trial summary
# ============================================================
def write_trial_summary(tmp_dir, params, loss_data):

    summary = {
        "params": params,
        "loss": None,
        "y_mse": None,
        "theta_mse": None,
        "delta_mse": None,
        "status": "failed"
    }

    if loss_data is not None:
        summary["loss"] = loss_data.get("net_loss", None)
        summary["y_mse"] = loss_data.get("y_mse", None)
        summary["theta_mse"] = loss_data.get("theta_mse", None)
        summary["delta_mse"] = loss_data.get("delta_mse", None)
        summary["status"] = "ok"

    path = os.path.join(tmp_dir, "trial_summary.json")

    with open(path, "w") as f:
        json.dump(summary, f, indent=4)


# ============================================================
# Objective
# ============================================================
def objective(trial):

    base_config, optuna_config = load_configs()

    params = sample_params(trial, optuna_config["param_settings"])

    base_results_dir = optuna_config["output_settings"]["results_path"]
    os.makedirs(base_results_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(dir=base_results_dir)

    # build full config
    config = build_trial_config(base_config, optuna_config, params, tmp_dir)

    # write config once
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # run main
    loss, loss_data = run_main_and_get_loss(config, tmp_dir)

    # write diagnostics
    write_trial_summary(tmp_dir, params, loss_data)

    return loss


# ============================================================
# Main
# ============================================================
def main():

    _, optuna_config = load_configs()

    study = optuna.create_study(
        direction=optuna_config["optuna_settings"]["direction"]
    )

    study.optimize(
        objective,
        n_trials=optuna_config["optuna_settings"]["num_trials"]
    )

    base_results_dir = optuna_config["output_settings"]["results_path"]
    os.makedirs(base_results_dir, exist_ok=True)

    with open(os.path.join(base_results_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(base_results_dir, "optuna_trials.csv"), index=False)

    save_optuna_diagnostics(study, base_results_dir)

    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()