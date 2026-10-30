"""Run calibration repeatedly, carrying shrinkage hyperparameter posteriors forward."""

import copy
import json
import math
import os
import subprocess
import sys


def _lognormal_mean(log_mean, log_std):
    return math.exp(log_mean + 0.5 * log_std**2)


def _set_prior_parameters(model, summary):
    """Apply a fitted posterior distribution and initialize at its mean."""
    kind = summary["kind"]
    if kind == "gp":
        prior_config = model["gp"]
        for parameter in ("ell", "var"):
            fitted = summary["parameters"][parameter]
            prior_config[f"{parameter}_prior"] = fitted
            prior_config[f"{parameter}_init"] = _lognormal_mean(
                fitted["mu"], fitted["sigma"]
            )
    elif kind == "gamma":
        prior_config = model["lasso"] if "lasso" in model else model["fused_lasso"]
        fitted = summary["parameters"]["lambda"]
        prior_config["lambda_prior"] = fitted
        prior_config["lambda_init"] = fitted["shape"] / fitted["rate"]
    elif kind == "beta":
        prior_config = model["spike_slab"]
        fitted = summary["parameters"]["pi"]
        prior_config["pi_prior"] = fitted
        prior_config["pi_init"] = fitted["alpha"] / (fitted["alpha"] + fitted["beta"])
    else:
        raise ValueError(f"Unsupported sequential prior family: {kind}")


def _apply_posterior(config, posterior):
    shrinkage = config["calibration_settings"]["shrinkage_models"]
    kappa_overrides = []

    for label, summary in posterior.items():
        if label == "delta_eta":
            model = shrinkage["delta_eta"]
            _set_prior_parameters(model, summary)
        elif label.startswith("kappa_"):
            model = shrinkage["kappa"]
            index = int(label.removeprefix("kappa_"))
            while len(kappa_overrides) <= index:
                kappa_overrides.append({})
            dimension_override = copy.deepcopy(model[model["shrinkage_prior"]])
            dimension_model = {
                "shrinkage_prior": model["shrinkage_prior"],
                model["shrinkage_prior"]: dimension_override,
            }
            _set_prior_parameters(dimension_model, summary)
            kappa_overrides[index] = dimension_model[model["shrinkage_prior"]]
        else:
            raise ValueError(f"Unrecognized hyperparameter source: {label}")

    if kappa_overrides:
        shrinkage["kappa"]["by_dimension"] = kappa_overrides


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python sequential_run.py <sequential_config.json>")

    sequential_config_path = os.path.abspath(sys.argv[1])
    with open(sequential_config_path, "r") as stream:
        sequential_config = json.load(stream)

    run_count = sequential_config.get("run_count")
    if not isinstance(run_count, int) or isinstance(run_count, bool) or run_count < 1:
        raise ValueError("run_count must be a positive integer")

    base_config_path = sequential_config.get("base_config", "config.json")
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.path.dirname(sequential_config_path), base_config_path)
    with open(base_config_path, "r") as stream:
        base_config = json.load(stream)

    # Generated run configs live in per-run folders, so resolve existing relative
    # inputs against the original config before writing those copies.
    base_dir = os.path.dirname(base_config_path)
    for section, keys in (("input_settings", ("model_data_path", "obs_data_path")),):
        settings = base_config.get(section, {})
        for key in keys:
            value = settings.get(key)
            if value and not os.path.isabs(value):
                settings[key] = os.path.abspath(os.path.join(base_dir, value))
    plot_path = base_config.get("plot_settings_path")
    if plot_path and not os.path.isabs(plot_path):
        base_config["plot_settings_path"] = os.path.abspath(os.path.join(base_dir, plot_path))

    output_root = sequential_config.get("output_path", "results/sequential")
    if not os.path.isabs(output_root):
        output_root = os.path.abspath(
            os.path.join(os.path.dirname(sequential_config_path), output_root)
        )
    os.makedirs(output_root, exist_ok=True)

    config = copy.deepcopy(base_config)
    shrinkage = config["calibration_settings"]["shrinkage_models"]
    for model in shrinkage.values():
        prior_kind = model.get("shrinkage_prior")
        if prior_kind == "gp":
            gp = model["gp"]
            for parameter in ("ell", "var"):
                prior = gp[f"{parameter}_prior"]
                gp[f"{parameter}_init"] = _lognormal_mean(prior["mu"], prior["sigma"])
        elif prior_kind in {"lasso", "fused_lasso"}:
            prior = model[prior_kind]
            prior["lambda_init"] = prior["lambda_prior"]["shape"] / prior["lambda_prior"]["rate"]
        elif prior_kind == "spike_slab":
            prior = model[prior_kind]
            beta_prior = prior["pi_prior"]
            prior["pi_init"] = beta_prior["alpha"] / (beta_prior["alpha"] + beta_prior["beta"])

    posterior = None
    for run_index in range(1, run_count + 1):
        if posterior is not None:
            _apply_posterior(config, posterior)

        run_dir = os.path.join(output_root, f"run_{run_index:03d}")
        os.makedirs(run_dir, exist_ok=True)
        config["output_settings"]["results_path"] = run_dir
        config["output_settings"]["figures_path"] = os.path.join(run_dir, "figures")
        run_config_path = os.path.join(run_dir, "config.json")
        with open(run_config_path, "w") as stream:
            json.dump(config, stream, indent=2)

        print(f"Starting sequential calibration run {run_index}/{run_count}: {run_config_path}", flush=True)
        main_path = os.path.join(os.path.dirname(__file__), "main.py")
        subprocess.run(
            [sys.executable, main_path, run_config_path],
            cwd=os.path.dirname(main_path),
            check=True,
        )

        posterior_path = os.path.join(run_dir, "hyperparameter_posterior.json")
        with open(posterior_path, "r") as stream:
            posterior = json.load(stream)
    print(f"Completed {run_count} sequential calibration runs in {output_root}")


if __name__ == "__main__":
    main()
