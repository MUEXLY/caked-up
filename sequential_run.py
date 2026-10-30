"""Run each kappa prior family in parallel, sequentially within each family."""

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
    if len(sys.argv) not in {2, 4}:
        raise ValueError(
            "Usage: python sequential_run.py <sequential_config.json> "
            "[--prior PRIOR | --prior-index INDEX]"
        )

    sequential_config_path = os.path.abspath(sys.argv[1])
    with open(sequential_config_path, "r") as stream:
        sequential_config = json.load(stream)

    run_count = sequential_config.get("run_count")
    if not isinstance(run_count, int) or isinstance(run_count, bool) or run_count < 1:
        raise ValueError("run_count must be a positive integer")

    prior_types = sequential_config.get(
        "kappa_prior_types", ["gp", "lasso", "fused_lasso", "spike_slab"]
    )
    supported_prior_types = {"gp", "lasso", "fused_lasso", "spike_slab"}
    if (
        not prior_types
        or any(not isinstance(prior, str) for prior in prior_types)
        or len(set(prior_types)) != len(prior_types)
        or any(prior not in supported_prior_types for prior in prior_types)
    ):
        raise ValueError(
            "kappa_prior_types must contain supported prior names: "
            + ", ".join(sorted(supported_prior_types))
        )

    selected_prior = None
    if len(sys.argv) == 4:
        selector, value = sys.argv[2:]
        if selector == "--prior":
            selected_prior = value
            if selected_prior not in prior_types:
                raise ValueError(f"Prior {selected_prior!r} is not listed in sequential_config.json")
        elif selector == "--prior-index":
            try:
                prior_index = int(value)
                if prior_index < 0:
                    raise IndexError
                selected_prior = prior_types[prior_index]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid kappa prior array index: {value}") from exc
        else:
            raise ValueError(f"Unknown option: {selector}")

    base_config_path = sequential_config.get("base_config", "config.json")
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.path.dirname(sequential_config_path), base_config_path)
    with open(base_config_path, "r") as stream:
        base_config = json.load(stream)

    if selected_prior is None:
        # Local invocation runs all prior families concurrently. On SLURM, the
        # array task selects one family and runs its chain serially.
        script_path = os.path.abspath(__file__)
        commands = [
            [sys.executable, script_path, sequential_config_path, "--prior", prior]
            for prior in prior_types
        ]
        processes = [
            subprocess.Popen(command, cwd=os.path.dirname(script_path))
            for command in commands
        ]
        failures = []
        for prior, process in zip(prior_types, processes):
            return_code = process.wait()
            if return_code:
                failures.append(f"{prior} (exit {return_code})")
        if failures:
            raise RuntimeError("Sequential prior runs failed: " + ", ".join(failures))
        return

    kappa_settings = base_config["calibration_settings"]["shrinkage_models"]["kappa"]
    if selected_prior not in kappa_settings:
        raise ValueError(
            f"Base config is missing kappa settings for prior {selected_prior!r}"
        )
    kappa_settings["shrinkage_prior"] = selected_prior
    kappa_settings.pop("by_dimension", None)

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
    output_root = os.path.join(output_root, selected_prior)
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
    print(
        f"Completed {run_count} sequential {selected_prior} calibration runs "
        f"in {output_root}"
    )


if __name__ == "__main__":
    main()
