"""Overlay real-space calibration results and correction fields from many runs.

Each ``--run`` points to a run's results directory (or its parent) and supplies
the display label used in the legend. The figure contains emulator prior,
posterior predictive, delta, and one panel per kappa field. Saved hyperprior
prescriptions and NRMSE values are summarized in the figure legend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def locate_file(directory: str | Path, filename: str) -> Path | None:
    directory = Path(directory).expanduser().resolve()
    for candidate in (directory / filename, directory / "results" / filename):
        if candidate.is_file():
            return candidate
    return None


def locate_archive(directory: str | Path) -> Path:
    path = locate_file(directory, "results_physical.npz")
    if path is None:
        raise FileNotFoundError(f"Could not find results_physical.npz under {directory}")
    return path


def select_key(files: list[str], requested: str | None, prefix: str, legacy: str) -> str:
    if requested:
        if requested not in files:
            raise KeyError(f"Array {requested!r} not found. Available arrays: {files}")
        return requested
    matches = [key for key in files if key.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches and legacy in files:
        return legacy
    raise ValueError(f"Cannot uniquely infer {prefix} array from {files}; specify the key")


def read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def prior_summary(config: dict) -> str:
    settings = config.get(
        "hyperprior_settings",
        config.get("calibration_settings", {}).get("hyperprior_settings", {}),
    )
    items = []
    kappa_fields = settings.get("kappa_fields", {})
    specifications = [(name, spec) for name, spec in kappa_fields.items()]
    specifications.extend(
        (name, spec) for name, spec in settings.items()
        if name != "kappa_fields" and isinstance(spec, dict)
    )
    for field, specification in specifications:
        kind = specification.get("shrinkage_prior", specification.get("kind", "?"))
        if field.startswith("kappa"):
            family = specification.get(kind, {})
            if kind in ("lasso", "fused_lasso"):
                prior = family.get("lambda_prior", {})
                bits = ",".join(f"{k}={v:g}" for k, v in prior.items() if isinstance(v, (int, float)))
            elif kind == "spike_slab":
                pi = family.get("pi_prior", {})
                bits = f"pi~Beta({pi.get('alpha','?')},{pi.get('beta','?')}), spike={family.get('sigma_spike','?')}, slab={family.get('sigma_slab','?')}"
            elif kind == "gp":
                bits = "GP ell/var priors"
            else:
                bits = ""
            items.append(f"{field}:{kind}" + (f"({bits})" if bits else ""))
        elif field == "delta_eta":
            gp = specification.get("gp", {})
            ell = gp.get("ell_prior", {})
            var = gp.get("var_prior", {})
            items.append(f"delta:GP ell={ell.get('mu','?')}/{ell.get('sigma','?')} var={var.get('mu','?')}/{var.get('sigma','?')}")
    return "; ".join(items) if items else "prior settings unavailable"


def load_run(directory: str, label: str, x_key: str | None, obs_key: str | None) -> dict:
    archive_path = locate_archive(directory)
    with np.load(archive_path) as archive:
        files = list(archive.files)
        selected_x = select_key(files, x_key, "x_obs_", "x_obs")
        selected_obs = select_key(files, obs_key, "zeta_obs_", "zeta_obs")
        arrays = {key: np.asarray(archive[key], dtype=float).reshape(-1) for key in files}

    x = arrays[selected_x]
    required = [selected_obs, "y_post_mean", "y_post_std", "delta_eta_mean", "delta_eta_std"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{archive_path} missing arrays: {missing}")
    order = np.argsort(x, kind="stable")
    for key in required:
        if len(arrays[key]) != len(x):
            raise ValueError(f"{archive_path}: {key} length {len(arrays[key])} differs from x length {len(x)}")
        arrays[key] = arrays[key][order]
    arrays[selected_x] = x[order]
    kappa_keys = sorted(
        (key for key in arrays if key.startswith("kappa_") and key.endswith("_mean")),
        key=lambda key: int(key.split("_")[1]),
    )
    for key in kappa_keys:
        std_key = key.removesuffix("_mean") + "_std"
        if std_key not in arrays or len(arrays[key]) != len(x) or len(arrays[std_key]) != len(x):
            raise ValueError(f"{archive_path}: incomplete or mismatched posterior arrays for {key}")
        arrays[key] = arrays[key][order]
        arrays[std_key] = arrays[std_key][order]

    root = Path(directory).expanduser().resolve()
    result_dir = archive_path.parent
    prior_path = locate_file(result_dir, "emulator_prior.txt")
    emulator = None
    if prior_path:
        try:
            prior = np.genfromtxt(prior_path, names=True, comments="#", delimiter="\t")
            names = prior.dtype.names or ()
            x_prior_key = "x_phys" if "x_phys" in names else "x_norm"
            mean_key = "y_prior_mean_phys" if "y_prior_mean_phys" in names else "y_prior_mean_norm"
            std_key = "y_prior_std_phys" if "y_prior_std_phys" in names else "y_prior_std_norm"
            if all(key in names for key in (x_prior_key, mean_key, std_key)):
                emulator = tuple(np.asarray(prior[key], dtype=float).reshape(-1) for key in (x_prior_key, mean_key, std_key))
        except (OSError, ValueError):
            pass

    config = read_json(locate_file(result_dir, "used_config.json"))
    metrics = read_json(locate_file(result_dir, "cross_validation_losses.json"))
    nrmse = metrics.get("net_nrmse", metrics.get("y_nrmse"))
    if nrmse is None:
        nrmse_text = "NRMSE n/a"
    else:
        nrmse_text = f"NRMSE={float(nrmse):.3g}"
    return {
        "label": label,
        "archive": archive_path,
        "arrays": arrays,
        "x_key": selected_x,
        "obs_key": selected_obs,
        "kappa_keys": kappa_keys,
        "emulator": emulator,
        "legend": f"{label} ({nrmse_text})\n{prior_summary(config)}",
    }


def plot_consolidated_results(
    runs: list[tuple[str, str]], output_path: str | Path, *, x_key: str | None = None,
    observation_key: str | None = None, title: str = "Consolidated real-space posterior results",
    x_label: str | None = None, posterior_y_label: str | None = None,
    delta_y_label: str | None = None, kappa_y_label: str | None = None,
    emulator_y_label: str | None = None,
) -> Path:
    if not runs:
        raise ValueError("Provide at least one --run RESULTS_DIRECTORY LABEL pair")
    loaded = [load_run(directory, label, x_key, observation_key) for directory, label in runs]
    kappa_count = max((len(run["kappa_keys"]) for run in loaded), default=0)
    nrows = 3 + kappa_count
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(10, 3.0 * nrows)), sharex=False, squeeze=False)
    axes = axes[:, 0]
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(loaded)))
    for run, color in zip(loaded, colors):
        a = run["arrays"]
        x = a[run["x_key"]]
        obs = a[run["obs_key"]]
        axes[1].plot(x, a["y_post_mean"], color=color, lw=1.8, label=run["legend"])
        axes[1].fill_between(x, a["y_post_mean"] - 1.96*a["y_post_std"], a["y_post_mean"] + 1.96*a["y_post_std"], color=color, alpha=.12)
        axes[2].plot(x, a["delta_eta_mean"], color=color, lw=1.8, label=run["label"])
        axes[2].fill_between(x, a["delta_eta_mean"] - 1.96*a["delta_eta_std"], a["delta_eta_mean"] + 1.96*a["delta_eta_std"], color=color, alpha=.12)
        for k in range(kappa_count):
            key = f"kappa_{k}_mean"
            if key not in a:
                continue
            std_key = f"kappa_{k}_std"
            axes[3+k].plot(x, a[key], color=color, lw=1.8, label=run["label"])
            axes[3+k].fill_between(x, a[key] - 1.96*a[std_key], a[key] + 1.96*a[std_key], color=color, alpha=.12)
        if run["emulator"] is not None:
            px, pm, ps = run["emulator"]
            order = np.argsort(px, kind="stable")
            px, pm, ps = px[order], pm[order], ps[order]
            axes[0].plot(px, pm, color=color, lw=1.8, ls="--", label=run["label"])
            axes[0].fill_between(px, pm - 1.96*ps, pm + 1.96*ps, color=color, alpha=.08)
            # emulator_prior.txt also stores the simulator training observations.
            prior_path = locate_file(run["archive"].parent, "emulator_prior.txt")
            try:
                prior_data = np.genfromtxt(prior_path, names=True, comments="#", delimiter="\t")
                if "x_phys" in prior_data.dtype.names and "y_phys" in prior_data.dtype.names:
                    axes[0].scatter(prior_data["x_phys"], prior_data["y_phys"], color="#303030", s=9, alpha=.25, edgecolors="none")
            except (OSError, ValueError, TypeError):
                pass

    # Observations are common to most grid comparisons; draw them once when
    # identical and otherwise show each data set with a neutral point style.
    first = loaded[0]
    same_observations = all(
        np.array_equal(run["arrays"][run["x_key"]], first["arrays"][first["x_key"]])
        and np.array_equal(run["arrays"][run["obs_key"]], first["arrays"][first["obs_key"]])
        for run in loaded[1:]
    )
    if same_observations:
        axes[1].scatter(first["arrays"][first["x_key"]], first["arrays"][first["obs_key"]], color="black", s=18, alpha=.7, label="Observed data", zorder=4)
    else:
        for run in loaded:
            a = run["arrays"]
            axes[1].scatter(a[run["x_key"]], a[run["obs_key"]], color="#303030", s=15, alpha=.45, zorder=4)

    axes[0].set_title("Previous emulator")
    axes[1].set_title("Posterior predictive")
    axes[2].set_title("Additive discrepancy δ(x)")
    axes[0].set_ylabel(emulator_y_label or "Emulator output")
    axes[1].set_ylabel(posterior_y_label or "Predicted response")
    axes[2].set_ylabel(delta_y_label or "Discrepancy")
    for k in range(kappa_count):
        axes[3+k].set_title(f"Kappa field κ{k}")
        axes[3+k].set_ylabel(kappa_y_label or "Parameter correction")
    for ax in axes:
        ax.grid(True, alpha=.22)
    axes[-1].set_xlabel(x_label or "Application domain")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.extend([Patch(facecolor="#777777", alpha=.12, label="95% interval")])
    labels.append("95% interval")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995), ncol=1, frameon=True, fontsize="small")
    fig.suptitle(title, y=1.0, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .91))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, required=True, metavar=("RESULTS_DIRECTORY", "LABEL"), help="Repeat for each calibration run.")
    parser.add_argument("--output", default="post_processing/consolidated_posterior_predictive.png")
    parser.add_argument("--x-key", help="Exact x array name for multi-domain results.")
    parser.add_argument("--observation-key", help="Exact observed-y array name for multi-output results.")
    parser.add_argument("--title", default="Consolidated real-space posterior results")
    parser.add_argument("--x-label", help="Override the shared domain axis title.")
    parser.add_argument("--emulator-y-label", help="Override the previous-emulator y-axis title.")
    parser.add_argument("--posterior-y-label", help="Override the posterior-prediction y-axis title.")
    parser.add_argument("--delta-y-label", help="Override the delta y-axis title.")
    parser.add_argument("--kappa-y-label", help="Override all kappa y-axis titles.")
    args = parser.parse_args()
    output = plot_consolidated_results(
        args.run, args.output, x_key=args.x_key, observation_key=args.observation_key,
        title=args.title, x_label=args.x_label, emulator_y_label=args.emulator_y_label,
        posterior_y_label=args.posterior_y_label, delta_y_label=args.delta_y_label,
        kappa_y_label=args.kappa_y_label,
    )
    print(f"Saved consolidated posterior figure to {output}")


if __name__ == "__main__":
    main()
