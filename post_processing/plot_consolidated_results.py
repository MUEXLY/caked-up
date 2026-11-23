"""Overlay real-space calibration results and correction fields from many runs.

Each ``--run`` points to a run's results directory (or its parent) and supplies
the display label used in the legend. The figure uses the publication-style
PPC layout: emulator prior and additive discrepancy across the top, one
full-width panel per kappa field, and posterior prediction plus hyperprior
densities along the bottom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy import stats


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


def hyperprior_distributions(config: dict) -> list[dict]:
    settings = config.get(
        "hyperprior_settings",
        config.get("calibration_settings", {}).get("hyperprior_settings", {}),
    )
    specifications = []

    kappa_fields = settings.get("kappa_fields", {})
    for field_name, field in kappa_fields.items():
        kind = field.get("shrinkage_prior")
        if kind in ("lasso", "fused_lasso"):
            prior = field.get(kind, {}).get("lambda_prior", {})
            specifications.append({
                "label": rf"$\lambda$ ({kind.replace('_', ' ')})",
                "kind": "gamma",
                "params": prior,
            })
        elif kind == "spike_slab":
            prior = field.get(kind, {}).get("pi_prior", {})
            specifications.append({
                "label": r"$\pi$ (spike-slab)",
                "kind": "beta",
                "params": prior,
            })
        elif kind == "horseshoe":
            prior = field.get(kind, {}).get("tau_prior", {})
            specifications.append({
                "label": r"$\tau$ (horseshoe)",
                "kind": "halfcauchy",
                "params": prior,
            })
        elif kind == "gp":
            gp = field.get("gp", {})
            specifications.extend([
                {"label": rf"$\ell$ ({field_name})", "kind": "lognormal",
                 "params": gp.get("ell_prior", {})},
                {"label": rf"$\sigma$ ({field_name})", "kind": "lognormal",
                 "params": gp.get("var_prior", {})},
            ])

    delta = settings.get("delta_eta")
    if isinstance(delta, dict):
        delta_gp = delta.get("gp", {})
        specifications.extend([
            {"label": r"$\ell$ ($\delta_\eta$)", "kind": "lognormal",
             "params": delta_gp.get("ell_prior", {})},
            {"label": r"$\sigma$ ($\delta_\eta$)", "kind": "lognormal",
             "params": delta_gp.get("var_prior", {})},
        ])
    return specifications


def density_points(specification: dict, x: np.ndarray) -> np.ndarray:
    kind = specification["kind"]
    prior = specification["params"]
    if kind == "gamma":
        shape = float(prior.get("shape", 1.0))
        rate = float(prior.get("rate", 1.0))
        return stats.gamma.pdf(x, a=shape, scale=1.0 / rate)
    if kind == "beta":
        alpha = float(prior.get("alpha", 1.0))
        beta = float(prior.get("beta", 1.0))
        return stats.beta.pdf(x, alpha, beta)
    if kind == "lognormal":
        mu = float(prior.get("mu", 0.0))
        sigma = float(prior.get("sigma", 0.5))
        if sigma <= 0:
            sigma = 0.5
        return stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    scale = float(prior.get("scale", prior.get("sigma", 1.0)))
    if scale <= 0:
        scale = 1.0
    return stats.cauchy.pdf(x, loc=0.0, scale=scale)


HYPERPRIOR_COLORS = {
    "lasso": "#0072B2",
    "fused_lasso": "#E69F00",
    "spike_slab": "#D55E00",
    "horseshoe": "#CC79A7",
    "gp_ell": "#56B4E9",
    "gp_sigma": "#009E73",
}

HYPERPRIOR_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#F0E442", "#7A76D1", "#B15A00", "#8B0000",
    "#20B2AA", "#6A5ACD",
]


def hyperprior_family(specification: dict) -> str:
    kind = specification["kind"]
    label = specification["label"].lower()
    if kind == "gamma":
        if "fused" in label:
            return "fused_lasso"
        if "lasso" in label:
            return "lasso"
    if kind == "beta":
        return "spike_slab"
    if kind == "halfcauchy":
        return "horseshoe"
    if kind == "lognormal":
        return "gp_ell" if "ell" in label or "length" in label else "gp_sigma"
    return "gp_sigma"


def specification_key(specification: dict) -> tuple:
    prior = specification["params"]
    return (
        specification["label"],
        hyperprior_family(specification),
        tuple(sorted((str(key), repr(value)) for key, value in prior.items())),
    )


def deduplicate_axis_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label != "_nolegend_" and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(
            list(unique.values()),
            list(unique.keys()),
            loc="best",
            fontsize="small",
            frameon=True,
        )


def known_delta_curve(config: dict, x: np.ndarray):
    cv = config.get("cross_validation_settings", {})
    if not cv.get("conduct_cross_validation", False):
        return None
    form = cv.get("known_delta_form")
    params = cv.get("known_delta_form_params", {}).get(form, {})
    if form == "power_law" and params:
        coeff = params.get("coeff")
        exponent = params.get("exponent")
        if coeff is not None and exponent is not None:
            return np.asarray(coeff * np.power(x, exponent), dtype=float)
    if form == "trig_funct":
        if params.get("function") == "sin":
            return -np.sin(x)
        if params.get("function") == "cos":
            return np.cos(x)
        return np.zeros_like(x)
    if form == "linear":
        return float(params.get("m", 0.0)) * x + float(params.get("b", 0.0))
    if form == "polynomial":
        coeffs = params.get("coeffs", [])
        if coeffs:
            return np.polyval(coeffs, x)
    return None


def known_theta_curve(config: dict, k: int, x: np.ndarray, theta_fixed=None):
    cv = config.get("cross_validation_settings", {})
    if not cv.get("conduct_cross_validation", False):
        return None
    form = cv.get("known_theta_form")
    params = cv.get("known_theta_form_params", {}).get(form, {})
    correction = lambda value: value
    theta_settings = config.get("calibration_settings", {}).get("theta_settings", {})
    if theta_settings.get("theta_initialization", "fixed") == "fixed":
        fixed_values = None
        if theta_fixed is not None:
            fixed_values = np.asarray(theta_fixed).ravel()
        if fixed_values is not None and k < fixed_values.size:
            fixed = float(fixed_values[k])
            correction = lambda value: (
                float(value) - fixed if np.isscalar(value)
                else np.asarray(value, dtype=float) - fixed
            )
    if form == "constant":
        values = params.get("values", [])
        return correction(float(values[k])) if k < len(values) else None
    if form == "trig_funct":
        functions = params.get("functions", [])
        if k >= len(functions):
            return None
        if functions[k] == "sin":
            return correction(np.sin(x))
        if functions[k] == "cos":
            return correction(np.cos(x))
        return correction(np.zeros_like(x))
    if form == "linear":
        return correction(float(params.get("m", 0.0)) * x + float(params.get("b", 0.0)))
    return None


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
    holdout = None
    for prefix in ("holdout_", "x_holdout_", "y_holdout_"):
        matching = [key for key in arrays if key.startswith(prefix)]
        if matching:
            break
    holdout_x = next(
        (arrays[key] for key in arrays if key in ("x_holdout", "holdout_x")),
        None,
    )
    holdout_y = next(
        (arrays[key] for key in arrays if key in ("y_holdout", "holdout_y")),
        None,
    )
    holdout_mean = next(
        (arrays[key] for key in arrays if key in ("y_holdout_post_mean", "holdout_post_mean")),
        None,
    )
    holdout_std = next(
        (arrays[key] for key in arrays if key in ("y_holdout_post_std", "holdout_post_std")),
        None,
    )
    if holdout_x is not None and holdout_y is not None:
        holdout = {
            "x": np.asarray(holdout_x).ravel(),
            "y": np.asarray(holdout_y).ravel(),
            "mean": None if holdout_mean is None else np.asarray(holdout_mean).ravel(),
            "std": None if holdout_std is None else np.asarray(holdout_std).ravel(),
        }
    return {
        "label": label,
        "archive": archive_path,
        "arrays": arrays,
        "x_key": selected_x,
        "obs_key": selected_obs,
        "kappa_keys": kappa_keys,
        "emulator": emulator,
        "config": config,
        "metrics": metrics,
        "holdout": holdout,
        "nrmse_text": nrmse_text,
        "legend": f"{label} ({nrmse_text})",
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
    fig = plt.figure(figsize=(12, max(11, 3.2 * nrows)))
    gs = fig.add_gridspec(
        nrows, 2,
        height_ratios=[1.15, 1.15] + [1.0] * (kappa_count + 1),
    )
    ax_emulator = fig.add_subplot(gs[0, 0])
    ax_hyperprior = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, :])
    kappa_axes = [fig.add_subplot(gs[k + 2, :]) for k in range(kappa_count)]
    ax_posterior = fig.add_subplot(gs[-1, :])
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(loaded)))
    legend_handles = []
    legend_entries = {}
    hyperprior_specs = {}
    hyperprior_colors = {}
    beta_ax = None
    delta_ax = None
    for run_index, (run, color) in enumerate(zip(loaded, colors)):
        a = run["arrays"]
        x = a[run["x_key"]]
        obs = a[run["obs_key"]]
        posterior_handle, = ax_posterior.plot(
            x, a["y_post_mean"], color=color, lw=1.8, label="_nolegend_"
        )
        ax_posterior.fill_between(
            x,
            a["y_post_mean"] - 1.96*a["y_post_std"],
            a["y_post_mean"] + 1.96*a["y_post_std"],
            color=color, alpha=.12,
        )
        if run["label"] not in legend_entries:
            legend_entries[run["label"]] = run["legend"]
            legend_handles.append(posterior_handle)
        ax_delta.plot(x, a["delta_eta_mean"], color=color, lw=1.8)
        ax_delta.fill_between(
            x,
            a["delta_eta_mean"] - 1.96*a["delta_eta_std"],
            a["delta_eta_mean"] + 1.96*a["delta_eta_std"],
            color=color, alpha=.12,
        )
        for k in range(kappa_count):
            key = f"kappa_{k}_mean"
            if key not in a:
                continue
            std_key = f"kappa_{k}_std"
            kappa_axes[k].plot(x, a[key], color=color, lw=1.8)
            kappa_axes[k].fill_between(
                x,
                a[key] - 1.96*a[std_key],
                a[key] + 1.96*a[std_key],
                color=color, alpha=.12,
            )
            known_theta = (
                known_theta_curve(
                    run["config"], k, x,
                    theta_fixed=a.get("theta_fixed_physical"),
                )
                if run_index == 0
                else None
            )
            if known_theta is not None:
                if np.isscalar(known_theta):
                    kappa_axes[k].axhline(
                        float(known_theta),
                        color="red", linestyle="--", lw=1.4,
                        label=r"$\kappa^{true}$",
                    )
                else:
                    kappa_axes[k].plot(
                        x, known_theta, color="red", linestyle="--", lw=1.4,
                        label=r"$\kappa^{true}(x)$",
                    )
        if run["emulator"] is not None:
            px, pm, ps = run["emulator"]
            order = np.argsort(px, kind="stable")
            px, pm, ps = px[order], pm[order], ps[order]
            ax_emulator.plot(px, pm, color=color, lw=1.8, ls="--")
            ax_emulator.fill_between(
                px, pm - 1.96*ps, pm + 1.96*ps,
                color=color, alpha=.08,
            )
            # emulator_prior.txt also stores the simulator training observations.
            prior_path = locate_file(run["archive"].parent, "emulator_prior.txt")
            try:
                prior_data = np.genfromtxt(prior_path, names=True, comments="#", delimiter="\t")
                if "x_phys" in prior_data.dtype.names and "y_phys" in prior_data.dtype.names:
                    ax_emulator.scatter(
                        prior_data["x_phys"], prior_data["y_phys"],
                        color="#303030", s=9, alpha=.25, edgecolors="none",
                    )
            except (OSError, ValueError, TypeError):
                pass
        known_delta = known_delta_curve(run["config"], x) if run_index == 0 else None
        if known_delta is not None:
            ax_delta.plot(
                x, known_delta, color="red", linestyle="--", lw=1.4,
                label=r"$\delta_\eta^{true}(x)$",
            )
        if run["holdout"] is not None:
            holdout = run["holdout"]
            ax_posterior.scatter(
                holdout["x"], holdout["y"], color=color,
                marker="x", s=28, alpha=.9,
                label="Cross-validation holdout",
            )
            if holdout["mean"] is not None:
                order = np.argsort(holdout["x"], kind="stable")
                hx = holdout["x"][order]
                hm = holdout["mean"][order]
                ax_posterior.plot(hx, hm, color=color, ls=":", lw=1.5)
                if holdout["std"] is not None:
                    hs = holdout["std"][order]
                    ax_posterior.fill_between(
                        hx, hm - 1.96*hs, hm + 1.96*hs,
                        color=color, alpha=.08,
                    )

        for specification in hyperprior_distributions(run["config"]):
            key = specification_key(specification)
            if key in hyperprior_specs:
                continue
            hyperprior_specs[key] = specification
            kind = specification["kind"]
            prior = specification["params"]
            family = hyperprior_family(specification)
            hyperprior_color = HYPERPRIOR_PALETTE[
                len(hyperprior_colors) % len(HYPERPRIOR_PALETTE)
            ]
            hyperprior_colors[key] = hyperprior_color
            if kind == "gamma":
                shape = float(prior.get("shape", 1.0))
                rate = float(prior.get("rate", 1.0))
                upper = stats.gamma.ppf(.995, a=shape, scale=1.0 / rate)
                grid = np.linspace(0, max(upper, 1e-6), 500)
            elif kind == "beta":
                grid = np.linspace(0.002, 0.998, 500)
                if beta_ax is None:
                    beta_ax = inset_axes(
                        ax_hyperprior,
                        width="52%",
                        height="52%",
                        loc="lower right",
                        borderpad=1.2,
                    )
                    beta_ax.set_title(r"$\pi$ (Beta prior)", fontsize="x-small")
                    beta_ax.set_xlim(0, 1)
                    beta_ax.set_xlabel(r"$\pi$", fontsize="x-small")
                    beta_ax.set_ylabel("Density", fontsize="x-small")
                    beta_ax.tick_params(labelsize="x-small")
                    beta_ax.grid(True, alpha=.18)
            elif kind == "lognormal":
                sigma = float(prior.get("sigma", 0.5))
                mu = float(prior.get("mu", 0.0))
                sigma = sigma if sigma > 0 else 0.5
                upper = stats.lognorm.ppf(.995, s=sigma, scale=np.exp(mu))
                grid = np.linspace(max(0.0, upper * 1e-4), upper, 500)
                if "delta_eta" in specification["label"].lower():
                    if delta_ax is None:
                        delta_ax = inset_axes(
                            ax_hyperprior,
                            width="44%",
                            height="48%",
                            loc="lower left",
                            borderpad=1.2,
                        )
                        delta_ax.set_title(r"$\delta_\eta$ GP priors", fontsize="x-small")
                        delta_ax.set_xlabel("Hyperparameter value", fontsize="x-small")
                        delta_ax.set_ylabel("Density", fontsize="x-small")
                        delta_ax.tick_params(labelsize="x-small")
                        delta_ax.grid(True, alpha=.18)
            else:
                scale = float(prior.get("scale", prior.get("sigma", 1.0)))
                scale = scale if scale > 0 else 1.0
                grid = np.linspace(0, stats.cauchy.ppf(.995, scale=scale), 500)
            density = density_points(specification, grid)
            if kind == "beta":
                target_ax = beta_ax
            elif "delta_eta" in specification["label"].lower():
                target_ax = delta_ax
            else:
                target_ax = ax_hyperprior
            target_ax.plot(
                grid, density, color=hyperprior_color, lw=1.8,
                label="_nolegend_",
            )
            target_ax.fill_between(
                grid, 0, density,
                color=hyperprior_color, alpha=.12,
            )

    # Observations are common to most grid comparisons; draw them once when
    # identical and otherwise show each data set with a neutral point style.
    first = loaded[0]
    same_observations = all(
        np.array_equal(run["arrays"][run["x_key"]], first["arrays"][first["x_key"]])
        and np.array_equal(run["arrays"][run["obs_key"]], first["arrays"][first["obs_key"]])
        for run in loaded[1:]
    )
    if same_observations:
        ax_emulator.scatter(
            first["arrays"][first["x_key"]],
            first["arrays"][first["obs_key"]],
            color="black", s=18, alpha=.7,
            label="Observed data", zorder=4,
        )
        ax_posterior.scatter(
            first["arrays"][first["x_key"]],
            first["arrays"][first["obs_key"]],
            color="black", s=18, alpha=.7,
            label="Observed data", zorder=4,
        )
    else:
        for run in loaded:
            a = run["arrays"]
            ax_emulator.scatter(
                a[run["x_key"]], a[run["obs_key"]],
                color="#303030", s=15, alpha=.45, zorder=4,
            )
            ax_posterior.scatter(
                a[run["x_key"]], a[run["obs_key"]],
                color="#303030", s=15, alpha=.45, zorder=4,
            )

    ax_emulator.set_title("Emulator Prior (No Discrepancy)")
    delta_nrmse_values = [
        run["metrics"].get("delta_nrmse")
        for run in loaded
        if run["metrics"].get("delta_nrmse") is not None
    ]
    ax_delta.set_title(
        r"Additive Discrepancy $\delta_\eta(x)$"
        + (
            rf" (NRMSE={float(np.mean(delta_nrmse_values)):.3g})"
            if delta_nrmse_values
            else ""
        )
    )
    ax_emulator.set_ylabel(emulator_y_label or "Emulator output")
    ax_posterior.set_ylabel(posterior_y_label or "Predicted response")
    ax_delta.set_ylabel(delta_y_label or "Correction")
    ax_emulator.set_xlabel(x_label or "Application domain")
    ax_delta.set_xlabel(x_label or "Application domain")
    ax_hyperprior.set_title("Prescribed Hyperprior Distributions")
    ax_hyperprior.set_ylabel("Density")
    ax_hyperprior.set_xlabel("Hyperparameter value")
    for k in range(kappa_count):
        kappa_title = rf"Calibration discrepancy: $\kappa_{k}(x)$"
        field_values = []
        for run in loaded:
            theta_nrmse = run["metrics"].get("theta_nrmse") or {}
            value = theta_nrmse.get(f"theta_{k}")
            if value is not None:
                field_values.append(float(value))
        if field_values:
            kappa_title += rf" (kappa NRMSE={float(np.mean(field_values)):.3g})"
        kappa_axes[k].set_title(kappa_title)
        kappa_axes[k].set_ylabel(kappa_y_label or "Parameter correction")
        kappa_axes[k].set_xlabel(x_label or "Application domain")
    ax_posterior.set_title("Full Posterior Prediction")
    ax_posterior.set_xlabel(x_label or "Application domain")
    for ax in [ax_emulator, ax_delta, ax_posterior, ax_hyperprior, *kappa_axes]:
        ax.grid(True, alpha=.22)
        if ax is not ax_hyperprior:
            deduplicate_axis_legend(ax)
    unique_hyperprior = {}
    for key, specification in hyperprior_specs.items():
        label = specification["label"]
        if label not in unique_hyperprior:
            unique_hyperprior[label] = Line2D(
                [0], [0],
                color=hyperprior_colors[key],
                lw=1.8,
            )
    if unique_hyperprior:
        ax_hyperprior.legend(
            list(unique_hyperprior.values()),
            list(unique_hyperprior.keys()),
            loc="upper right",
            fontsize="x-small",
            frameon=True,
        )
    fig.legend(
        legend_handles,
        list(legend_entries.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(loaded), 4),
        frameon=True,
        fontsize="small",
        title="Runs",
    )
    fig.suptitle(title, y=1.02, fontsize=15)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.04,
        top=0.94,
        hspace=0.52,
    )
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
