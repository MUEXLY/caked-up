import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _math_label(label):
    """Format a configured LaTeX command for Matplotlib mathtext."""
    label = str(label)
    if label.startswith("$") and label.endswith("$"):
        return label
    if label.startswith("\\"):
        return f"${label}$"
    return label


def _density(values, x_grid):
    """Return a KDE where possible, falling back for constant samples."""
    values = np.asarray(values, dtype=float)
    if np.ptp(values) == 0 or len(values) < 2:
        return np.zeros_like(x_grid)
    try:
        return gaussian_kde(values)(x_grid)
    except np.linalg.LinAlgError:
        return np.zeros_like(x_grid)


def plot_prior_posterior_densities(
    prior_samples,
    posterior_samples,
    parameter_labels,
    figures_directory,
    filename="prior_posterior_densities.png",
):
    """Plot prior theta and posterior theta* = theta^0 + delta_theta densities."""
    prior_samples = np.asarray(prior_samples, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    if prior_samples.ndim != 2 or posterior_samples.ndim != 2:
        raise ValueError("prior_samples and posterior_samples must be two-dimensional")
    if prior_samples.shape[1] != posterior_samples.shape[1]:
        raise ValueError("Prior and posterior must have the same parameter count")
    if len(parameter_labels) != prior_samples.shape[1]:
        raise ValueError("parameter_labels must match the parameter count")

    n_parameters = prior_samples.shape[1]
    fig, axes = plt.subplots(
        n_parameters, 1, figsize=(8, max(3, 2.8 * n_parameters)), squeeze=False
    )
    axes = axes[:, 0]
    for index, label in enumerate(parameter_labels):
        ax = axes[index]
        prior = prior_samples[:, index]
        posterior = posterior_samples[:, index]
        combined = np.concatenate((prior, posterior))
        x_grid = np.linspace(np.min(combined), np.max(combined), 300)
        if np.ptp(combined) > 0:
            ax.plot(
                x_grid,
                _density(prior, x_grid),
                label=r"Prior $\theta$",
            )
            ax.plot(
                x_grid,
                _density(posterior, x_grid),
                label=r"Posterior $\theta^*=\theta^0+\delta_\theta$",
            )
        else:
            ax.axvline(
                combined[0],
                label=r"Prior $\theta$ and posterior $\theta^*=\theta^0+\delta_\theta$",
            )
        ax.set_ylabel("Density")
        ax.set_title(_math_label(label))
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(r"Calibration parameter value: $\theta$ or $\theta^*$")
    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved prior-posterior density figure to {path}")
    return path


def _compute_emulator_sensitivities(
    emulator,
    emulator_inputs,
    theta_indices,
    finite_difference_step=1e-4,
):
    """Return local emulator derivatives with respect to each theta."""
    emulator_inputs = np.asarray(emulator_inputs, dtype=float)
    theta_indices = np.asarray(theta_indices, dtype=int)
    if emulator_inputs.ndim != 2:
        raise ValueError("emulator_inputs must be a two-dimensional array")
    if theta_indices.ndim != 1 or theta_indices.size == 0:
        raise ValueError("theta_indices must contain at least one index")
    if np.any(theta_indices < 0) or np.any(theta_indices >= emulator_inputs.shape[1]):
        raise ValueError("theta_indices must refer to emulator input columns")
    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive")

    sensitivity = np.empty((len(emulator_inputs), len(theta_indices)))
    for column, theta_index in enumerate(theta_indices):
        plus = emulator_inputs.copy()
        minus = emulator_inputs.copy()
        plus[:, theta_index] += finite_difference_step
        minus[:, theta_index] -= finite_difference_step
        plus_prediction = np.asarray(emulator.predict(plus)).reshape(-1)
        minus_prediction = np.asarray(emulator.predict(minus)).reshape(-1)
        sensitivity[:, column] = (
            plus_prediction - minus_prediction
        ) / (2 * finite_difference_step)
    return sensitivity


def plot_emulator_sensitivity_pca(
    emulator,
    emulator_inputs,
    theta_indices,
    theta_labels,
    figures_directory,
    filename="emulator_sensitivity_pca.png",
    finite_difference_step=1e-4,
):
    """Plot a PCA biplot of local emulator sensitivities to each theta.

    Sensitivities are finite-difference derivatives of the emulator response
    with respect to each normalized theta input. Each point is one emulator
    evaluation, while arrows show the contribution of each theta to the first
    two principal components.
    """
    theta_indices = np.asarray(theta_indices, dtype=int)
    if len(theta_labels) != theta_indices.size:
        raise ValueError("theta_labels must match theta_indices")
    sensitivity = _compute_emulator_sensitivities(
        emulator,
        emulator_inputs,
        theta_indices,
        finite_difference_step=finite_difference_step,
    )

    sensitivity_mean = sensitivity.mean(axis=0)
    sensitivity_scale = sensitivity.std(axis=0)
    sensitivity_scale[sensitivity_scale == 0] = 1.0
    standardized = (sensitivity - sensitivity_mean) / sensitivity_scale

    n_components = min(2, standardized.shape[0], standardized.shape[1])
    if np.allclose(standardized, 0):
        scores = np.zeros((len(standardized), 2))
        components = np.zeros((2, standardized.shape[1]))
        explained_variance = np.zeros(2)
    else:
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(standardized)
        if n_components == 1:
            scores = np.column_stack((scores[:, 0], np.zeros(len(scores))))
            components = np.vstack(
                (pca.components_[0], np.zeros(pca.n_features_in_))
            )
            explained_variance = np.array(
                [pca.explained_variance_ratio_[0], 0.0]
            )
        else:
            components = pca.components_[:2]
            explained_variance = pca.explained_variance_ratio_[:2]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    color_values = emulator_inputs[:, 0]
    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=color_values,
        cmap="viridis",
        s=28,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
        rasterized=len(scores) > 500,
    )

    arrow_scale = max(np.max(np.abs(scores)), 1.0) * 0.72
    for index, label in enumerate(theta_labels):
        vector = components[:, index]
        display_label = str(label)
        if display_label.startswith("\\") and not display_label.startswith("$"):
            display_label = f"${display_label}$"
        ax.arrow(
            0,
            0,
            vector[0] * arrow_scale,
            vector[1] * arrow_scale,
            color=f"C{index % 10}",
            width=0.006 * arrow_scale,
            head_width=0.045 * arrow_scale,
            length_includes_head=True,
            alpha=0.9,
        )
        ax.text(
            vector[0] * arrow_scale * 1.08,
            vector[1] * arrow_scale * 1.08,
            display_label,
            color=f"C{index % 10}",
            fontsize=11,
            ha="center",
            va="center",
        )

    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
    ax.set_title("Emulator sensitivity to calibration parameters")
    ax.grid(True, alpha=0.22)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Normalized application-domain input")
    fig.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved emulator sensitivity PCA figure to {path}")
    return path


def plot_emulator_sensitivity_boxplot(
    emulator,
    emulator_inputs,
    theta_indices,
    theta_labels,
    figures_directory,
    filename="emulator_sensitivity_boxplot.png",
    finite_difference_step=1e-4,
):
    """Plot a ranked boxplot of absolute local emulator sensitivities."""
    theta_indices = np.asarray(theta_indices, dtype=int)
    if len(theta_labels) != theta_indices.size:
        raise ValueError("theta_labels must match theta_indices")

    sensitivity = _compute_emulator_sensitivities(
        emulator,
        emulator_inputs,
        theta_indices,
        finite_difference_step=finite_difference_step,
    )
    absolute_sensitivity = np.abs(sensitivity)
    order = np.argsort(np.median(absolute_sensitivity, axis=0))[::-1]
    ranked_values = absolute_sensitivity[:, order]
    ranked_labels = [theta_labels[index] for index in order]

    fig_height = max(4.4, 0.72 * len(ranked_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    positions = np.arange(1, len(ranked_labels) + 1)
    box = ax.boxplot(
        [ranked_values[:, index] for index in range(len(ranked_labels))],
        vert=False,
        positions=positions,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "0.35",
            "markeredgecolor": "0.35",
            "alpha": 0.45,
        },
        medianprops={"color": "black", "linewidth": 1.6},
        whiskerprops={"color": "0.35", "linewidth": 1.1},
        capprops={"color": "0.35", "linewidth": 1.1},
    )
    for patch, index in zip(box["boxes"], range(len(ranked_labels))):
        patch.set_facecolor(plt.get_cmap("viridis")(
            index / max(len(ranked_labels) - 1, 1)
        ))
        patch.set_alpha(0.78)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.8)

    means = ranked_values.mean(axis=0)
    display_labels = [
        f"${label}$"
        if str(label).startswith("\\") and not str(label).startswith("$")
        else str(label)
        for label in ranked_labels
    ]
    ax.scatter(
        means,
        positions,
        marker="D",
        s=32,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
        label="Mean",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(display_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Absolute local sensitivity $|\partial \eta / \partial \theta|$")
    ax.set_title("Ranked emulator sensitivity to calibration parameters")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved emulator sensitivity boxplot to {path}")
    return path


def plot_holdout_predictions(
    holdout_cases,
    parameter_labels,
    figures_directory,
    full_x,
    full_y_true,
    full_y_pred,
    full_y_pred_std,
    full_parameter_corrections,
    filename="holdout_predictions.png",
):
    """Plot full-domain predictive checks and corrections for each trial.

    Each trial occupies one row. The first column shows the full posterior
    predictive over the application domain, with held-out points highlighted;
    the remaining columns show one correction field per calibration parameter.
    """
    if not holdout_cases:
        raise ValueError("holdout_cases must not be empty")
    full_x = np.asarray(full_x, dtype=float)
    full_x = full_x[:, 0] if full_x.ndim > 1 else full_x
    full_y_true = np.asarray(full_y_true, dtype=float).ravel()
    full_y_pred = np.asarray(full_y_pred, dtype=float).ravel()
    full_y_pred_std = np.asarray(full_y_pred_std, dtype=float).ravel()
    full_parameter_corrections = np.asarray(
        full_parameter_corrections, dtype=float
    )
    n_cases = len(holdout_cases)
    n_parameters = len(parameter_labels)
    if not (
        len(full_x)
        == len(full_y_true)
        == len(full_y_pred)
        == len(full_y_pred_std)
        == full_parameter_corrections.shape[0]
    ):
        raise ValueError("Full-domain arrays must have the same number of points")
    if full_parameter_corrections.shape[1] != n_parameters:
        raise ValueError("Correction columns must match parameter_labels")

    fig, axes = plt.subplots(
        n_cases,
        n_parameters + 1,
        figsize=(5 * (n_parameters + 1), max(4, 3.2 * n_cases)),
        squeeze=False,
        sharex="col",
    )

    for row, case in enumerate(holdout_cases):
        holdout_indices = np.asarray(case["indices"], dtype=int)
        prediction_ax = axes[row, 0]
        prediction_ax.fill_between(
            full_x,
            full_y_pred - 2 * full_y_pred_std,
            full_y_pred + 2 * full_y_pred_std,
            alpha=0.2,
            label="95% posterior interval",
        )
        prediction_ax.scatter(
            full_x, full_y_true, color="0.35", s=18, label="Observations"
        )
        prediction_ax.plot(full_x, full_y_pred, label="Posterior mean")
        prediction_ax.scatter(
            full_x[holdout_indices],
            full_y_true[holdout_indices],
            color="tab:red",
            marker="x",
            s=55,
            label="Holdout",
        )
        prediction_ax.set_title(
            f"Full posterior\nfraction={case['fraction']:.2g}, trial={row + 1}"
        )
        prediction_ax.set_xlabel("Application domain")
        prediction_ax.set_ylabel("Response")
        prediction_ax.grid(True, alpha=0.3)
        prediction_ax.legend(fontsize="small")

        for parameter_index, label in enumerate(parameter_labels):
            correction_ax = axes[row, parameter_index + 1]
            correction_ax.plot(
                full_x,
                full_parameter_corrections[:, parameter_index],
                color=f"C{parameter_index}",
            )
            correction_ax.scatter(
                full_x[holdout_indices],
                full_parameter_corrections[holdout_indices, parameter_index],
                color="tab:red",
                marker="x",
                s=55,
                label="Holdout",
            )
            correction_ax.axhline(0, color="black", linewidth=0.8)
            correction_ax.set_title(
                f"{label} correction\nfraction={case['fraction']:.2g}, "
                f"trial={row + 1}"
            )
            correction_ax.set_xlabel("Application domain")
            correction_ax.set_ylabel("Correction")
            correction_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved holdout prediction figure to {path}")
    return path


def plot_kappa_hyperprior_sensitivity(
    sensitivity_metrics,
    correction_labels,
    theta_labels,
    figures_directory,
    filename="kappa_hyperprior_sensitivity.png",
):
    """Plot posterior sensitivity to each delta_theta rho/sigma hyperprior."""
    scenarios = sensitivity_metrics.get("scenarios", [])
    if not scenarios:
        raise ValueError("sensitivity_metrics contains no scenarios")

    groups = []
    labels = []
    if len(correction_labels) != len(theta_labels):
        raise ValueError("correction_labels and theta_labels must have equal length")
    for kappa_index, (correction_label, theta_label) in enumerate(
        zip(correction_labels, theta_labels), start=1
    ):
        for hyperparameter in ("rho", "sigma"):
            values = [
                item["kappa_posterior_rmse"]
                for item in scenarios
                if item["kappa"] == kappa_index
                and item["hyperparameter"] == hyperparameter
            ]
            if values:
                groups.append(values)
                labels.append(
                    f"{_math_label(correction_label)}"
                    f"\n({_math_label(theta_label)})"
                    f"\n${hyperparameter}$"
                )

    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(groups)), 5.5))
    positions = np.arange(1, len(groups) + 1)
    box = ax.boxplot(
        groups,
        positions=positions,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("tab:blue" if index % 2 == 0 else "tab:orange")
        patch.set_alpha(0.72)
    ax.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Posterior $\delta_\theta$ RMS change")
    ax.set_title(
        r"Sensitivity of posterior $\delta_\theta$ fields to GP hyperpriors"
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved delta_theta hyperprior sensitivity figure to {path}")
    return path


def plot_holdout_nrmse_sensitivity(
    holdout_metrics,
    figures_directory,
    filename="holdout_nrmse_sensitivity.png",
    holdout_cases=None,
    full_x=None,
    full_y_true=None,
    full_y_pred=None,
    full_y_pred_std=None,
):
    """Plot aggregate NRMSE sensitivity with a largest-fraction posterior inset."""
    holdout_results = holdout_metrics.get("holdout_results", {})
    if not holdout_results:
        raise ValueError("holdout_metrics contains no holdout results")

    sorted_results = sorted(
        ((float(fraction), result) for fraction, result in holdout_results.items()),
        key=lambda item: item[0],
    )
    fractions = np.array([item[0] for item in sorted_results], dtype=float)
    result_values = [item[1] for item in sorted_results]
    means = np.array([result["nrmse"] for result in result_values])
    spreads = np.array([result["nrmse_std"] for result in result_values])
    full_nrmse = float(holdout_metrics["full_nrmse"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        full_nrmse,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Full dataset (NRMSE={full_nrmse:.3f})",
    )
    ax.fill_between(
        fractions,
        means - spreads,
        means + spreads,
        color="tab:blue",
        alpha=0.18,
        label="Holdout mean +/- 1 SD",
    )
    ax.errorbar(
        fractions,
        means,
        yerr=spreads,
        color="tab:blue",
        marker="o",
        linewidth=2,
        capsize=4,
        label="Mean holdout NRMSE",
    )
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    ax.set_xlabel("Fraction of observations withheld")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Holdout sensitivity across the application domain")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if holdout_cases and all(
        value is not None
        for value in (full_x, full_y_true, full_y_pred, full_y_pred_std)
    ):
        selected_case = max(
            holdout_cases,
            key=lambda case: float(case["fraction"]),
        )
        holdout_indices = np.asarray(selected_case["indices"], dtype=int)
        full_x = np.asarray(full_x, dtype=float).ravel()
        full_y_true = np.asarray(full_y_true, dtype=float).ravel()
        full_y_pred = np.asarray(full_y_pred, dtype=float).ravel()
        full_y_pred_std = np.asarray(full_y_pred_std, dtype=float).ravel()
        inset_ax = inset_axes(
            ax,
            width="43%",
            height="43%",
            loc="lower right",
            borderpad=1.2,
        )
        order = np.argsort(full_x)
        inset_ax.fill_between(
            full_x[order],
            full_y_pred[order] - 2 * full_y_pred_std[order],
            full_y_pred[order] + 2 * full_y_pred_std[order],
            color="tab:blue",
            alpha=0.2,
            label="95% posterior interval",
        )
        inset_ax.scatter(
            full_x,
            full_y_true,
            color="0.35",
            s=10,
            label="Observations",
        )
        inset_ax.plot(
            full_x[order],
            full_y_pred[order],
            color="tab:blue",
            linewidth=1.2,
            label="Posterior mean",
        )
        inset_ax.scatter(
            full_x[holdout_indices],
            full_y_true[holdout_indices],
            color="tab:red",
            marker="x",
            s=22,
            label="Holdout",
        )
        inset_ax.set_title(
            f"Full posterior\n"
            f"holdout={float(selected_case['fraction']):.1%}",
            fontsize=8,
        )
        inset_ax.tick_params(labelsize=7)
        inset_ax.grid(True, alpha=0.2)
        inset_ax.legend(fontsize=6, loc="best")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved holdout NRMSE sensitivity figure to {path}")
    return path


def plot_external_validation(
    x_sim,
    y_sim,
    x_observation,
    y_observation,
    x_obs,
    y_post_mean,
    y_post_std,
    x_external,
    y_external,
    y_external_pred,
    parameter_names,
    external_parameters,
    parameter_corrections,
    figures_directory,
    observation_set_label="Observation",
    simulation_set_label="Simulation",
    application_domain_label="Application domain",
    observation_label="Response",
    filename="external_validation.png",
    prior_parameters=None,
):
    """Plot validation points and prior/posterior parameter corrections.

    ``prior_parameters`` are the calibration routine's prior input points,
    while ``external_parameters`` are the parameter values from the external
    validation file used to form the corrected posterior points.
    """
    x_obs = np.asarray(x_obs, dtype=float).ravel()
    y_post_mean = np.asarray(y_post_mean, dtype=float).ravel()
    y_post_std = np.asarray(y_post_std, dtype=float).ravel()
    x_external = np.asarray(x_external, dtype=float).ravel()
    y_external = np.asarray(y_external, dtype=float).ravel()
    y_external_pred = np.asarray(y_external_pred, dtype=float).ravel()
    external_parameters = np.asarray(external_parameters, dtype=float)
    parameter_corrections = np.asarray(parameter_corrections, dtype=float)

    x_sim = np.asarray(x_sim, dtype=float).ravel()
    y_sim = np.asarray(y_sim, dtype=float).ravel()
    x_observation = np.asarray(x_observation, dtype=float).ravel()
    y_observation = np.asarray(y_observation, dtype=float).ravel()

    n_parameters = len(parameter_names)
    if prior_parameters is None:
        raise ValueError("prior_parameters must be provided")
    prior_parameters = np.asarray(prior_parameters, dtype=float)
    if prior_parameters.ndim != 2 or prior_parameters.shape[1] != n_parameters:
        raise ValueError(
            "prior_parameters must be a two-dimensional array with one column "
            "per parameter"
        )
    if external_parameters.ndim != 2 or external_parameters.shape[1] != n_parameters:
        raise ValueError(
            "external_parameters must be a two-dimensional array with one column "
            "per parameter"
        )
    if parameter_corrections.shape != external_parameters.shape:
        raise ValueError(
            "parameter_corrections must have the same shape as external_parameters"
        )
    if len(x_external) != len(external_parameters):
        raise ValueError("x_external must have one value per external parameter row")

    unique_external_x = np.unique(x_external)
    marker_cycle = ("x", "D", "^", "s", "P", "*")
    external_markers = {
        x_value: marker_cycle[index % len(marker_cycle)]
        for index, x_value in enumerate(unique_external_x)
    }

    fig, axes = plt.subplots(
        1,
        n_parameters + 1,
        figsize=(7 + 3.0 * n_parameters, 5.5),
        gridspec_kw={"width_ratios": [2.2] + [1] * n_parameters},
        squeeze=False,
    )
    prediction_ax = axes[0, 0]
    parameter_axes = axes[0, 1:]
    order = np.argsort(x_obs)
    prediction_ax.scatter(
        x_sim,
        y_sim,
        color="0.35",
        marker="x",
        alpha=0.45,
        label=f"{simulation_set_label} runs",
    )
    prediction_ax.scatter(
        x_observation,
        y_observation,
        color="black",
        s=35,
        alpha=0.7,
        label=f"{observation_set_label} data",
    )
    prediction_ax.plot(
        x_obs[order],
        y_post_mean[order],
        color="tab:blue",
        label="Calibration posterior predictive",
    )
    prediction_ax.fill_between(
        x_obs[order],
        y_post_mean[order] - 2 * y_post_std[order],
        y_post_mean[order] + 2 * y_post_std[order],
        color="tab:blue",
        alpha=0.2,
        label="95% posterior interval",
    )
    corrected_handles = []
    for x_value in unique_external_x:
        rows = x_external == x_value
        corrected_scatter = prediction_ax.scatter(
            x_external[rows],
            y_external[rows],
            color="tab:red",
            marker=external_markers[x_value],
            s=90 if external_markers[x_value] == "x" else 58,
            alpha=0.95,
            edgecolors="black" if external_markers[x_value] != "x" else None,
            linewidths=1.2 if external_markers[x_value] == "x" else 0.6,
            label="_nolegend_",
        )
        corrected_handles.append(corrected_scatter)
    # prediction_ax.scatter(
    #     x_external,
    #     y_external_pred,
    #     color="tab:green",
    #     marker="D",
    #     s=48,
    #     edgecolors="black",
    #     label="Emulator at corrected parameters",
    # )
    # for x_value, observed, predicted in zip(
    #     x_external, y_external, y_external_pred
    # ):
    #     prediction_ax.plot(
    #         [x_value, x_value],
    #         [observed, predicted],
    #         color="tab:green",
    #         alpha=0.35,
    #         linewidth=1,
    #     )
    prediction_ax.set_xlabel(_math_label(application_domain_label))
    prediction_ax.set_ylabel(_math_label(observation_label))
    prediction_ax.set_title("External validation against calibration posterior")
    prediction_ax.grid(True, alpha=0.3)
    prediction_ax.legend()

    prior_handle = None
    for index, (parameter_ax, parameter_name) in enumerate(
        zip(parameter_axes, parameter_names)
    ):
        prior_scatter = parameter_ax.scatter(
            np.zeros(len(prior_parameters)),
            prior_parameters[:, index],
            color="0.45",
            marker="o",
            s=20,
            alpha=0.65,
        )
        if prior_handle is None:
            prior_handle = prior_scatter

        for x_value in unique_external_x:
            rows = x_external == x_value
            posterior_scatter = parameter_ax.scatter(
                np.ones(np.count_nonzero(rows)),
                (external_parameters + parameter_corrections)[rows, index],
                color="tab:orange",
                marker=external_markers[x_value],
                s=42,
                alpha=0.85,
                edgecolors="black" if external_markers[x_value] != "x" else None,
                linewidths=0.6,
            )
        parameter_ax.set_xticks([0, 1])
        parameter_ax.set_xticklabels(["Prior", "Posterior"], rotation=30)
        parameter_ax.set_ylabel("Parameter value")
        parameter_ax.set_title(_math_label(parameter_name))
        parameter_ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [prior_handle] + corrected_handles
    legend_labels = [r"Prior $\theta$"] + [
        f"Corrected {simulation_set_label} at {_math_label(application_domain_label)}={x_value:g}"
        for x_value in unique_external_x
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(1 + len(corrected_handles), 3),
        fontsize="large",
        frameon=False,
        markerscale=0.9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved external validation figure to {path}")
    return path


def generate_rawData_figure(
    model_data,
    obs_data,
    figures_directory,
    observation_set_label="Observation",
    simulation_set_label="Simulation",
    theta_labels=None,
    application_domain_label=None,
    observation_label=None,
):
    """
    Create subplots for each theta parameter showing:
    - X-axis: application domain (x values)
    - Y-axis: calibration metric (y values)
    - Point color: theta parameter value
    - Observed data points
    """
    
    model_x_columns = [col for col in model_data.columns if col.startswith('x_')]
    model_y_columns = [col for col in model_data.columns if col.startswith('y_')]
    observation_x_columns = [col for col in obs_data.columns if col.startswith('x_')] if obs_data is not None else []
    observation_y_columns = [col for col in obs_data.columns if col.startswith('xi_')] if obs_data is not None else []
    
    theta_columns = [col for col in model_data.columns if col.startswith('theta_')]
    
    if not model_x_columns or not model_y_columns or not theta_columns:
        print("Missing required columns (x_, y_, or theta_)")
        return
    
    # Use first y column as calibration metric
    y_col = model_y_columns[0]
    x_col = model_x_columns[0]
    
    if theta_labels is None:
        theta_labels = [column.replace("theta_", "") for column in theta_columns]
    if len(theta_labels) != len(theta_columns):
        raise ValueError("theta_labels must match the theta column count")
    application_domain_label = application_domain_label or x_col
    observation_label = observation_label or y_col

    # Use a fixed three-column layout for consistent publication figures.
    n_theta = len(theta_columns)
    n_cols = 3
    n_rows = int(np.ceil(n_theta / n_cols))
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    axes = np.atleast_1d(axes).flatten()
    
    for idx, theta_col in enumerate(theta_columns):
        ax = axes[idx]
        
        # Scatter plot colored by theta parameter value
        scatter = ax.scatter(model_data[x_col], model_data[y_col], 
                            c=model_data[theta_col], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plot observed data
        if obs_data is not None and observation_y_columns and observation_y_columns[0] in obs_data.columns and observation_x_columns and observation_x_columns[0] in obs_data.columns:
            ax.scatter(
                obs_data[observation_x_columns[0]],
                obs_data[observation_y_columns[0]],
                color='red',
                label=f'{observation_set_label} data',
                alpha=0.8,
                s=30,
                edgecolors='black',
            )
        
        ax.set_xlabel(_math_label(application_domain_label))
        ax.set_ylabel(_math_label(observation_label))
        ax.set_title(
            f"{_math_label(observation_label)} colored by "
            f"{_math_label(theta_labels[idx])}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(_math_label(theta_labels[idx]))
    
    # Hide unused subplots
    for idx in range(n_theta, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    # plt_path = os.path.join(figures_directory, f'rawData_theta_parameters.png')
    # # plt.savefig(plt_path, dpi=150)
    # # print(f'Saved raw data figure to {plt_path}')
    # plt.close()
    # plt.show()
    plt_path = os.path.join(figures_directory, f'rawData_theta_parameters.png')
    plt.savefig(plt_path, dpi=150)
    print(f'Saved raw data figure to {plt_path}')
    plt.close()

def plot_delta_acceptance_trajectory(accept_trace, window=50, figures_directory=None):
    """
    Plot acceptance-rate trajectories for each discrepancy GP delta_k.

    Parameters
    ----------
    accept_trace : list of lists
        accept_trace[k][t] = 1 if delta_k accepted at iteration t, else 0

    window : int
        Rolling window size for smoothing acceptance rates

    Output
    ------
    Displays a trajectory plot of rolling acceptance rates.
    """

    n_delta = len(accept_trace)

    plt.figure(figsize=(10, 6))

    for k in range(n_delta):
        accepts = np.array(accept_trace[k])

        # Rolling acceptance rate
        rolling = np.convolve(
            accepts,
            np.ones(window) / window,
            mode="valid"
        )

        plt.plot(
            rolling,
            label=f"delta[{k}]"
        )

    plt.axhline(0.25, linestyle="--", label="Target ~0.25")
    plt.axhline(0.50, linestyle="--", label="Target ~0.50")

    plt.xlabel("Iteration")
    plt.ylabel(f"Rolling Acceptance Rate (window={window})")
    plt.title("Delta GP Acceptance Rate Trajectories")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt_path = os.path.join(figures_directory, f"delta_acceptance_trajectory.png")
    plt.savefig(plt_path, dpi=150)
    print(f"Saved delta acceptance trajectory figure to {plt_path}")
    plt.close()

def plot_delta_jump_sizes(delta_chain, figures_directory=None):
    """
    Plot jump magnitudes ||delta^(t) - delta^(t-1)|| for each delta_k.

    Parameters
    ----------
    delta_chain : array
        Shape (Nmcmc, dtheta, n_obs)
        Stored delta latent fields across iterations.

    Output
    ------
    Line plot of jump norms for each discrepancy GP.
    """

    Nmcmc, dtheta, n_obs = delta_chain.shape

    plt.figure(figsize=(10, 6))

    for k in range(dtheta):

        # difference between successive iterations
        diffs = delta_chain[1:, k, :] - delta_chain[:-1, k, :]

        # norm of each jump
        jump_norms = np.linalg.norm(diffs, axis=1)

        plt.plot(jump_norms, label=f"delta[{k}] jump size")

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\delta^{(t)} - \delta^{(t-1)}\|$")
    plt.title("Delta GP Jump Magnitudes (Mixing Diagnostic)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt_path = os.path.join(figures_directory, f"delta_jump_sizes.png")
    plt.savefig(plt_path, dpi=150)
    print(f"Saved delta jump size figure to {plt_path}")
    plt.close()