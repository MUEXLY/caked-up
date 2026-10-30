import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from scipy.stats import beta as beta_distribution
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm
from scipy.stats import halfcauchy, gaussian_kde
from functs import get_theta_at_obs, eta_predict
from sklearn.decomposition import PCA


def _domain_matrix(values):
    values = np.asarray(values, dtype=float)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def _domain_labels(columns, n):
    labels = [str(c) for c in columns]
    return labels if len(labels) == n else [f"x_{j}" for j in range(n)]

def _configured_label(value, index, fallback):
    if isinstance(value, dict):
        return str(value.get(str(index), value.get(index, fallback)))
    if isinstance(value, (list, tuple)) and index < len(value):
        return str(value[index])
    return fallback


def _format_plot_label(label):
    label = str(label)
    if len(label) >= 2 and label.startswith("$") and label.endswith("$"):
        return label
    if any(token in label for token in ("\\", "^", "_")):
        return f"${label}$"
    return label


def _plot_label(plot_settings, key, index, fallback):
    return _format_plot_label(_configured_label(plot_settings.get(key), index, fallback))


def _compute_emulator_sensitivities(
    emulator,
    emulator_inputs,
    parameter_indices,
    finite_difference_step=1e-4,
):
    """Compute local emulator derivatives with respect to selected inputs."""
    emulator_inputs = np.asarray(emulator_inputs, dtype=float)
    parameter_indices = np.asarray(parameter_indices, dtype=int)
    if emulator_inputs.ndim != 2:
        raise ValueError("emulator_inputs must be two-dimensional")
    if parameter_indices.ndim != 1 or parameter_indices.size == 0:
        raise ValueError("parameter_indices must contain at least one index")
    if np.any(parameter_indices < 0) or np.any(
        parameter_indices >= emulator_inputs.shape[1]
    ):
        raise ValueError("parameter_indices contain invalid emulator columns")
    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive")

    sensitivity = np.empty((len(emulator_inputs), len(parameter_indices)))
    for column, parameter_index in enumerate(parameter_indices):
        plus = emulator_inputs.copy()
        minus = emulator_inputs.copy()
        plus[:, parameter_index] += finite_difference_step
        minus[:, parameter_index] -= finite_difference_step
        sensitivity[:, column] = (
            np.asarray(emulator.predict(plus)).reshape(-1)
            - np.asarray(emulator.predict(minus)).reshape(-1)
        ) / (2 * finite_difference_step)
    return sensitivity


def plot_emulator_sensitivity_boxplot(
    emulator,
    emulator_inputs,
    parameter_indices,
    parameter_labels,
    figures_directory,
    filename="emulator_sensitivity_boxplot.png",
    finite_difference_step=1e-4,
):
    """Plot ranked distributions of absolute local emulator sensitivities."""
    parameter_indices = np.asarray(parameter_indices, dtype=int)
    if len(parameter_labels) != parameter_indices.size:
        raise ValueError("parameter_labels must match parameter_indices")

    sensitivity = _compute_emulator_sensitivities(
        emulator, emulator_inputs, parameter_indices, finite_difference_step
    )
    absolute_sensitivity = np.abs(sensitivity)
    order = np.argsort(np.median(absolute_sensitivity, axis=0))[::-1]
    ranked_values = absolute_sensitivity[:, order]
    ranked_labels = [parameter_labels[index] for index in order]

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
        patch.set_facecolor(
            plt.get_cmap("viridis")(index / max(len(ranked_labels) - 1, 1))
        )
        patch.set_alpha(0.78)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.8)

    means = ranked_values.mean(axis=0)
    ax.scatter(
        means, positions, marker="D", s=32, color="white", edgecolor="black",
        linewidth=0.8, zorder=3, label="Mean",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([_format_plot_label(label) for label in ranked_labels])
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


def plot_emulator_sensitivity_pca(
    emulator,
    emulator_inputs,
    parameter_indices,
    parameter_labels,
    figures_directory,
    filename="emulator_sensitivity_pca.png",
    finite_difference_step=1e-4,
):
    """Plot principal directions of local emulator sensitivity."""
    parameter_indices = np.asarray(parameter_indices, dtype=int)
    if len(parameter_labels) != parameter_indices.size:
        raise ValueError("parameter_labels must match parameter_indices")

    sensitivity = _compute_emulator_sensitivities(
        emulator, emulator_inputs, parameter_indices, finite_difference_step
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
            explained_variance = np.array([pca.explained_variance_ratio_[0], 0.0])
        else:
            components = pca.components_[:2]
            explained_variance = pca.explained_variance_ratio_[:2]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    color_values = np.asarray(emulator_inputs)[:, 0]
    scatter = ax.scatter(
        scores[:, 0], scores[:, 1], c=color_values, cmap="viridis",
        s=28, alpha=0.78, edgecolors="white", linewidths=0.35,
        rasterized=len(scores) > 500,
    )
    arrow_scale = max(np.max(np.abs(scores)), 1.0) * 0.72
    for index, label in enumerate(parameter_labels):
        vector = components[:, index]
        ax.arrow(
            0, 0, vector[0] * arrow_scale, vector[1] * arrow_scale,
            color=f"C{index % 10}", width=0.006 * arrow_scale,
            head_width=0.045 * arrow_scale, length_includes_head=True,
            alpha=0.9,
        )
        ax.text(
            vector[0] * arrow_scale * 1.08,
            vector[1] * arrow_scale * 1.08,
            _format_plot_label(label),
            color=f"C{index % 10}", fontsize=11, ha="center", va="center",
        )

    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
    ax.set_title("Emulator sensitivity to calibration parameters")
    ax.grid(True, alpha=0.22)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("First emulator input")
    fig.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved emulator sensitivity PCA figure to {path}")
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
    posterior_parameters=None,
    external_discrepancy=None,
    external_discrepancy_std=None,
):
    """Plot external validation with corrected parameters and additive discrepancy."""
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
    prior_parameters = np.asarray(prior_parameters, dtype=float)
    posterior_parameters = np.asarray(posterior_parameters, dtype=float)
    if prior_parameters.ndim != 2 or prior_parameters.shape[1] != n_parameters:
        raise ValueError(
            "prior_parameters must have one column per calibration parameter"
        )
    if posterior_parameters.shape != external_parameters.shape:
        raise ValueError(
            "posterior_parameters must match external_parameters in shape"
        )
    if external_parameters.ndim != 2 or external_parameters.shape[1] != n_parameters:
        raise ValueError(
            "external_parameters must have one column per calibration parameter"
        )
    if parameter_corrections.shape != external_parameters.shape:
        raise ValueError(
            "parameter_corrections must match external_parameters in shape"
        )
    if len(x_external) != len(external_parameters):
        raise ValueError("x_external must have one value per external parameter row")
    if external_discrepancy is not None:
        external_discrepancy = np.asarray(external_discrepancy, dtype=float).ravel()
        if len(external_discrepancy) != len(x_external):
            raise ValueError("external_discrepancy must match x_external in length")
    if external_discrepancy_std is not None:
        external_discrepancy_std = np.asarray(
            external_discrepancy_std, dtype=float
        ).ravel()
        if len(external_discrepancy_std) != len(x_external):
            raise ValueError(
                "external_discrepancy_std must match x_external in length"
            )

    unique_external_x = np.unique(x_external)
    marker_cycle = ("x", "D", "^", "s", "P", "*")
    external_markers = {
        value: marker_cycle[index % len(marker_cycle)]
        for index, value in enumerate(unique_external_x)
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
        x_sim, y_sim, color="0.35", marker="x", alpha=0.45,
        label=f"{simulation_set_label} runs",
    )
    prediction_ax.scatter(
        x_observation, y_observation, color="black", s=35, alpha=0.7,
        label=f"{observation_set_label} data",
    )
    prediction_ax.plot(
        x_obs[order], y_post_mean[order], color="tab:blue",
        label="Calibration posterior predictive",
    )
    prediction_ax.fill_between(
        x_obs[order],
        y_post_mean[order] - 2 * y_post_std[order],
        y_post_mean[order] + 2 * y_post_std[order],
        color="tab:blue", alpha=0.2, label="95% posterior interval",
    )

    corrected_handles = []
    for x_value in unique_external_x:
        rows = x_external == x_value
        marker = external_markers[x_value]
        corrected = prediction_ax.scatter(
            x_external[rows], y_external[rows], color="tab:red", marker=marker,
            s=90 if marker == "x" else 58, alpha=0.95,
            edgecolors="black" if marker != "x" else None,
            linewidths=1.2 if marker == "x" else 0.6, label="_nolegend_",
        )
        corrected_handles.append(corrected)

    prediction_ax.scatter(
        x_external, y_external_pred, color="tab:green", marker="D", s=48,
        edgecolors="black", label="Corrected prediction + additive discrepancy",
    )
    if external_discrepancy_std is not None:
        prediction_ax.errorbar(
            x_external, y_external_pred, yerr=2 * external_discrepancy_std,
            fmt="none", ecolor="tab:green", alpha=0.35, capsize=2,
        )
    application_domain_label = _format_plot_label(application_domain_label)
    observation_label = _format_plot_label(observation_label)
    prediction_ax.set_xlabel(application_domain_label)
    prediction_ax.set_ylabel(observation_label)
    prediction_ax.set_title("External validation against calibration posterior")
    prediction_ax.grid(True, alpha=0.3)
    prediction_ax.legend()

    prior_handle = None
    posterior_handle = None
    external_handle = None
    for index, (parameter_ax, parameter_name) in enumerate(
        zip(parameter_axes, parameter_names)
    ):
        prior_scatter = parameter_ax.scatter(
            np.zeros(len(prior_parameters)), prior_parameters[:, index],
            color="0.45", marker="o", s=20, alpha=0.65,
        )
        if prior_handle is None:
            prior_handle = prior_scatter
        posterior_scatter = parameter_ax.scatter(
            np.ones(len(posterior_parameters)),
            posterior_parameters[:, index],
            color="tab:blue",
            marker="o",
            s=34,
            alpha=0.8,
        )
        if posterior_handle is None:
            posterior_handle = posterior_scatter
        for x_value in unique_external_x:
            rows = x_external == x_value
            marker = external_markers[x_value]
            external_scatter = parameter_ax.scatter(
                np.full(np.count_nonzero(rows), 2.0),
                external_parameters[rows, index],
                color="tab:red", marker=marker, s=42, alpha=0.85,
                edgecolors="black" if marker != "x" else None,
                linewidths=0.6,
            )
            if external_handle is None:
                external_handle = external_scatter
        parameter_ax.set_xticks([0, 1, 2])
        parameter_ax.set_xticklabels(["Prior", "Posterior", "External"], rotation=30)
        parameter_ax.set_ylabel("Parameter value")
        parameter_ax.set_title(_format_plot_label(parameter_name))
        parameter_ax.grid(True, axis="y", alpha=0.3)

    fig.legend(
        [prior_handle, posterior_handle, external_handle] + corrected_handles,
        [r"Prior $\theta$", r"Calibration posterior $\theta$", r"External-file $\theta$"] + [
            f"Corrected {simulation_set_label} at {application_domain_label}={value:g}"
            for value in unique_external_x
        ],
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


def generate_rawData_figure(model_data, obs_data, figures_directory, figure_name="raw_data_theta_parameters.png", suptitle="Raw Data Colored by Theta Parameters", plot_settings=None):
    """
    Create subplots for each theta parameter showing:
    - X-axis: application domain (x values)
    - Y-axis: calibration metric (y values)
    - Point color: theta parameter value
    - Observed data points
    """
    
    plot_settings = plot_settings or {}
    model_x_columns = [col for col in model_data.columns if col.startswith('x_')]
    model_y_columns = [col for col in model_data.columns if col.startswith('zeta_')]
    observation_x_columns = [col for col in obs_data.columns if col.startswith('x_')] if obs_data is not None else []
    observation_y_columns = [col for col in obs_data.columns if col.startswith('zeta_')] if obs_data is not None else []
    
    theta_columns = [col for col in model_data.columns if col.startswith('theta_')]
    
    if not model_x_columns or not model_y_columns or not theta_columns:
        print("Missing required columns (x_, y_, or theta_)")
        return
    
    # Use first y column as calibration metric
    y_col = model_y_columns[0]
    x_values = _domain_matrix(model_data[model_x_columns].values)
    x_obs_values = (
        _domain_matrix(obs_data[observation_x_columns].values)
        if obs_data is not None and observation_x_columns else None
    )
    n_domain = x_values.shape[1]
    
    # Create subplot grid for theta parameters
    n_theta = len(theta_columns)
    n_cols = n_domain
    n_rows = n_theta
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, theta_col in enumerate(theta_columns):
        for j in range(n_domain):
            ax = axes[idx * n_domain + j]
        
        # Scatter plot colored by theta parameter value
            scatter = ax.scatter(x_values[:, j], model_data[y_col],
                            c=model_data[theta_col], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plot observed data
            if obs_data is not None and observation_y_columns and x_obs_values is not None:
                ax.scatter(x_obs_values[:, j], obs_data[observation_y_columns[0]], color='red', label='Observed Data', alpha=0.8, s=30, edgecolors='black')
        
            ax.set_xlabel(
                _plot_label(plot_settings, "application_domain_labels", j, model_x_columns[j])
            )
            ax.set_ylabel(f'{y_col}')
            ax.set_title(f'{y_col} vs {model_x_columns[j]}, colored by {theta_col}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Add colorbar for each subplot
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(theta_col)
    
    # Hide unused subplots
    for idx in range(n_theta * n_domain, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    # plt_path = os.path.join(figures_directory, f'rawData_theta_parameters.png')
    # # plt.savefig(plt_path, dpi=150)
    # # print(f'Saved raw data figure to {plt_path}')
    # plt.close()
    # plt.show()
    plt_path = os.path.join(figures_directory, figure_name)
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


def plot_hyperparameter_updates(
    hyperparameter_trace,
    hyperparameter_priors,
    initial_values,
    parameter_labels,
    figures_directory,
    filename="hyperparameter_updates.png",
    animation_filename="hyperparameter_updates.gif",
    make_animation=False,
    animation_stride=100,
):
    """Plot GP hyperparameter traces against their configured hyperpriors."""
    trace = np.asarray(hyperparameter_trace, dtype=float)
    if trace.ndim != 3 or trace.shape[2] != 2:
        raise ValueError("hyperparameter_trace must have shape (steps, groups, 2)")
    if len(parameter_labels) != trace.shape[1]:
        raise ValueError("parameter_labels must match the trace groups")

    initial = np.asarray(initial_values, dtype=float)
    if initial.shape != (trace.shape[1], 2):
        raise ValueError("initial_values must have shape (groups, 2)")

    n_steps, n_groups, _ = trace.shape
    names = ("ell", "var")

    def prior_interval(prior, name):
        values = prior.get(f"{name}_prior", {})
        mu = float(values["mu"])
        sigma = float(values["sigma"])
        return np.exp(mu - 1.96 * sigma), np.exp(mu + 1.96 * sigma), np.exp(mu)

    fig, axes = plt.subplots(
        n_groups, 2, figsize=(11, max(3.2, 2.8 * n_groups)), squeeze=False
    )
    for group in range(n_groups):
        for column, name in enumerate(names):
            ax = axes[group, column]
            lower, upper, median = prior_interval(
                hyperparameter_priors[group], name
            )
            ax.axhspan(lower, upper, color="tab:orange", alpha=0.16,
                       label="95% hyperprior interval")
            ax.axhline(median, color="tab:orange", linestyle="--",
                       label="Hyperprior median")
            ax.plot(trace[:, group, column], color="tab:blue",
                    linewidth=1.0, label="MCMC state")
            ax.axhline(initial[group, column], color="black", linestyle=":",
                       label="Initial value")
            ax.set_yscale("log")
            ax.set_xlabel("MCMC iteration")
            ax.set_ylabel(name)
            ax.set_title(f"{parameter_labels[group]}: {name}")
            ax.grid(True, alpha=0.25)
            if group == 0 and column == 0:
                ax.legend(loc="best")
    fig.suptitle("GP hyperparameter updates and configured hyperpriors")
    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hyperparameter trace figure to {path}")

    if not make_animation:
        return path

    frame_indices = np.unique(np.append(
        np.arange(0, n_steps, max(1, int(animation_stride))), n_steps - 1
    ))
    fig, axes = plt.subplots(
        n_groups, 2, figsize=(11, max(3.2, 2.8 * n_groups)), squeeze=False
    )
    y_limits = []
    for group in range(n_groups):
        limits = []
        for column, name in enumerate(names):
            lower, upper, _ = prior_interval(hyperparameter_priors[group], name)
            limits.extend([lower, upper, trace[:, group, column].min(),
                           trace[:, group, column].max()])
        y_limits.append((min(limits) * 0.8, max(limits) * 1.25))

    def update(frame_position):
        step = int(frame_indices[frame_position])
        for group in range(n_groups):
            for column, name in enumerate(names):
                ax = axes[group, column]
                ax.clear()
                lower, upper, median = prior_interval(
                    hyperparameter_priors[group], name
                )
                ax.axhspan(lower, upper, color="tab:orange", alpha=0.16)
                ax.axhline(median, color="tab:orange", linestyle="--")
                ax.plot(trace[:step + 1, group, column], color="tab:blue")
                ax.axhline(initial[group, column], color="black", linestyle=":")
                ax.set_xlim(0, max(n_steps - 1, 1))
                ax.set_ylim(y_limits[group])
                ax.set_yscale("log")
                ax.set_xlabel("MCMC iteration")
                ax.set_ylabel(name)
                ax.set_title(f"{parameter_labels[group]}: {name} (iteration {step})")
                ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return axes.ravel().tolist()

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), blit=False, repeat=False
    )
    gif_path = os.path.join(figures_directory, animation_filename)
    animation.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved hyperparameter animation to {gif_path}")
    return path, gif_path


def plot_hyperparameter_density_animation(
    hyperparameter_traces,
    hyperparameter_specs,
    figures_directory,
    filename="hyperparameter_density_updates.gif",
    animation_stride=100,
):
    """Animate hyperparameter states over their configured prior densities.

    Each specification must contain ``label``, ``kind`` (``lognormal``,
    ``gamma``, or ``beta``), and the corresponding distribution parameters.
    """
    if len(hyperparameter_traces) != len(hyperparameter_specs):
        raise ValueError("Traces and hyperparameter specifications must match")
    if not hyperparameter_traces:
        raise ValueError("At least one hyperparameter trace is required")

    traces = [np.asarray(values, dtype=float).ravel()
              for values in hyperparameter_traces]
    n_steps = len(traces[0])
    if any(len(values) != n_steps for values in traces):
        raise ValueError("All hyperparameter traces must have equal length")

    def distribution(spec):
        kind = spec["kind"]
        if kind == "lognormal":
            prior = spec["prior"]
            return lognorm(
                s=float(prior["sigma"]),
                scale=np.exp(float(prior["mu"])),
            )
        if kind == "gamma":
            prior = spec["prior"]
            return gamma_distribution(
                a=float(prior["shape"]),
                scale=1.0 / float(prior["rate"]),
            )
        if kind == "beta":
            prior = spec["prior"]
            return beta_distribution(
                a=float(prior["alpha"]), b=float(prior["beta"])
            )
        raise ValueError(f"Unsupported hyperparameter density kind: {kind}")

    distributions = [distribution(spec) for spec in hyperparameter_specs]
    grids = []
    densities = []
    for dist, values in zip(distributions, traces):
        finite_values = values[np.isfinite(values) & (values > 0)]
        lower = max(float(dist.ppf(1e-4)), np.min(finite_values) * 0.5)
        upper = max(float(dist.ppf(1 - 1e-4)), np.max(finite_values) * 1.5)
        if not np.isfinite(lower) or lower <= 0:
            lower = max(np.min(finite_values) * 0.5, 1e-12)
        if not np.isfinite(upper) or upper <= lower:
            upper = lower * 100.0
        grid = np.geomspace(lower, upper, 500)
        grids.append(grid)
        densities.append(dist.pdf(grid))

    frame_indices = np.unique(np.append(
        np.arange(0, n_steps, max(1, int(animation_stride))), n_steps - 1
    ))
    n_panels = len(traces)
    n_columns = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(6.0 * n_columns, 3.8 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    def update(frame_position):
        step = int(frame_indices[frame_position])
        for panel, (ax, spec, values, grid, density) in enumerate(
            zip(axes_flat, hyperparameter_specs, traces, grids, densities)
        ):
            ax.clear()
            current = values[step]
            ax.plot(grid, density, color="tab:blue", linewidth=2)
            if np.isfinite(current) and current > 0:
                band = np.array([current / 1.08, current * 1.08])
                band = np.clip(band, grid[0], grid[-1])
                mask = (grid >= band[0]) & (grid <= band[1])
                if np.any(mask):
                    ax.fill_between(
                        grid[mask], density[mask], color="tab:red", alpha=0.35,
                        label="Current-state region",
                    )
                ax.axvline(
                    current, color="tab:red", linestyle="--", linewidth=2,
                    label=f"Current = {current:.4g}",
                )
            ax.set_xscale("log")
            ax.set_xlabel(spec["name"])
            ax.set_ylabel("Prior density")
            ax.set_title(f'{spec["label"]} ({spec["kind"]}) — iteration {step}')
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.tight_layout()
        return axes_flat.tolist()

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), blit=False, repeat=False
    )
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    animation.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved hyperparameter density animation to {path}")
    return path


def plot_calibration_prediction_animation(
    x_obs,
    y_obs,
    kappa_theta_chain,
    delta_eta_chain,
    theta_fixed,
    gp_eta,
    figures_directory,
    filename="calibration_prediction_updates.gif",
    frame_stride=100,
    max_posterior_samples=200,
    y_scale=1.0,
    y_offset=0.0,
    x_scale=1.0,
    x_offset=0.0,
    theta_scales=None,
    theta_offsets=None,
    theta_labels=None,
    space_label="physical",
):
    """Animate cumulative kappa, additive-discrepancy, and response predictions."""
    x = np.asarray(x_obs, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y_obs, dtype=float).reshape(-1)
    kappa_chain = np.asarray(kappa_theta_chain, dtype=float)
    delta_chain = np.asarray(delta_eta_chain, dtype=float)
    theta_base = np.asarray(theta_fixed, dtype=float)

    n_steps, dtheta, n_obs = kappa_chain.shape
    if len(x) != n_obs or len(y) != n_obs:
        raise ValueError("Prediction animation inputs have inconsistent row counts")
    if delta_chain.shape != (n_steps, n_obs):
        raise ValueError("delta_eta_chain must have shape (steps, observations)")

    theta_scales = (
        np.ones(dtheta) if theta_scales is None
        else np.asarray(theta_scales, dtype=float)
    )
    theta_offsets = (
        np.zeros(dtheta) if theta_offsets is None
        else np.asarray(theta_offsets, dtype=float)
    )
    theta_labels = theta_labels or [rf"$\theta_{k}$" for k in range(dtheta)]
    if len(theta_labels) != dtheta:
        raise ValueError("theta_labels must match the kappa dimension")

    x_plot = x[:, 0] * x_scale + x_offset
    y_plot = y * y_scale + y_offset
    kappa_plot = kappa_chain * theta_scales[None, :, None]
    delta_plot = delta_chain * y_scale
    frame_indices = np.unique(np.append(
        np.arange(0, n_steps, max(1, int(frame_stride))), n_steps - 1
    ))

    def prediction_snapshot(step):
        prefix = np.arange(step + 1)
        if len(prefix) > max_posterior_samples:
            prefix = np.linspace(0, step, max_posterior_samples, dtype=int)
        predictions = np.empty((len(prefix), n_obs))
        for sample_index, chain_index in enumerate(prefix):
            for observation_index in range(n_obs):
                theta_i = get_theta_at_obs(theta_base, observation_index)
                theta_star = theta_i + kappa_chain[
                    chain_index, :, observation_index
                ]
                mean, _ = eta_predict(x[observation_index], theta_star, gp_eta)
                predictions[sample_index, observation_index] = (
                    mean + delta_chain[chain_index, observation_index]
                )
        predictions = predictions * y_scale + y_offset
        return predictions.mean(axis=0), predictions.std(axis=0)

    n_rows = dtheta + 2
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, max(7, 2.8 * n_rows)), squeeze=False
    )
    axes = axes[:, 0]

    def update(frame_position):
        step = int(frame_indices[frame_position])
        prefix = slice(0, step + 1)
        for k, ax in enumerate(axes[:dtheta]):
            ax.clear()
            mean = kappa_plot[prefix, k].mean(axis=0)
            std = kappa_plot[prefix, k].std(axis=0)
            ax.plot(
                x_plot, mean, color=f"C{k}",
                label=f"{theta_labels[k]} discrepancy",
            )
            ax.fill_between(
                x_plot, mean - 2 * std, mean + 2 * std,
                color=f"C{k}", alpha=0.25, label="±2 std",
            )
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylabel(rf"$\kappa_{k}$")
            ax.set_title(f"Calibration discrepancy: {theta_labels[k]}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        delta_mean = delta_plot[prefix].mean(axis=0)
        delta_std = delta_plot[prefix].std(axis=0)
        axes[dtheta].clear()
        axes[dtheta].plot(
            x_plot, delta_mean, color="tab:purple",
            label=r"$\delta_\eta$ mean",
        )
        axes[dtheta].fill_between(
            x_plot, delta_mean - 2 * delta_std,
            delta_mean + 2 * delta_std,
            color="tab:purple", alpha=0.25, label="±2 std",
        )
        axes[dtheta].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[dtheta].set_title("Additive discrepancy")
        axes[dtheta].set_ylabel(r"$\delta_\eta$")
        axes[dtheta].grid(True, alpha=0.25)
        axes[dtheta].legend(loc="best")

        prediction_mean, prediction_std = prediction_snapshot(step)
        axes[-1].clear()
        axes[-1].scatter(
            x_plot, y_plot, color="black", s=18, label="Observations"
        )
        axes[-1].plot(
            x_plot, prediction_mean, color="tab:blue",
            label="Reality prediction",
        )
        axes[-1].fill_between(
            x_plot, prediction_mean - 2 * prediction_std,
            prediction_mean + 2 * prediction_std,
            color="tab:blue", alpha=0.25, label="±2 std",
        )
        axes[-1].set_title(
            f"Full posterior predictive ({space_label}) — iteration {step}"
        )
        axes[-1].set_ylabel("Response")
        axes[-1].set_xlabel("Application domain")
        axes[-1].grid(True, alpha=0.25)
        axes[-1].legend(loc="best")
        fig.tight_layout()
        return axes.tolist()

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), blit=False, repeat=False
    )
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    animation.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved calibration prediction animation to {path}")
    return path


def plot_prior_posterior_comparison(
    theta_prior_physical,
    theta_posterior_physical,
    theta_labels,
    hyperparameter_traces,
    hyperparameter_specs,
    figures_directory,
    filename="prior_posterior_comparison.png",
    burn_in=0,
):
    """Overlay physical input and shrinkage-hyperparameter prior/posterior densities."""
    theta_prior = np.asarray(theta_prior_physical, dtype=float)
    theta_posterior = np.asarray(theta_posterior_physical, dtype=float)
    if theta_prior.ndim != 2 or theta_posterior.ndim != 2:
        raise ValueError("Theta prior and posterior arrays must be two-dimensional")
    if theta_prior.shape[1] != theta_posterior.shape[1]:
        raise ValueError("Theta prior and posterior dimensions must match")
    if len(theta_labels) != theta_prior.shape[1]:
        raise ValueError("theta_labels must match the theta dimension")
    if len(hyperparameter_traces) != len(hyperparameter_specs):
        raise ValueError("Hyperparameter traces and specifications must match")

    def make_distribution(spec):
        prior = spec["prior"]
        if spec["kind"] == "lognormal":
            return lognorm(s=float(prior["sigma"]), scale=np.exp(float(prior["mu"])))
        if spec["kind"] == "gamma":
            return gamma_distribution(
                a=float(prior["shape"]), scale=1.0 / float(prior["rate"])
            )
        if spec["kind"] == "beta":
            return beta_distribution(
                a=float(prior["alpha"]), b=float(prior["beta"])
            )
        if spec["kind"] == "halfcauchy":
            return halfcauchy(scale=float(prior.get("scale", 1.0)))
        raise ValueError(f"Unsupported hyperparameter density kind: {spec['kind']}")

    def density_curve(values, points=300):
        values = np.asarray(values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot plot a density for an empty sample")
        if np.allclose(values, values[0]):
            spread = max(abs(values[0]) * 0.05, 1e-8)
            grid = np.linspace(values[0] - spread, values[0] + spread, points)
            density = np.exp(-0.5 * ((grid - values[0]) / (spread / 2)) ** 2)
            density /= np.trapz(density, grid)
            return grid, density
        grid = np.linspace(values.min(), values.max(), points)
        return grid, gaussian_kde(values)(grid)

    def prior_curve(spec, posterior_values):
        dist = make_distribution(spec)
        positive = posterior_values[posterior_values > 0]
        lower = float(dist.ppf(1e-4))
        upper = float(dist.ppf(1 - 1e-4))
        if not np.isfinite(lower):
            lower = max(float(np.min(positive)) * 0.5, 1e-12)
        if not np.isfinite(upper):
            upper = max(float(np.max(positive)) * 1.5, lower * 10)
        lower = min(lower, float(np.min(posterior_values)))
        upper = max(upper, float(np.max(posterior_values)))
        grid = np.linspace(lower, upper, 400)
        return grid, dist.pdf(grid)

    n_theta = theta_prior.shape[1]
    n_hyper = len(hyperparameter_specs)
    n_columns = max(n_theta, n_hyper, 1)
    fig, axes = plt.subplots(
        2, n_columns,
        figsize=(4.5 * n_columns, 8),
        squeeze=False,
    )

    for k in range(n_theta):
        ax = axes[0, k]
        prior_grid, prior_density = density_curve(theta_prior[:, k])
        post_grid, post_density = density_curve(theta_posterior[:, k])
        ax.plot(prior_grid, prior_density, color="tab:orange", linewidth=2,
                label="Input prior")
        ax.plot(post_grid, post_density, color="tab:blue", linewidth=2,
                label="Posterior")
        ax.fill_between(prior_grid, prior_density, color="tab:orange", alpha=0.18)
        ax.fill_between(post_grid, post_density, color="tab:blue", alpha=0.18)
        ax.set_title(f"{theta_labels[k]}: physical input")
        ax.set_xlabel(theta_labels[k])
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for k in range(n_theta, n_columns):
        axes[0, k].set_visible(False)

    for h, (trace, spec) in enumerate(zip(
        hyperparameter_traces, hyperparameter_specs
    )):
        ax = axes[1, h]
        values = np.asarray(trace, dtype=float).ravel()
        values = values[min(max(int(burn_in) + 1, 0), len(values) - 1):]
        if values.size == 0:
            raise ValueError(f"Empty posterior trace for {spec['label']}")
        prior_grid, prior_density = prior_curve(spec, values)
        post_grid, post_density = density_curve(values)
        ax.plot(prior_grid, prior_density, color="tab:orange", linewidth=2,
                label="Hyperprior")
        ax.plot(post_grid, post_density, color="tab:blue", linewidth=2,
                label="Hyperparameter posterior")
        ax.fill_between(prior_grid, prior_density, color="tab:orange", alpha=0.18)
        ax.fill_between(post_grid, post_density, color="tab:blue", alpha=0.18)
        ax.set_title(f"{spec['label']}: {spec['name']}")
        ax.set_xlabel(spec["name"])
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for h in range(n_hyper, n_columns):
        axes[1, h].set_visible(False)

    fig.suptitle("Prior–posterior comparison")
    fig.tight_layout()
    os.makedirs(figures_directory, exist_ok=True)
    path = os.path.join(figures_directory, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prior-posterior comparison figure to {path}")
    return path


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

def plot_emulator_prior(y_prior_mean, y_prior_var, model_data, model_normalized, reverse_normalization, figures_directory=None, output_directory=None, plot_settings=None):
    plot_settings = plot_settings or {}
    
    # ---- normalized ----
    x_columns = [col for col in model_normalized.columns if col.startswith('x')]
    x_sim_norm = _domain_matrix(model_normalized[x_columns].values)
    y_sim_norm = model_normalized[[col for col in model_normalized.columns if col.startswith('zeta')]].values.ravel()

    # ---- physical ----
    x_sim_phys = _domain_matrix(model_data[[col for col in model_data.columns if col.startswith('x')]].values)
    y_sim_phys = model_data[[col for col in model_data.columns if col.startswith('zeta')]].values.ravel()

    y_prior_std = np.sqrt(y_prior_var)

    # Reverse normalization
    y_col = list(reverse_normalization["y"].keys())[0]
    y_mu = reverse_normalization["y"][y_col]["mean"]
    y_sd = reverse_normalization["y"][y_col]["std"]

    y_prior_mean_phys = y_prior_mean * y_sd + y_mu
    y_prior_std_phys = y_prior_std * y_sd

    # Sort indices
    n_domain = x_sim_norm.shape[1]
    fig, axes = plt.subplots(2, n_domain, figsize=(6 * n_domain, 8), squeeze=False)
    for j in range(n_domain):
        idx_norm = np.argsort(x_sim_norm[:, j])
        idx_phys = np.argsort(x_sim_phys[:, j])
        for row, x_values, y_values, prior, std, title, color, idx in [
            (0, x_sim_norm[:, j], y_sim_norm, y_prior_mean, y_prior_std, "Normalized Space", "orange", idx_norm),
            (1, x_sim_phys[:, j], y_sim_phys, y_prior_mean_phys, y_prior_std_phys, "Physical Space", "green", idx_phys),
        ]:
            ax = axes[row, j]
            ax.scatter(
                x_values, y_values, color="blue", alpha=0.6,
                label=plot_settings.get("simulation_data_label", "Simulator Data"),
            )
            ax.plot(x_values[idx], prior[idx], color=color, label="Emulator Prior Mean")
            ax.fill_between(
                x_values[idx], (prior - 2 * std)[idx], (prior + 2 * std)[idx],
                color=color, alpha=0.2, label="Prior ±2σ",
            )
            ax.set_title(f"{title}: {x_columns[j]}")
            domain_label = _plot_label(
                plot_settings, "application_domain_labels", j, x_columns[j]
            )
            ax.set_xlabel(
                f"{domain_label} ({'normalized' if row == 0 else 'physical'})"
            )
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt_path = os.path.join(figures_directory, f"emulator_prior.png")
    plt.savefig(plt_path, dpi=150)
    print(f"Saved emulator prior figure to {plt_path}")
    plt.close()

    #if output_directory is not none, save the plot data to txt files for later use in animations
    if output_directory is not None:
        output_data = np.column_stack([
            x_sim_norm, y_sim_norm, y_prior_mean, y_prior_std,
            x_sim_phys, y_sim_phys, y_prior_mean_phys, y_prior_std_phys
        ])
        header = "x_norm\ty_norm\ty_prior_mean_norm\ty_prior_std_norm\tx_phys\ty_phys\ty_prior_mean_phys\ty_prior_std_phys"
        np.savetxt(os.path.join(output_directory, "emulator_prior.txt"), output_data, header=header, delimiter="\t")

        return()


def plot_discrepancy_diagnostics(
    x_obs,
    y_obs,
    x_sim,
    y_sim,
    y_prior_mean,
    y_prior_var,
    delta_eta_mean,
    delta_eta_std,
    y_post_mean,
    y_post_var,
    theta_fixed,
    theta_fixed_phys,
    gp_eta,
    kappa_mean,
    kappa_std,
    idx,
    dtheta,
    calibration_settings,
    cross_validation_settings,
    holdout_data=None,
    figure_path=None,
    figure_name="discrepancy_diagnostics.png",
    suptitle="Discrepancy Diagnostics",
    plot_settings=None,
    domain_index=0,
    theta_labels=None,
    theta_prior_min=None,
    theta_prior_max=None,
    theta_fixed_plot=None,
):
    plot_settings = plot_settings or {}
    x_input = _domain_matrix(x_obs)
    if x_input.shape[1] > 1:
        x_sim_matrix = _domain_matrix(x_sim)
        for j in range(x_input.shape[1]):
            domain_label = _plot_label(
                plot_settings, "application_domain_labels", j, f"x_{j}"
            )
            holdout_j = holdout_data
            if holdout_data is not None and "x" in holdout_data:
                holdout_j = dict(holdout_data)
                holdout_j["x"] = _domain_matrix(holdout_data["x"])[:, j]
            plot_discrepancy_diagnostics(
                x_input[:, j], y_obs, x_sim_matrix[:, j], y_sim,
                y_prior_mean, y_prior_var, delta_eta_mean, delta_eta_std,
                y_post_mean, y_post_var, theta_fixed, theta_fixed_phys,
                gp_eta, kappa_mean, kappa_std, idx, dtheta,
                calibration_settings, cross_validation_settings,
                holdout_data=holdout_j, figure_path=figure_path,
                figure_name=f"{os.path.splitext(figure_name)[0]}_x{j}.png",
                suptitle=f"{suptitle} (domain coordinate {domain_label})",
                plot_settings=plot_settings,
                domain_index=j,
                theta_labels=theta_labels,
                theta_prior_min=theta_prior_min,
                theta_prior_max=theta_prior_max,
                theta_fixed_plot=theta_fixed_plot,
            )
        return

    x = np.asarray(x_obs).ravel()
    y_obs = np.asarray(y_obs).ravel()
    y_prior_mean = np.asarray(y_prior_mean).ravel()
    y_prior_var = np.asarray(y_prior_var).ravel()
    delta_eta_mean = np.asarray(delta_eta_mean).ravel()
    delta_eta_std = np.asarray(delta_eta_std).ravel()
    y_post_mean = np.asarray(y_post_mean).ravel()
    y_post_var = np.asarray(y_post_var).ravel()

    # Every domain coordinate is an independent 1-D plot. Sort the complete
    # observation series together so lines and uncertainty bands follow x.
    obs_order = np.argsort(x)
    prior_on_x = (
        y_prior_mean.shape[0] == x.shape[0]
        and y_prior_var.shape[0] == x.shape[0]
    )
    x = x[obs_order]
    y_obs = y_obs[obs_order]
    if prior_on_x:
        y_prior_mean = y_prior_mean[obs_order]
        y_prior_var = y_prior_var[obs_order]
    delta_eta_mean = delta_eta_mean[obs_order]
    delta_eta_std = delta_eta_std[obs_order]
    y_post_mean = y_post_mean[obs_order]
    y_post_var = y_post_var[obs_order]
    kappa_mean = np.asarray(kappa_mean)[:, obs_order]
    kappa_std = np.asarray(kappa_std)[:, obs_order]

    # If prior arrays are not on x_obs, recompute at x_obs using gp_eta + theta_fixed
    # if y_prior_mean.shape[0] != x.shape[0] or y_prior_var.shape[0] != x.shape[0]:
    #     Z_obs = np.hstack([x.reshape(-1, 1), np.tile(theta_fixed, (x.shape[0], 1))])
    #     y_prior_mean = gp_eta.predict(Z_obs)
    #     y_prior_var = np.diag(gp_eta.predict(Z_obs, return_cov=True)[1])

    #diagnostic print statements
    print("x.shape:", x.shape)
    print("y_prior_mean.shape:", y_prior_mean.shape)
    print("theta_fixed.shape:", np.shape(theta_fixed))

    # If prior arrays are not on x_obs, recompute at x_obs using gp_eta + theta_fixed
    if y_prior_mean.shape[0] != x.shape[0] or y_prior_var.shape[0] != x.shape[0]:

        theta_fixed = np.asarray(theta_fixed)

        if theta_fixed.ndim == 1:
            theta_block = np.tile(theta_fixed, (x.shape[0], 1))
        else:
            theta_block = theta_fixed

        Z_obs = np.hstack([
            x.reshape(-1, 1),
            theta_block
        ])

        y_prior_mean = gp_eta.predict(Z_obs)
        y_prior_var = np.diag(
            gp_eta.predict(Z_obs, return_cov=True)[1]
        )

    # Layout:
    # row 0 -> 2 columns (ax1, ax2)
    # rows 1..dtheta -> one full-width subplot each for delta_theta[k]
    # last row -> one full-width subplot for posterior
    n_rows = 2 + dtheta

    # add another row if holdout data is on
    if cross_validation_settings.get("holdout_data", False) and holdout_data is not None:
        n_rows += 1

    fig = plt.figure(figsize=(10, 3.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 2)

    # Top Left: Emulator Prior
    ax1 = fig.add_subplot(gs[0, 0])
    sim_x = np.asarray(x_sim).ravel()
    sim_y = np.asarray(y_sim).ravel()
    sim_order = np.argsort(sim_x)
    sim_x = sim_x[sim_order]
    sim_y = sim_y[sim_order]
    ax1.scatter(
        x, y_obs,
        label=plot_settings.get("observation_data_label", "Observed"),
        color="black",
    )
    ax1.plot(
        x, y_prior_mean,
        label=plot_settings.get("emulator_prior_label", r"$\mathcal{E}(x,\eta(x))$"),
        linestyle="--",
    )
    ax1.scatter(
        sim_x, sim_y,
        label=plot_settings.get("simulation_data_label", "Simulator Data"),
        color="blue", alpha=0.5,
    )
    ax1.set_title("Emulator Prior (No Discrepancy)")
    domain_label = _plot_label(
        plot_settings, "application_domain_labels", domain_index, "x"
    )
    ax1.set_xlabel(domain_label)
    ax1.set_ylabel(plot_settings.get("observation_value_label", "y"))
    ax1.legend()

    # Top Right: δ_eta contribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        x, delta_eta_mean,
        label=plot_settings.get("delta_eta_mean_label", r"$\delta_\eta$ mean"),
    )
    ax2.fill_between(
        x,
        delta_eta_mean - 2 * delta_eta_std,
        delta_eta_mean + 2 * delta_eta_std,
        alpha=0.3
    )
    if cross_validation_settings['conduct_cross_validation'] is True and "known_delta_form" in cross_validation_settings:
        known_delta_form = cross_validation_settings.get("known_delta_form")
        known_delta_form_params = cross_validation_settings.get("known_delta_form_params", {})
        form_config = known_delta_form_params.get(known_delta_form, {})
        
        if known_delta_form == "power_law" and form_config:
            coeff = form_config.get("coeff")
            exponent = form_config.get("exponent")
            if coeff is not None and exponent is not None:
                delta_known = coeff * np.power(x, exponent)
                ax2.plot(x, delta_known, label=r"$\delta_\eta^{\mathrm{true}}(x)$", linestyle="--", color="red")
        elif known_delta_form == "trig_funct" and form_config:
            trig_function = form_config.get("function")
            if trig_function == "sin":
                delta_known = -np.sin(x)
            elif trig_function == "cos":
                delta_known = np.cos(x)
            else:
                delta_known = np.zeros_like(x)
            ax2.plot(x, delta_known, label=r"$\delta_\eta^{\mathrm{true}}(x)$", linestyle="--", color="red")
        elif known_delta_form == "linear" and form_config:
            m = form_config.get("m", 0)
            b = form_config.get("b", 0)
            delta_known = m * x + b
            ax2.plot(x, delta_known, label=r"$\delta_\eta^{\mathrm{true}}(x)$", linestyle="--", color="red")
        elif known_delta_form == "polynomial" and form_config:
            coeffs = form_config.get("coeffs", [])
            delta_known = np.polyval(coeffs, x)
            ax2.plot(x, delta_known, label=r"$\delta_\eta^{\mathrm{true}}(x)$", linestyle="--", color="red")
        
    ax2.axhline(0, linestyle="--")
    ax2.set_title(r"Additive Discrepancy $\delta_{\eta}(x)$")
    ax2.set_xlabel(domain_label)
    ax2.set_ylabel(plot_settings.get("correction_label", "Correction"))
    ax2.legend()

    # Following subplots in a single column (full width)
    # Following subplots in a single column (full width)
    theta_settings = calibration_settings.get("theta_settings", {})
    theta_init = theta_settings.get("theta_initialization", "fixed")

    for k in range(dtheta):

        ax_k = fig.add_subplot(gs[1 + k, :])

        delta_theta_k_mean = kappa_mean[k, :]
        delta_theta_k_std  = kappa_std[k, :]
        parameter_label = _plot_label(
            plot_settings, "calibration_parameter_labels", k, rf"$\kappa_{{{k}}}$"
        )
        theta_name = (
            str(theta_labels[k])
            if theta_labels is not None and k < len(theta_labels)
            else str(k)
        )
        theta_label = (
            _format_plot_label(theta_name)
            if theta_name.startswith("$") or theta_name.startswith("\\")
            else rf"$\theta_{{{theta_name}}}$"
        )
        theta0_label = rf"$\theta^0_{{{theta_name}}}$"
        theta0_phys_label = rf"$\theta^0_{{{theta_name},\mathrm{{phys}}}}$"
        kappa_title = f"{parameter_label} ({theta_label})"

        theta_plot = np.asarray(
            theta_fixed if theta_fixed_plot is None else theta_fixed_plot
        )
        theta0_plot = (
            theta_plot[k]
            if theta_plot.ndim == 1
            else theta_plot[:, k]
        )
        theta0_phys = np.asarray(theta_fixed_phys[k]).ravel()
        theta0_plot_value = np.asarray(theta0_plot).ravel()

        # -------------------------------------------------------------
        # Plot calibrated parameter
        # -------------------------------------------------------------
        if theta_init == "fixed":

            y_mean = delta_theta_k_mean
            y_lower = y_mean - 2 * delta_theta_k_std
            y_upper = y_mean + 2 * delta_theta_k_std

            ax_k.plot(x, y_mean, label=f"{parameter_label}(x)")
            ax_k.fill_between(x, y_lower, y_upper, alpha=0.3)

            if theta_prior_min is not None and theta_prior_max is not None:
                theta0 = float(theta0_plot_value[0])
                prior_min = np.asarray(theta_prior_min).ravel()[k] - theta0
                prior_max = np.asarray(theta_prior_max).ravel()[k] - theta0
                ax_k.axhspan(
                    prior_min,
                    prior_max,
                    color="tab:orange",
                    alpha=0.12,
                    label=r"Theta prior range about $\theta^0$",
                )

            ax_k.axhline(
                0,
                linestyle="--",
                color="gray",
                label=(
                    rf"$\theta^0=0$ baseline "
                    rf"({theta0_phys_label}={float(theta0_phys[0]):.4g})"
                )
            )

            ax_k.set_title(f"Calibration discrepancy: {kappa_title}(x)")
            ax_k.set_ylabel(f"{parameter_label}(x)")

        elif theta_init == "compositional":

            theta_base = np.asarray(theta_plot[:, k] if theta_plot.ndim == 2 else theta_plot[k]).ravel()
            if theta_base.size == x.size:
                theta_base = theta_base[obs_order]

            y_mean = theta_base + delta_theta_k_mean
            y_lower = y_mean - 2 * delta_theta_k_std
            y_upper = y_mean + 2 * delta_theta_k_std

            ax_k.plot(
                x,
                theta_base,
                "k--",
                linewidth=2,
                label=rf"$\eta_{{{k}}}^{{base}}(x)$"
            )

            ax_k.plot(
                x,
                y_mean,
                linewidth=2,
                label=rf"$\eta_{{{k}}}(x)+\kappa_{{{k}}}(x)$"
            )

            ax_k.fill_between(x, y_lower, y_upper, alpha=0.3)

            if theta_prior_min is not None and theta_prior_max is not None:
                ax_k.axhspan(
                    np.asarray(theta_prior_min).ravel()[k],
                    np.asarray(theta_prior_max).ravel()[k],
                    color="tab:orange",
                    alpha=0.12,
                    label=r"Theta prior range",
                )

            calibrated_label = _plot_label(
                plot_settings, "calibration_parameter_labels", k, rf"$\eta_{{{k}}}$"
            )
            ax_k.axhline(
                float(theta0_plot_value[0])
                if theta0_plot_value.size == 1
                else np.nanmean(theta0_plot_value),
                color="black",
                linestyle="--",
                linewidth=1,
                label=rf"{theta0_phys_label}={np.nanmean(theta0_phys):.4g}",
            )
            ax_k.text(
                0.98,
                0.96,
                rf"{theta0_phys_label}={np.nanmean(theta0_phys):.4g}",
                transform=ax_k.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.85,
                },
            )
            ax_k.set_title(f"Calibrated model: {kappa_title}(x)")
            ax_k.set_ylabel(f"{calibrated_label}(x)")

        if theta_init == "fixed":
            ax_k.text(
                0.98,
                0.96,
                rf"{theta0_phys_label}={float(theta0_plot_value[0]):.4g}",
                transform=ax_k.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.85,
                },
            )

        # -------------------------------------------------------------
        # Cross-validation truth
        # -------------------------------------------------------------
        if cross_validation_settings["conduct_cross_validation"] and \
        "known_theta_form" in cross_validation_settings:

            known_theta_form = cross_validation_settings.get("known_theta_form")
            form_config = cross_validation_settings.get(
                "known_theta_form_params", {}
            ).get(known_theta_form, {})

            theta_known = None

            if known_theta_form == "constant":

                values = form_config.get("values", [])
                if k < len(values):
                    theta_known = values[k]

            elif known_theta_form == "trig_funct":

                funcs = form_config.get("functions", [])
                if k < len(funcs):
                    if funcs[k] == "sin":
                        theta_known = np.sin(x)
                    elif funcs[k] == "cos":
                        theta_known = np.cos(x)

            elif known_theta_form == "linear":

                m = form_config.get("m", 0)
                b = form_config.get("b", 0)
                theta_known = m * x + b

            if theta_known is not None:

                if theta_init == "fixed":

                    theta_known = theta_known - theta_fixed_phys[k]

                    if np.isscalar(theta_known):
                        ax_k.axhline(
                            theta_known,
                            color="red",
                            linestyle="--",
                            label=rf"$\kappa_{{{k}}}^{{true}}$"
                        )
                    else:
                        ax_k.plot(
                            x,
                            theta_known,
                            "r--",
                            label=rf"$\kappa_{{{k}}}^{{true}}(x)$"
                        )

                else:

                    if np.isscalar(theta_known):
                        ax_k.axhline(
                            theta_known,
                            color="red",
                            linestyle="--",
                            label=rf"$\kappa_{{{k}}}^{{true}}$"
                        )
                    else:
                        ax_k.plot(
                            x,
                            theta_known,
                            "r--",
                            label=rf"$\kappa_{{{k}}}^{{true}}(x)$"
                        )

        ax_k.set_xlabel(domain_label)
        ax_k.legend()

    # Bottom: Full Posterior (also full width)
    ax3 = fig.add_subplot(gs[1 + dtheta, :])
    ax3.scatter(
        x, y_obs,
        label=plot_settings.get("observation_data_label", "Observed"),
        color="black",
    )
    ax3.plot(
        x, y_post_mean,
        label=plot_settings.get("posterior_mean_label", "Posterior Mean"),
    )
    ax3.fill_between(
        x,
        y_post_mean - 2 * np.sqrt(y_post_var),
        y_post_mean + 2 * np.sqrt(y_post_var),
        alpha=0.3
    )
    ax3.set_title("Full Posterior Prediction")
    ax3.set_xlabel(domain_label)
    ax3.set_ylabel(plot_settings.get("observation_value_label", "y"))
    ax3.legend()

    # if holdout data is available, plot posterior predictions for holdout data
    if cross_validation_settings.get("holdout_data", False) and holdout_data is not None:
        x_holdout = np.asarray(holdout_data['x']).ravel()
        y_holdout = np.asarray(holdout_data['y']).ravel()
        y_holdout_post_mean = np.asarray(holdout_data['y_holdout_post_mean']).ravel()
        y_holdout_post_std = np.asarray(holdout_data['y_holdout_post_std']).ravel()
        y_holdout_post_var = np.asarray(holdout_data['y_holdout_post_var']).ravel()
        holdout_order = np.argsort(x_holdout)
        x_holdout = x_holdout[holdout_order]
        y_holdout = y_holdout[holdout_order]
        y_holdout_post_mean = y_holdout_post_mean[holdout_order]
        y_holdout_post_std = y_holdout_post_std[holdout_order]
        y_holdout_post_var = y_holdout_post_var[holdout_order]


        ax4 = fig.add_subplot(gs[2+dtheta,:])

        ax4.scatter(
            x_holdout,
            y_holdout,
            color="black",
            label=plot_settings.get("holdout_data_label", "Holdout data")
        )

        ax4.plot(
            x_holdout,
            y_holdout_post_mean,
            label=plot_settings.get("posterior_prediction_label", "Posterior prediction")
        )

        ax4.fill_between(
            x_holdout,
            y_holdout_post_mean - 2*np.sqrt(y_holdout_post_var),
            y_holdout_post_mean + 2*np.sqrt(y_holdout_post_var),
            alpha=0.3
        )

        ax4.set_title(plot_settings.get("holdout_title", "Holdout Posterior Prediction"))
        ax4.legend()

    plt.suptitle(suptitle)
    plt.tight_layout()
    # plt.show()
    plt_path = os.path.join(figure_path, figure_name)
    plt.savefig(plt_path, dpi=150)
    

    return()


def plot_pca_diagnostics(
    S,
    W,
    x_obs,
    kappa_theta_chain,
    theta_labels=None,
    max_modes=None,
    figure_name="pca_diagnostics.png",
    figure_directory=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    x_matrix = _domain_matrix(x_obs)
    if x_matrix.shape[1] > 1:
        for j in range(x_matrix.shape[1]):
            plot_pca_diagnostics(
                S, W, x_matrix[:, j], kappa_theta_chain, theta_labels,
                max_modes=max_modes,
                figure_name=f"{os.path.splitext(figure_name)[0]}_x{j}.png",
                figure_directory=figure_directory,
            )
        return

    dtheta = W.shape[0]
    Nmcmc = kappa_theta_chain.shape[0]

    if theta_labels is None:
        theta_labels = [f"theta_{i}" for i in range(dtheta)]

    if max_modes is None:
        max_modes = dtheta

    # --------------------------------------------------
    # Project kappa_theta → PCA space
    # --------------------------------------------------
    # shape: (Nmcmc, dtheta, No)
    kappa_z_chain = np.einsum('ij,tjk->tik', W.T, kappa_theta_chain)

    kappa_z_mean = kappa_z_chain.mean(axis=0)
    kappa_z_std  = kappa_z_chain.std(axis=0)

    # --------------------------------------------------
    # Explained variance
    # --------------------------------------------------
    explained = S**2 / np.sum(S**2)
    cum_explained = np.cumsum(explained)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    n_rows = 2 + max_modes
    fig = plt.figure(figsize=(12, 3 * n_rows))
    gs = fig.add_gridspec(n_rows, 2)

    # --------------------------------------------------
    # (1) Singular value spectrum
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(S, 'o-')
    ax1.set_title("Singular Value Spectrum")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular Value (log)")
    ax1.grid(True)

    # --------------------------------------------------
    # (2) Cumulative variance explained
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cum_explained, 'o-')
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Explained Variance")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True)

    # --------------------------------------------------
    # (3) PCA mode shapes (θ-space)
    # --------------------------------------------------
    for j in range(max_modes):
        ax = fig.add_subplot(gs[1 + j, 0])
        ax.bar(range(dtheta), W[:, j])
        ax.set_title(f"PCA Mode {j} (θ-space direction)")
        ax.set_xticks(range(dtheta))
        ax.set_xticklabels(theta_labels)
        ax.set_ylabel("Weight")
        ax.grid(True)

    # --------------------------------------------------
    # (4) κ projections in PCA space
    # --------------------------------------------------
    x_plot = x_obs.flatten()

    for j in range(max_modes):
        ax = fig.add_subplot(gs[1 + j, 1])

        mean_j = kappa_z_mean[j]
        std_j  = kappa_z_std[j]

        ax.plot(x_plot, mean_j, label=f"Mode {j} mean")
        ax.fill_between(
            x_plot,
            mean_j - 2 * std_j,
            mean_j + 2 * std_j,
            alpha=0.3
        )

        ax.axhline(0, linestyle="--")
        ax.set_title(f"PCA Mode {j} Contribution")
        ax.set_xlabel("x")
        ax.set_ylabel("kappa_z")
        ax.grid(True)

    plt.tight_layout()
    # plt.show()
    plt_path = os.path.join(figure_directory, figure_name)
    plt.savefig(plt_path, dpi=150)
    
    return()



def discrepancy_variance_decomposition(
    x_obs,
    y_obs,
    theta_fixed,
    kappa_theta_chain,
    delta_eta_chain,
    kappa_mean,
    delta_eta_mean,
    gp_eta,
    x_obs_input,
    Nsamp=200,
    plot=True,
    figure_path=None,
    save_name="variance_decomposition.png",
    suptitle="Posterior Predictive Variance Decomposition"
):
    """
    Decomposes posterior predictive variance into:
    
    1. Emulator (GP) uncertainty
    2. θ / κ uncertainty
    3. δ(x) uncertainty
    4. Total variance
    
    Also computes Law of Total Variance decomposition:
        Var(Y) = E[Var(Y|θ,δ)] + Var(E[Y|θ,δ])
    """

    x_matrix = _domain_matrix(x_obs)
    if x_matrix.shape[1] > 1:
        x_input_matrix = _domain_matrix(x_obs_input)
        outputs = []
        for j in range(x_matrix.shape[1]):
            outputs.append(discrepancy_variance_decomposition(
                x_matrix[:, j], y_obs, theta_fixed, kappa_theta_chain,
                delta_eta_chain, kappa_mean, delta_eta_mean, gp_eta,
                x_input_matrix, Nsamp=Nsamp, plot=plot,
                figure_path=figure_path,
                save_name=f"{os.path.splitext(save_name)[0]}_x{j}.png",
                suptitle=f"{suptitle} (domain coordinate x_{j})",
            ))
        return outputs

    No = len(x_matrix)
    x_eval_matrix = _domain_matrix(x_obs_input)
    if x_eval_matrix.shape[0] != No:
        raise ValueError(
            "x_obs_input must have one row per observation. "
            f"Received {x_eval_matrix.shape[0]} rows for {No} observations."
        )
    Nmcmc = kappa_theta_chain.shape[0]
    dtheta = kappa_theta_chain.shape[1]

    idx = np.random.choice(Nmcmc, Nsamp, replace=False)

    # storage
    var_total = np.zeros(No)
    var_emulator = np.zeros(No)
    var_kappa = np.zeros(No)
    var_delta = np.zeros(No)

    var_epistemic = np.zeros(No)
    var_aleatoric = np.zeros(No)

    for i in range(No):

        # ------------------------------------------------------------
        # 1. TOTAL posterior predictive samples
        # ------------------------------------------------------------
        y_samps = []

        theta_i = get_theta_at_obs(theta_fixed, i)

        for s in idx:
            kappa_s = kappa_theta_chain[s, :, i]
            theta_star = theta_i + kappa_s
            delta_s = delta_eta_chain[s, i]

            m, v = eta_predict(x_eval_matrix[i], theta_star, gp_eta)

            y_samps.append(m + delta_s)

        y_samps = np.array(y_samps)
        var_total[i] = np.var(y_samps)

        # ------------------------------------------------------------
        # 2. Emulator-only uncertainty (GP conditional variance)
        # ------------------------------------------------------------
        theta_star_mean = theta_i + kappa_mean[:, i]
        _, v_gp = eta_predict(x_eval_matrix[i], theta_star_mean, gp_eta)
        var_emulator[i] = v_gp

        # ------------------------------------------------------------
        # 3. κ (parameter discrepancy) uncertainty
        # ------------------------------------------------------------
        y_kappa = []
        for s in idx:
            theta_star = theta_i + kappa_theta_chain[s, :, i]
            m, _ = eta_predict(x_eval_matrix[i], theta_star, gp_eta)
            y_kappa.append(m + delta_eta_mean[i])

        var_kappa[i] = np.var(y_kappa)

        # ------------------------------------------------------------
        # 4. δ(x) uncertainty
        # ------------------------------------------------------------
        y_delta = []
        for s in idx:
            theta_star = theta_i + kappa_theta_chain[s, :, i]
            m, _ = eta_predict(x_eval_matrix[i], theta_star, gp_eta)
            y_delta.append(m + delta_eta_chain[s, i])

        var_delta[i] = np.var(y_delta)

        # ------------------------------------------------------------
        # 5. Law of Total Variance decomposition
        # ------------------------------------------------------------

        # epistemic: variation of conditional mean
        cond_means = []

        for s in idx:
            kappa_s = kappa_theta_chain[s, :, i]
            theta_star = theta_i + kappa_s
            delta_s = delta_eta_chain[s, i]

            m, _ = eta_predict(x_eval_matrix[i], theta_star, gp_eta)
            cond_means.append(m + delta_s)

        cond_means = np.array(cond_means)

        var_epistemic[i] = np.var(cond_means)

        # aleatoric: expected GP variance
        gp_vars = []

        for s in idx:
            kappa_s = kappa_theta_chain[s, :, i]
            theta_star = theta_i + kappa_s

            _, v = eta_predict(x_eval_matrix[i], theta_star, gp_eta)
            gp_vars.append(v)

        var_aleatoric[i] = np.mean(gp_vars)

    # ============================================================
    # Normalize contributions
    # ============================================================
    denom = var_total + 1e-12

    frac_emulator = var_emulator / denom
    frac_kappa = var_kappa / denom
    frac_delta = var_delta / denom

    # ============================================================
    # Plotting
    # ============================================================
    if plot:

        x = np.arange(No)

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # ---------------------------
        # (1) Relative contributions
        # ---------------------------
        ax[0].bar(x, frac_emulator, label="Emulator (GP)")
        ax[0].bar(x, frac_kappa, bottom=frac_emulator, label="θ / κ")
        ax[0].bar(x, frac_delta, bottom=frac_emulator + frac_kappa, label="δ(x)")

        ax[0].set_title("Posterior Variance Decomposition")
        ax[0].set_xlabel("Observation index")
        ax[0].set_ylabel("Fraction of variance")
        ax[0].legend()

        # ---------------------------
        # (2) Law of total variance
        # ---------------------------
        ax[1].plot(var_total, label="Total variance", lw=2)
        ax[1].plot(var_epistemic, label="Epistemic (κ + δ)", ls="--")
        ax[1].plot(var_aleatoric, label="Aleatoric (GP)", ls=":")
        ax[1].set_title("Law of Total Variance")
        ax[1].set_xlabel("Observation index")
        ax[1].set_ylabel("Variance")
        ax[1].legend()

        plt.tight_layout()

        plt.suptitle(suptitle)

        if figure_path is not None:
            plt.savefig(f"{figure_path}/{save_name}", dpi=300)

        # plt.show()

    return {
        "var_total": var_total,
        "var_emulator": var_emulator,
        "var_kappa": var_kappa,
        "var_delta": var_delta,
        "var_epistemic": var_epistemic,
        "var_aleatoric": var_aleatoric,
        "frac_emulator": frac_emulator,
        "frac_kappa": frac_kappa,
        "frac_delta": frac_delta
    }

def plot_relative_contributions(
    x,
    contrib,
    figure_path=None,
    save_name="relative_contributions.png",
    suptitle="Relative Discrepancy Contributions"
):
    x_matrix = _domain_matrix(x)
    if x_matrix.shape[1] > 1:
        for j in range(x_matrix.shape[1]):
            plot_relative_contributions(
                x_matrix[:, j], contrib, figure_path=figure_path,
                save_name=f"{os.path.splitext(save_name)[0]}_x{j}.png",
                suptitle=f"{suptitle} (domain coordinate x_{j})",
            )
        return
    x = x_matrix[:, 0]

    # ------------------------------------------------------------
    # Magnitudes
    # ------------------------------------------------------------
    kappa_mag = np.abs(contrib["kappa_mean"])
    eta_mag = np.abs(contrib["eta_mean"])

    total_mag = kappa_mag + eta_mag + 1e-12

    kappa_ratio = kappa_mag / total_mag
    eta_ratio = eta_mag / total_mag

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True
    )

    # ============================================================
    # TOP: actual contribution magnitudes
    # ============================================================
    ax[0].plot(
        x,
        contrib["kappa_mean"],
        label=r"$\kappa_\theta$ contribution"
    )

    ax[0].fill_between(
        x,
        contrib["kappa_mean"] - contrib["kappa_std"],
        contrib["kappa_mean"] + contrib["kappa_std"],
        alpha=0.3
    )

    ax[0].plot(
        x,
        contrib["eta_mean"],
        label=r"$\delta_\eta$ contribution"
    )

    ax[0].fill_between(
        x,
        contrib["eta_mean"] - contrib["eta_std"],
        contrib["eta_mean"] + contrib["eta_std"],
        alpha=0.3
    )

    ax[0].axhline(0, color='k', lw=0.8)

    ax[0].set_ylabel("Contribution")
    ax[0].set_title("Posterior Mean Contributions")
    ax[0].legend()

    # ============================================================
    # BOTTOM: relative dominance
    # ============================================================
    ax[1].plot(
        x,
        kappa_ratio,
        label=r"$\kappa_\theta$ dominance"
    )

    ax[1].plot(
        x,
        eta_ratio,
        label=r"$\delta_\eta$ dominance"
    )

    ax[1].axhline(
        0.5,
        linestyle="--",
        color="k",
        lw=1
    )

    ax[1].set_ylim([0, 1])

    ax[1].set_ylabel("Relative magnitude")
    ax[1].set_xlabel("x")
    ax[1].set_title("Relative Contribution Fractions")

    ax[1].legend()

    plt.suptitle(suptitle)

    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(
            f"{figure_path}/{save_name}",
            dpi=300,
            bbox_inches='tight'
        )

    # plt.show()

def eta_predict(x, theta_star, gp_eta):

    x = np.atleast_1d(np.asarray(x))
    theta_star = np.atleast_1d(np.asarray(theta_star))

    z = np.hstack([x, theta_star]).reshape(1, -1)

    m, s2 = gp_eta.predict(z, return_std=True)

    return m[0], s2[0]**2

# def plot_discrepancy_diagnostics_normalized(
#     x_obs,
#     y_obs,
#     x_sim,
#     y_sim,
#     y_prior_mean,
#     y_prior_var,
#     delta_eta_mean,
#     delta_eta_std,
#     y_post_mean,
#     y_post_var,
#     theta_fixed,
#     gp_eta,
#     delta_theta_chain,
#     idx,
#     dtheta,
#     figures_directory=None,
#     output_directory=None,
# ):
#     x = np.asarray(x_obs).ravel()
#     y_obs = np.asarray(y_obs).ravel()
#     y_prior_mean = np.asarray(y_prior_mean).ravel()
#     y_prior_var = np.asarray(y_prior_var).ravel()
#     delta_eta_mean = np.asarray(delta_eta_mean).ravel()
#     delta_eta_std = np.asarray(delta_eta_std).ravel()
#     y_post_mean = np.asarray(y_post_mean).ravel()
#     y_post_var = np.asarray(y_post_var).ravel()

#     # If prior arrays are not on x_obs, recompute at x_obs using gp_eta + theta_fixed
#     if y_prior_mean.shape[0] != x.shape[0] or y_prior_var.shape[0] != x.shape[0]:
#         Z_obs = np.hstack([x.reshape(-1, 1), np.tile(theta_fixed, (x.shape[0], 1))])
#         y_prior_mean = gp_eta.predict(Z_obs)
#         y_prior_var = np.diag(gp_eta.predict(Z_obs, return_cov=True)[1])

#     # Layout:
#     # row 0 -> 2 columns (ax1, ax2)
#     # rows 1..dtheta -> one full-width subplot each for delta_theta[k]
#     # last row -> one full-width subplot for posterior
#     n_rows = 2 + dtheta
#     fig = plt.figure(figsize=(10, 3.2 * n_rows))
#     gs = fig.add_gridspec(n_rows, 2)

#     # Top Left: Emulator Prior
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.scatter(x, y_obs, label="Observed", color="black")
#     ax1.plot(x, y_prior_mean, label="η(x, θ)", linestyle="--")
#     ax1.scatter(np.asarray(x_sim).ravel(), np.asarray(y_sim).ravel(),
#                 label="Simulator Data", color="blue", alpha=0.5)
#     ax1.fill_between(
#         x,
#         y_prior_mean - 2 * np.sqrt(y_prior_var),
#         y_prior_mean + 2 * np.sqrt(y_prior_var),
#         alpha=0.3
#     )
#     ax1.set_title("Emulator Prior (No Discrepancy)")
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")
#     ax1.legend()

#     # Top Right: δ_eta contribution
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.plot(x, delta_eta_mean, label="δ_eta mean")
#     ax2.fill_between(
#         x,
#         delta_eta_mean - 2 * delta_eta_std,
#         delta_eta_mean + 2 * delta_eta_std,
#         alpha=0.3
#     )
#     ax2.axhline(0, linestyle="--")
#     ax2.set_title(r"Additive Discrepancy $\delta_{\eta}(x)$")
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("Correction")
#     ax2.legend()

#     # Following subplots in a single column (full width)
#     for k in range(dtheta):
#         ax_k = fig.add_subplot(gs[1 + k, :])
#         delta_theta_k_mean = delta_theta_chain[idx, k, :].mean(axis=0)
#         delta_theta_k_std = delta_theta_chain[idx, k, :].std(axis=0)

#         ax_k.plot(x, delta_theta_k_mean, label=f"δ_theta[{k}] mean")
#         ax_k.fill_between(
#             x,
#             delta_theta_k_mean - 2 * delta_theta_k_std,
#             delta_theta_k_mean + 2 * delta_theta_k_std,
#             alpha=0.3
#         )
#         ax_k.axhline(0, linestyle="--")
#         ax_k.set_title(rf"Calibration Discrepancy $\delta_{{\theta_{k}}}(x)$")
#         ax_k.set_xlabel("x")
#         ax_k.set_ylabel("Correction")
#         ax_k.legend()

#     # Bottom: Full Posterior (also full width)
#     ax3 = fig.add_subplot(gs[1 + dtheta, :])
#     ax3.scatter(x, y_obs, label="Observed", color="black")
#     ax3.plot(x, y_post_mean, label="Posterior Mean")
#     ax3.fill_between(
#         x,
#         y_post_mean - 2 * np.sqrt(y_post_var),
#         y_post_mean + 2 * np.sqrt(y_post_var),
#         alpha=0.3
#     )
#     ax3.set_title("Full Posterior Prediction")
#     ax3.set_xlabel("x")
#     ax3.set_ylabel("y")
#     ax3.legend()

#     plt.tight_layout()
#     # plt.show()
#     plt_path = os.path.join(figures_directory, f"discrepancy_diagnostics_normalized.png")
#     plt.savefig(plt_path, dpi=150)
#     print(f"Saved discrepancy diagnostics figure to {plt_path}")
#     plt.close()

#     if output_directory is not None:
#         # Save the data used for plotting to a text file for later use in animations
#         output_data = np.column_stack([
#             x, y_obs, y_prior_mean, y_prior_var,
#             delta_eta_mean, delta_eta_std,
#             y_post_mean, y_post_var
#         ])
#         header = "x\ty_obs\ty_prior_mean\ty_prior_var\tdelta_eta_mean\tdelta_eta_std\ty_post_mean\ty_post_var"
#         np.savetxt(os.path.join(output_directory, "discrepancy_diagnostics_normalized.txt"), output_data, header=header, delimiter="\t")

#         return()
    

# def plot_discrepancy_diagnostics_realspace(
#     obs_data,
#     model_data,
#     model_normalized,
#     x_obs,
#     y_obs,
#     theta_fixed,
#     gp_eta,
#     y_prior_mean,
#     y_prior_var,
#     delta_eta_mean,
#     delta_eta_std,
#     y_post_mean,
#     y_post_var,
#     y_sim_phys,
#     y_sd,
#     delta_theta_chain,
#     idx,
#     dtheta,
#     figures_directory=None,
#     output_directory=None
# ):
#     # --- normalized arrays ---
#     x_n = np.asarray(x_obs).ravel()
#     y_obs_n = np.asarray(y_obs).ravel()
#     y_prior_mean_n = np.asarray(y_prior_mean).ravel()
#     y_prior_var_n = np.asarray(y_prior_var).ravel()
#     delta_eta_mean_n = np.asarray(delta_eta_mean).ravel()
#     delta_eta_std_n = np.asarray(delta_eta_std).ravel()
#     y_post_mean_n = np.asarray(y_post_mean).ravel()
#     y_post_var_n = np.asarray(y_post_var).ravel()

#     # If prior arrays are not evaluated at x_obs, recompute in normalized space
#     if y_prior_mean_n.shape[0] != x_n.shape[0] or y_prior_var_n.shape[0] != x_n.shape[0]:
#         Z_obs = np.hstack([x_n.reshape(-1, 1), np.tile(theta_fixed, (x_n.shape[0], 1))])
#         y_prior_mean_n = gp_eta.predict(Z_obs)
#         y_prior_var_n = np.diag(gp_eta.predict(Z_obs, return_cov=True)[1])

#     # --- infer columns ---
#     obs_x_col = [c for c in obs_data.columns if c.startswith("x_")][0]
#     obs_y_col = [c for c in obs_data.columns if c.startswith("xi_")][0]
#     model_y_col = [c for c in model_data.columns if c.startswith("y_")][0]

#     # --- x reverse-normalization (observation scaling) ---
#     x_min, x_max = obs_data[obs_x_col].min(), obs_data[obs_x_col].max()
#     x_phys = x_n * (x_max - x_min) + x_min

#     # --- y reverse-normalization ---
#     # observation/posterior/delta_eta are in observation-normalized y-space
#     obs_y_mean = obs_data[obs_y_col].mean()
#     obs_y_std = obs_data[obs_y_col].std()

#     # emulator prior is in model-normalized y-space
#     prior_mean_col = f"{model_y_col}_mean"
#     prior_std_col = f"{model_y_col}_std"
#     if model_normalized is not None and prior_mean_col in model_normalized.columns and prior_std_col in model_normalized.columns:        
#         prior_y_mean = float(model_normalized[prior_mean_col].iloc[0])
#         prior_y_std = float(model_normalized[prior_std_col].iloc[0])
#     else:
#         prior_y_mean = float(np.mean(y_sim_phys)) if "y_sim_phys" in globals() else obs_y_mean
#         prior_y_std = float(y_sd) if "y_sd" in globals() else obs_y_std

#     y_obs_phys = y_obs_n * obs_y_std + obs_y_mean
#     y_prior_mean_phys = y_prior_mean_n * prior_y_std + prior_y_mean
#     y_prior_var_phys = y_prior_var_n * (prior_y_std ** 2)
#     delta_eta_mean_phys = delta_eta_mean_n * obs_y_std
#     delta_eta_std_phys = delta_eta_std_n * obs_y_std
#     y_post_mean_phys = y_post_mean_n * obs_y_std + obs_y_mean
#     y_post_var_phys = y_post_var_n * (obs_y_std ** 2)

#     # simulator data in physical space
#     x_sim_phys_local = model_data[[c for c in model_data.columns if c.startswith("x_")][0]].values
#     y_sim_phys_local = model_data[model_y_col].values

#     # sort by x for clean plotting
#     order = np.argsort(x_phys)
#     x_phys = x_phys[order]
#     y_obs_phys = y_obs_phys[order]
#     y_prior_mean_phys = y_prior_mean_phys[order]
#     y_prior_var_phys = y_prior_var_phys[order]
#     delta_eta_mean_phys = delta_eta_mean_phys[order]
#     delta_eta_std_phys = delta_eta_std_phys[order]
#     y_post_mean_phys = y_post_mean_phys[order]
#     y_post_var_phys = y_post_var_phys[order]

#     # layout
#     n_rows = 2 + dtheta
#     fig = plt.figure(figsize=(10, 3.2 * n_rows))
#     gs = fig.add_gridspec(n_rows, 2)

#     # Top Left: Emulator Prior (physical)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.scatter(x_phys, y_obs_phys, label="Observed", color="black")
#     ax1.plot(x_phys, y_prior_mean_phys, label="η(x, θ)", linestyle="--")
#     ax1.scatter(x_sim_phys_local, y_sim_phys_local, label="Simulator Data", color="orange", alpha=0.5)
#     ax1.fill_between(
#         x_phys,
#         y_prior_mean_phys - 2 * np.sqrt(y_prior_var_phys),
#         y_prior_mean_phys + 2 * np.sqrt(y_prior_var_phys),
#         alpha=0.15
#     )
#     ax1.set_title("Emulator Prior")
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.legend()

   

#     # Top Right: delta_eta contribution (physical y-units)
#     ax2 = fig.add_subplot(gs[0, 1])
    

#     obs_x_col = [c for c in obs_data.columns if c.startswith("x_")][0]
#     # --- use MODEL normalization, not OBS ---
#     model_x_col = [c for c in model_data.columns if c.startswith("x_")][0]
#     model_x_min = model_data[model_x_col].min()
#     model_x_max = model_data[model_x_col].max()

#     x_norm_model = (x_phys - model_x_min) / (model_x_max - model_x_min)

#     # delta_eta_true_norm = x_norm_model ** 2

#     delta_eta_true_phys = 0.5 * np.sqrt(x_phys)

#     # --- correct transformation ---
#     # --- reconstruct δ_eta in physical space ---
#     delta_eta_mean_phys = y_post_mean_phys - y_prior_mean_phys

#     # uncertainty: combine posterior + prior (approx)
#     delta_eta_std_phys = np.sqrt(
#         y_post_var_phys + y_prior_var_phys
#     )


#     ax2.plot(x_phys, delta_eta_mean_phys, label=rf"$\delta_{{\eta}}(x)$ mean")
#     ax2.fill_between(
#         x_phys,
#         delta_eta_mean_phys - 2 * delta_eta_std_phys,
#         delta_eta_mean_phys + 2 * delta_eta_std_phys,
#         alpha=0.85
#     )

#     ax2.plot(
#         x_phys,
#         delta_eta_true_phys,
#         color="red",
#         linestyle="--",
#         linewidth=2,
#         label=r"True $\delta_{\eta}(x)=0.5\sqrt{x}$"
#     )

#     ax2.axhline(0, linestyle="--")
#     ax2.set_title(r"Additive Discrepancy $\delta_{\eta}(x)$")
#     ax2.set_xlabel('x')
#     ax2.set_ylabel("Correction")
#     ax2.legend()

#     # delta_theta subplots (physical theta-units)
#     theta_cols = [c for c in model_data.columns if c.startswith("theta_")]
#     use_idx = idx if "idx" in globals() else np.random.choice(delta_theta_chain.shape[0], size=min(200, delta_theta_chain.shape[0]), replace=False)
#     # add 'true values of theta for comparison
#     theta_0_true = 1.5
#     theta_1_true = 1.75


#     for k in range(dtheta):
#         plot_label=k+1
#         ax_k = fig.add_subplot(gs[1 + k, :])

#         dt_mean_n = delta_theta_chain[use_idx, k, :].mean(axis=0)
#         dt_std_n = delta_theta_chain[use_idx, k, :].std(axis=0)

#         theta_min = model_data[theta_cols[k]].min()
#         theta_max = model_data[theta_cols[k]].max()
#         theta_range = theta_max - theta_min

#         dt_mean_phys = (dt_mean_n * theta_range)[order]
#         dt_std_phys = (dt_std_n * theta_range)[order]

#         ax_k.plot(x_phys, dt_mean_phys, label=rf"$\kappa_{plot_label}(x)$ mean")
#         ax_k.fill_between(
#             x_phys,
#             dt_mean_phys - 2 * dt_std_phys,
#             dt_mean_phys + 2 * dt_std_phys,
#             alpha=0.3
#         )

#         # add horizontal line for true theta correction: (true value - mean value used)
#         theta_true_vals = [theta_0_true, theta_1_true]       
#         if k < len(theta_true_vals):
#             theta_col_k = theta_cols[k]
#             theta_mean_col = f"{theta_col_k}_mean"

#             # if model_normalized is not None and prior_mean_col in model_normalized.columns and prior_std_col in model_normalized.columns:                theta_mean_used = float(model_normalized[theta_mean_col].iloc[0])
#             #     # print(f"Using normalized mean for {theta_col_k}: {theta_mean_used:.3f}")
#             # elif "theta_fixed" in globals():
#             #     theta_mean_used = float(theta_fixed[k])
#             #     # print(f"Using fixed theta for {theta_col_k}: {theta_mean_used:.3f}")
#             # else:
#             #     theta_mean_used = float(model_data[theta_col_k].mean())
#             #     # print(f"Using data mean for {theta_col_k}: {theta_mean_used:.3f}")

#             # true_correction = float(theta_true_vals[k]) - theta_mean_used
#             # ax_k.axhline(
#             # true_correction,
#             # color="red",
#             # linestyle=":",
#             # label=rf"True $\kappa_{plot_label}$ correction"
#             # )

#         ax_k.axhline(0, linestyle="--")
#         ax_k.set_title(rf"Calibration Discrepancy $\kappa_{plot_label}(x)$")
#         ax_k.set_xlabel('x')
#         ax_k.set_ylabel(rf"$\kappa_{{{plot_label}}} - \bar{{\kappa}}_{{{plot_label}}}$")
#         ax_k.legend()

#     # Bottom: Full Posterior (physical) with uncertainty band
#     ax3 = fig.add_subplot(gs[1 + dtheta, :])
#     ax3.scatter(x_phys, y_obs_phys, label="Observed", color="black", s=18, zorder=3)
#     ax3.plot(x_phys, y_post_mean_phys, label="Posterior Mean", color="C1", linewidth=2, zorder=4)

#     # Coupled uncertainty from discrepancy terms: δη(x) + Σk δθk(x)
#     # (captures cross-correlation across theta components via joint chain samples)
#     theta_ranges = np.array(
#         [model_data[c].max() - model_data[c].min() for c in theta_cols],
#         dtype=float
#     )

#     # delta_theta_chain: (n_samples, dtheta, n_x) in normalized theta units
#     dt_samples_phys = delta_theta_chain[use_idx, :, :] * theta_ranges[None, :, None]
#     dt_sum_phys = dt_samples_phys.sum(axis=1)[:, order]  # (n_samples, n_x), sorted by x

#     # Sample delta_eta in physical y-units (no joint chain with delta_theta available)
#     rng = np.random.default_rng(0)
#     eta_samples_phys = rng.normal(
#         loc=delta_eta_mean_phys[None, :],
#         scale=np.clip(delta_eta_std_phys[None, :], 0.0, None),
#         size=dt_sum_phys.shape
#     )

#     # Total discrepancy samples and uncertainty envelope
#     discrepancy_samples_phys = eta_samples_phys + dt_sum_phys
#     coupled_std_phys = np.std(discrepancy_samples_phys, axis=0, ddof=1)

#     ax3.fill_between(
#         x_phys,
#         y_post_mean_phys - 2.0 * coupled_std_phys,
#         y_post_mean_phys + 2.0 * coupled_std_phys,
#         color="C1",
#         alpha=0.85,
#         label="Coupled Discrepancy Uncertainty (±2σ)",
#         zorder=2
#     )

#     # Include the prior as translucent points for reference
#     ax3.scatter(x_phys, y_prior_mean_phys, label="Emulator Prior Mean", color="C0", alpha=0.5, s=12, zorder=1)
    

#     ax3.set_title("Full Posterior Prediction")
#     ax3.set_xlabel('x')
#     ax3.set_ylabel('y')
#     ax3.legend()

#     plt.tight_layout()
#     # plt.show()
#     plt_path = os.path.join(figures_directory, f"discrepancy_diagnostics_realspace.png")
#     plt.savefig(plt_path, dpi=150)
#     print(f"Saved discrepancy diagnostics figure to {plt_path}")
#     plt.close()

#     if output_directory is not None:
#         # Save the data used for plotting to a text file for later use in animations
#         output_data = np.column_stack([
#             x_phys, y_obs_phys, y_prior_mean_phys, y_prior_var_phys,
#             delta_eta_mean_phys, delta_eta_std_phys,
#             y_post_mean_phys, y_post_var_phys
#         ])
#         header = "x_phys\ty_obs_phys\ty_prior_mean_phys\ty_prior_var_phys\tdelta_eta_mean_phys\tdelta_eta_std_phys\ty_post_mean_phys\ty_post_var_phys"
#         np.savetxt(os.path.join(output_directory, "discrepancy_diagnostics_realspace.txt"), output_data, header=header, delimiter="\t")

#         return()
