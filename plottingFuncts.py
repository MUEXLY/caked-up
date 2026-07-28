import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functs import get_theta_at_obs, eta_predict

def generate_rawData_figure(model_data, obs_data, figures_directory, figure_name="raw_data_theta_parameters.png", suptitle="Raw Data Colored by Theta Parameters"):
    """
    Create subplots for each theta parameter showing:
    - X-axis: application domain (x values)
    - Y-axis: calibration metric (y values)
    - Point color: theta parameter value
    - Observed data points
    """
    
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
    x_col = model_x_columns[0]
    
    # Create subplot grid for theta parameters
    n_theta = len(theta_columns)
    n_cols = int(np.ceil(np.sqrt(n_theta)))
    n_rows = int(np.ceil(n_theta / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, theta_col in enumerate(theta_columns):
        ax = axes[idx]
        
        # Scatter plot colored by theta parameter value
        scatter = ax.scatter(model_data[x_col], model_data[y_col], 
                            c=model_data[theta_col], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plot observed data
        if obs_data is not None and observation_y_columns and observation_y_columns[0] in obs_data.columns and observation_x_columns and observation_x_columns[0] in obs_data.columns:
            ax.scatter(obs_data[observation_x_columns[0]], obs_data[observation_y_columns[0]], color='red', label='Observed Data', alpha=0.8, s=30, edgecolors='black')
        
        ax.set_xlabel(f'{x_col}')
        ax.set_ylabel(f'{y_col}')
        ax.set_title(f'{y_col} colored by {theta_col}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(theta_col)
    
    # Hide unused subplots
    for idx in range(n_theta, len(axes)):
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

def plot_emulator_prior(y_prior_mean, y_prior_var, model_data, model_normalized, reverse_normalization, figures_directory=None, output_directory=None):
    
    # ---- normalized ----
    x_sim_norm = model_normalized[[col for col in model_normalized.columns if col.startswith('x')]].values.ravel()
    y_sim_norm = model_normalized[[col for col in model_normalized.columns if col.startswith('zeta')]].values.ravel()

    # ---- physical ----
    x_sim_phys = model_data[[col for col in model_data.columns if col.startswith('x')]].values.ravel()
    y_sim_phys = model_data[[col for col in model_data.columns if col.startswith('zeta')]].values.ravel()

    y_prior_std = np.sqrt(y_prior_var)

    # Reverse normalization
    y_col = list(reverse_normalization["y"].keys())[0]
    y_mu = reverse_normalization["y"][y_col]["mean"]
    y_sd = reverse_normalization["y"][y_col]["std"]

    y_prior_mean_phys = y_prior_mean * y_sd + y_mu
    y_prior_std_phys = y_prior_std * y_sd

    # Sort indices
    idx_norm = np.argsort(x_sim_norm)
    idx_phys = np.argsort(x_sim_phys)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))

    # Top: normalized
    axes[0].scatter(x_sim_norm, y_sim_norm, color="blue", alpha=0.6, label="Simulator Data")
    axes[0].plot(x_sim_norm[idx_norm], y_prior_mean[idx_norm], color="orange", label="Emulator Prior Mean")
    axes[0].fill_between(
        x_sim_norm[idx_norm],
        (y_prior_mean - 2 * y_prior_std)[idx_norm],
        (y_prior_mean + 2 * y_prior_std)[idx_norm],
        color="orange",
        alpha=0.2,
        label="Prior ±2σ",
    )
    axes[0].set_title("Normalized Space")
    axes[0].set_xlabel("x (normalized)")
    axes[0].set_ylabel("y (normalized)")
    axes[0].grid(True)
    axes[0].legend()

    # Bottom: physical
    axes[1].scatter(x_sim_phys, y_sim_phys, color="blue", alpha=0.6, label="Simulator Data")
    axes[1].plot(x_sim_phys[idx_phys], y_prior_mean_phys[idx_phys], color="green", label="Emulator Prior Mean")
    axes[1].fill_between(
        x_sim_phys[idx_phys],
        (y_prior_mean_phys - 2 * y_prior_std_phys)[idx_phys],
        (y_prior_mean_phys + 2 * y_prior_std_phys)[idx_phys],
        color="green",
        alpha=0.2,
        label="Prior ±2σ",
    )
    axes[1].set_title("Physical Space")
    axes[1].set_xlabel("x_x_sim")
    axes[1].set_ylabel("y_y_sim")
    axes[1].grid(True)
    axes[1].legend()

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
):
    x = np.asarray(x_obs).ravel()
    y_obs = np.asarray(y_obs).ravel()
    y_prior_mean = np.asarray(y_prior_mean).ravel()
    y_prior_var = np.asarray(y_prior_var).ravel()
    delta_eta_mean = np.asarray(delta_eta_mean).ravel()
    delta_eta_std = np.asarray(delta_eta_std).ravel()
    y_post_mean = np.asarray(y_post_mean).ravel()
    y_post_var = np.asarray(y_post_var).ravel()

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
    ax1.scatter(x, y_obs, label="Observed", color="black")
    ax1.plot(x, y_prior_mean, label="η(x, θ)", linestyle="--")
    ax1.scatter(np.asarray(x_sim).ravel(), np.asarray(y_sim).ravel(),
                label="Simulator Data", color="blue", alpha=0.5)
    # ax1.fill_between(
    #     x,
    #     y_prior_mean - 2 * np.sqrt(y_prior_var),
    #     y_prior_mean + 2 * np.sqrt(y_prior_var),
    #     alpha=0.3
    # )
    ax1.set_title("Emulator Prior (No Discrepancy)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # Top Right: δ_eta contribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, delta_eta_mean, label="$\delta_\eta$ mean")
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
                delta_known = np.sin(x)
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
    ax2.set_xlabel("x")
    ax2.set_ylabel("Correction")
    ax2.legend()

    # Following subplots in a single column (full width)
    # Following subplots in a single column (full width)
    theta_settings = calibration_settings.get("theta_settings", {})
    theta_init = theta_settings.get("theta_initialization", "fixed")

    for k in range(dtheta):

        ax_k = fig.add_subplot(gs[1 + k, :])

        delta_theta_k_mean = kappa_mean[k, :]
        delta_theta_k_std  = kappa_std[k, :]

        # -------------------------------------------------------------
        # Plot calibrated parameter
        # -------------------------------------------------------------
        if theta_init == "fixed":

            y_mean = delta_theta_k_mean
            y_lower = y_mean - 2 * delta_theta_k_std
            y_upper = y_mean + 2 * delta_theta_k_std

            ax_k.plot(x, y_mean, label=rf"$\kappa_{{{k}}}(x)$")
            ax_k.fill_between(x, y_lower, y_upper, alpha=0.3)

            ax_k.axhline(
                0,
                linestyle="--",
                color="gray",
                label=r"$\theta_0$"
            )

            ax_k.set_title(rf"Calibration discrepancy: $\kappa_{{{k}}}(x)$")
            ax_k.set_ylabel(rf"$\kappa_{{{k}}}(x)$")

        elif theta_init == "compositional":

            theta_base = theta_fixed_phys[k]

            y_mean = theta_base + delta_theta_k_mean
            y_lower = y_mean - 2 * delta_theta_k_std
            y_upper = y_mean + 2 * delta_theta_k_std

            ax_k.plot(
                x,
                theta_base,
                "k--",
                linewidth=2,
                label=rf"$\theta_{{{k}}}^{{base}}(x)$"
            )

            ax_k.plot(
                x,
                y_mean,
                linewidth=2,
                label=rf"$\theta_{{{k}}}(x)+\kappa_{{{k}}}(x)$"
            )

            ax_k.fill_between(x, y_lower, y_upper, alpha=0.3)

            ax_k.set_title(rf"Calibrated parameter: $\theta_{{{k}}}(x)$")
            ax_k.set_ylabel(rf"$\theta_{{{k}}}(x)$")

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
                            label=rf"$\theta_{{{k}}}^{{true}}$"
                        )
                    else:
                        ax_k.plot(
                            x,
                            theta_known,
                            "r--",
                            label=rf"$\theta_{{{k}}}^{{true}}(x)$"
                        )

        ax_k.set_xlabel("x")
        ax_k.legend()

    # Bottom: Full Posterior (also full width)
    ax3 = fig.add_subplot(gs[1 + dtheta, :])
    ax3.scatter(x, y_obs, label="Observed", color="black")
    ax3.plot(x, y_post_mean, label="Posterior Mean")
    ax3.fill_between(
        x,
        y_post_mean - 2 * np.sqrt(y_post_var),
        y_post_mean + 2 * np.sqrt(y_post_var),
        alpha=0.3
    )
    ax3.set_title("Full Posterior Prediction")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()

    # if holdout data is available, plot posterior predictions for holdout data
    if cross_validation_settings.get("holdout_data", False) and holdout_data is not None:
        x_holdout = holdout_data['x']
        y_holdout = holdout_data['y']
        y_holdout_post_mean = holdout_data['y_holdout_post_mean']
        y_holdout_post_std = holdout_data['y_holdout_post_std']
        y_holdout_post_var = holdout_data['y_holdout_post_var']


        ax4 = fig.add_subplot(gs[2+dtheta,:])

        ax4.scatter(
            x_holdout,
            y_holdout,
            color="black",
            label="Holdout data"
        )

        ax4.plot(
            x_holdout,
            y_holdout_post_mean,
            label="Posterior prediction"
        )

        ax4.fill_between(
            x_holdout,
            y_holdout_post_mean - 2*np.sqrt(y_holdout_post_var),
            y_holdout_post_mean + 2*np.sqrt(y_holdout_post_var),
            alpha=0.3
        )

        ax4.set_title(
            "Holdout Posterior Prediction"
        )
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

    No = len(x_obs)
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

            m, v = eta_predict(x_obs[i], theta_star, gp_eta)

            y_samps.append(m + delta_s)

        y_samps = np.array(y_samps)
        var_total[i] = np.var(y_samps)

        # ------------------------------------------------------------
        # 2. Emulator-only uncertainty (GP conditional variance)
        # ------------------------------------------------------------
        theta_star_mean = theta_i + kappa_mean[:, i]
        _, v_gp = eta_predict(x_obs[i], theta_star_mean, gp_eta)
        var_emulator[i] = v_gp

        # ------------------------------------------------------------
        # 3. κ (parameter discrepancy) uncertainty
        # ------------------------------------------------------------
        y_kappa = []
        for s in idx:
            theta_star = theta_i + kappa_theta_chain[s, :, i]
            m, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            y_kappa.append(m + delta_eta_mean[i])

        var_kappa[i] = np.var(y_kappa)

        # ------------------------------------------------------------
        # 4. δ(x) uncertainty
        # ------------------------------------------------------------
        y_delta = []
        for s in idx:
            theta_star = theta_i + kappa_theta_chain[s, :, i]
            m, _ = eta_predict(x_obs[i], theta_star, gp_eta)
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

            m, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            cond_means.append(m + delta_s)

        cond_means = np.array(cond_means)

        var_epistemic[i] = np.var(cond_means)

        # aleatoric: expected GP variance
        gp_vars = []

        for s in idx:
            kappa_s = kappa_theta_chain[s, :, i]
            theta_star = theta_i + kappa_s

            _, v = eta_predict(x_obs[i], theta_star, gp_eta)
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

