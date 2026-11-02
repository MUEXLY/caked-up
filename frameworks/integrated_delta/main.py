import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from plottingFuncts import *
from functs import *
from functs import _prior_for_kappa


def _math_label(label):
    """Format a configured LaTeX command for Matplotlib mathtext."""
    label = str(label)
    if label.startswith("$") and label.endswith("$"):
        return label
    if label.startswith("\\"):
        return f"${label}$"
    return label


def main(config_path="config.json"):

    # Read in data for processing
    with open(config_path, 'r') as f:
            config = json.load(f)

    calibration_settings = config['calibration_settings']
    input_settings = config['input_settings']
    output_settings = config['output_settings']
    live_monitor_enabled = bool(output_settings.get("live_monitor", False))
    live_monitor_path = output_settings.get("live_monitor_path")
    live_monitor_interval = max(int(output_settings.get("live_monitor_interval", 25)), 1)
    cross_validation_settings = config['cross_validation_settings']
    label_settings = config['label_settings']
    observation_set_label = label_settings.get(
        "observation_set_label", "Observation"
    )
    simulation_set_label = label_settings.get(
        "simulation_set_label", "Simulation"
    )
    results_options = output_settings['results_options']
    results_directory = results_options['results_path']
    validation_options = output_settings.get('validation_options', {})

    observation_path = input_settings['input_observations_path']
    model_path = input_settings['input_model_path']
    file_delimiter = input_settings['input_delimiter']

    model_data = pd.DataFrame()

    #populate x values from appDomain.txt
    # The first line of appDomain.txt contains the header names
    # The subsequent lines contain the x values
    with open(f'{model_path}appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    # #populate theta values from thetaVals.txt
    # The first line of thetaVals.txt contains the header names
    # The subsequent lines contain the theta values
    with open(f'{model_path}thetaVals.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'theta_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    #populate y values from modelPredictions.txt
    # The first line of modelPredictions.txt contains the header names
    # The subsequent lines contain the y values
    # If there are blank lines at the end of the file, ignore
    with open(f'{model_path}modelPredictions.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'y_{header}'
            model_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]


    # Create a dataframe to store the observation data
    # The DataFrame can have any number of rows and any number of columns
    # The columns will be identified by header names x_{label}, y_{label}, etc.
    # x will be read from obsData/appDomain.txt
    # y will be read from obsData/observationData.txt
    # Each row will correspond to a different observation
    obs_data = pd.DataFrame()
    with open(f'{observation_path}appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            obs_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]

    with open(f'{observation_path}observationData.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(file_delimiter)
        for i, header in enumerate(headers):
            col_name = f'xi_{header}'
            obs_data[col_name] = [float(line.strip().split(file_delimiter)[i]) for line in lines[1:]]


    model_domain = model_data[[col for col in model_data.columns if col.startswith('x_')]]
    calib_params = model_data[[col for col in model_data.columns if col.startswith('theta_')]]

    configured_theta_labels = label_settings.get("theta_labels", {})
    theta_labels = []
    for index, column in enumerate(calib_params.columns):
        parameter_name = column.replace("theta_", "")
        configured_label = next(
            (
                configured_theta_labels[key]
                for key in (
                    column,
                    parameter_name,
                    f"theta{index + 1}",
                    f"theta_{index + 1}",
                )
                if key in configured_theta_labels
            ),
            parameter_name,
        )
        theta_labels.append(configured_label)
    configured_app_labels = label_settings.get("application_domain_labels", {})
    app_labels = []
    for index, column in enumerate(model_domain.columns):
        domain_name = column.replace("x_", "")
        app_labels.append(
            next(
                (
                    configured_app_labels[key]
                    for key in (
                        column,
                        domain_name,
                        f"domain{index + 1}",
                        f"domain_{index + 1}",
                    )
                    if key in configured_app_labels
                ),
                domain_name,
            )
        )
    configured_observation_labels = label_settings.get("observation_labels", {})
    observation_labels = []
    for index, column in enumerate(
        column for column in obs_data.columns if column.startswith("xi_")
    ):
        observation_name = column.replace("xi_", "")
        observation_labels.append(
            next(
                (
                    configured_observation_labels[key]
                    for key in (
                        column,
                        observation_name,
                        f"obs{index + 1}",
                        f"obs_{index + 1}",
                    )
                    if key in configured_observation_labels
                ),
                observation_name,
            )
        )
    configured_kappa_labels = label_settings.get("kappa_labels", {})
    kappa_labels = [
        next(
            (
                configured_kappa_labels[key]
                for key in (
                    f"kappa{index + 1}",
                    f"kappa_{index + 1}",
                    f"delta{index + 1}",
                )
                if key in configured_kappa_labels
            ),
            "\\delta_{\\theta}",
        )
        for index in range(len(calib_params.columns))
    ]

    # create standardized/normalized dataframes
    model_normalized = model_data.copy()
    observation_normalized = obs_data.copy()

    # transform the simulator inputs (x,theta) to the unit hypercube [0,1] using min-max scaling
    for col in model_data.columns:
        if col.startswith('x_') or col.startswith('theta_'):
            min_val = model_data[col].min()
            max_val = model_data[col].max()
            model_normalized[col] = (model_data[col] - min_val) / (max_val - min_val) 

    # transform the simulator outputs to be zero mean and unit variance
    for col in model_data.columns:
        if col.startswith('y_'):
            mean_val = model_data[col].mean()
            std_val = model_data[col].std()
            model_normalized[col] = (model_data[col] - mean_val) / std_val
            
    # repeat the procedure for the observation data
    for col in obs_data.columns:
        if col.startswith('x_'):
            min_val = obs_data[col].min()
            max_val = obs_data[col].max()
            observation_normalized[col] = (obs_data[col] - min_val) / (max_val - min_val)
            
    for col in obs_data.columns:
        if col.startswith('xi_'):
            mean_val = obs_data[col].mean()
            std_val = obs_data[col].std()
            observation_normalized[col] = (obs_data[col] - mean_val) / std_val
        

    # Create figures directory if it doesn't exist
    figures_directory = output_settings['figures_path']
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    figure_options = output_settings['figure_options']
    
    if figure_options['data_priors']:
        generate_rawData_figure(
            model_data,
            obs_data,
            figures_directory,
            observation_set_label=observation_set_label,
            simulation_set_label=simulation_set_label,
            theta_labels=theta_labels,
            application_domain_label=app_labels[0],
            observation_label=observation_labels[0],
        )

    # Simulator inputs
    x_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]]
    theta_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('theta_')]]
    y_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('y_')]]

    # Joint emulator input z = (x, theta)
    Z_sim = np.hstack([x_sim, theta_sim])



    d = Z_sim.shape[1]  # dimension = 1 + dtheta

    kernel_eta = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2)
    )

    allow_singular_covariance = calibration_settings.get("allow_singular_covariance", False)

    gp_eta = GaussianProcessRegressor(
        kernel=kernel_eta,
        alpha=1e-6,           # nugget for stability
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp_eta.fit(Z_sim, y_sim)

    x_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('x_')]].values
    y_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('xi_')]].values.flatten()

    dtheta = theta_sim.shape[1]
    No = len(x_obs)
    kappa_hyperpriors = build_kappa_hyperpriors(
        validation_options.get("kappa_hyperpriors"),
        dtheta,
    )

    # ============================================================
    # 2. Train probabilistic emulator GP η(x,θ)
    # ============================================================

    Z_sim = np.hstack([x_sim, theta_sim])

    kernel_eta = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(Z_sim.shape[1]),
        length_scale_bounds=(1e-2, 1e2)
    )

    gp_eta = GaussianProcessRegressor(
        kernel=kernel_eta,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp_eta.fit(Z_sim, y_sim)

    print("\nTrained emulator kernel:")
    print(gp_eta.kernel_)

    if figure_options.get("emulator_sensitivity_pca", False):
        plot_emulator_sensitivity_pca(
            gp_eta,
            Z_sim,
            theta_indices=np.arange(x_sim.shape[1], Z_sim.shape[1]),
            theta_labels=theta_labels,
            figures_directory=figures_directory,
            filename=figure_options.get(
                "emulator_sensitivity_pca_filename",
                "emulator_sensitivity_pca.png",
            ),
        )

    if figure_options.get("emulator_sensitivity_boxplot", False):
        plot_emulator_sensitivity_boxplot(
            gp_eta,
            Z_sim,
            theta_indices=np.arange(x_sim.shape[1], Z_sim.shape[1]),
            theta_labels=theta_labels,
            figures_directory=figures_directory,
            filename=figure_options.get(
                "emulator_sensitivity_boxplot_filename",
                "emulator_sensitivity_boxplot.png",
            ),
        )

    # ============================================================
    # 3. Initialize θ (fixed plug-in estimate)
    # ============================================================
    if calibration_settings['theta_initialization'] == 'fixed':
        theta_fixed = theta_sim.mean(axis=0)

        print("\nFixed theta used:")
        print(theta_fixed)
    else:
        print("Only fixed theta initialization is currently implemented.")

    # ============================================================
    # 4. Initialize discrepancy δₖ(x) latent vectors
    # ============================================================

    delta = np.zeros((dtheta, No))

    # Initial hyperparameters for each δₖ
    ell_delta = 0.3 * np.ones(dtheta)
    var_delta = 0.05 * np.ones(dtheta)

    # Initial noise variance
    sigma2 = 0.02**2

    Nmcmc=calibration_settings["N_mcmc"]
    mh_scale_delta=calibration_settings["mh_scale_delta_prior"]

    mh_scales = {
        "log_ell_delta": 0.1,
        "log_var_delta": 0.1
    }

    a_sigma = 2.0
    b_sigma = 0.5


    # Storage
    delta_chain = np.zeros((Nmcmc, dtheta, No))
    sigma2_chain = np.zeros(Nmcmc)
    live_hyperparameter_specs = []
    live_hyperparameter_values = []
    for k in range(dtheta):
        prior_ell, prior_var = _prior_for_kappa(kappa_hyperpriors, k)
        live_hyperparameter_specs.extend([
            {"label": f"kappa_{k + 1}", "name": r"$\rho$", "kind": "lognormal", "prior": prior_ell},
            {"label": f"kappa_{k + 1}", "name": r"$\sigma$", "kind": "lognormal", "prior": prior_var},
        ])
        live_hyperparameter_values.extend([ell_delta[k], var_delta[k]])
    live_hyperparameter_stride = max(
        1, int(figure_options.get("hyperprior_density_animation_stride", 100))
    )

    accept_delta = np.zeros(dtheta)

    # --- acceptance trace storage ---
    accept_trace = [[] for _ in range(dtheta)]

    # --- MH scaling adaptation settings ---
    burnin = calibration_settings["burn_in"]
    adapt_interval = 50
    target_accept = 0.35
    mh_scale_delta = calibration_settings["mh_scale_delta_prior"]
    # mh_scale_delta = 0.5  # initial guess

    # store scale history (optional)
    mh_scale_trace = []



    # ============================================================
    # 6. Run Embedded Discrepancy Sampler
    # ============================================================

    print("\nRunning embedded δ-MH sampler...\n")

    for it in range(Nmcmc):

        # ---- update δₖ fields ----
        for k in range(dtheta):
            delta, acc = mh_update_delta_k(
                k,
                delta,
                theta_fixed,
                ell_delta[k],
                var_delta[k],
                x_obs,
                y_obs,
                gp_eta,
                sigma2,
                mh_scale_delta
            )
            accept_delta[k] += acc
            accept_trace[k].append(acc)
            
        # ---- adapt mh_scale_delta during burn-in ----
        if it < burnin and (it + 1) % adapt_interval == 0:

            # acceptance rate over last adapt_interval steps
            recent_accept = accept_delta / (it + 1)

            # average across delta dimensions
            mean_accept = np.mean(recent_accept)

            # log-scale update (stable + standard)
            gamma = 0.05  # adaptation speed

            mh_scale_delta *= np.exp(gamma * (mean_accept - target_accept))

            print(f"[ADAPT] Iter {it+1}: mean accept = {mean_accept:.3f}")
            print(f"[ADAPT] Updated mh_scale_delta = {mh_scale_delta:.4f}")


        # ---- update δ hyperparameters ----
        for k in range(dtheta):
            prior_ell, prior_var = _prior_for_kappa(kappa_hyperpriors, k)
            ell_delta[k], var_delta[k], _ = mh_update_delta_hyperparams(
                delta[k],
                ell_delta[k],
                var_delta[k],
                x_obs,
                prior_ell,
                prior_var,
                mh_scales,
                allow_singular_covariance=allow_singular_covariance
            )

        # ---- Gibbs update σ² ----
        sigma2 = gibbs_sigma2(
            y_obs,
            x_obs,
            theta_fixed,
            delta,
            gp_eta,
            a_sigma,
            b_sigma
        )

        # ---- store ----
        delta_chain[it] = delta
        sigma2_chain[it] = sigma2

        if (
            figure_options.get(
                "live_hyperparameter_updates",
                figure_options.get("hyperprior_density_animation", False),
            )
            and live_hyperparameter_specs
            and ((it + 1) % live_hyperparameter_stride == 0 or it + 1 == Nmcmc)
        ):
            live_hyperparameter_values = [
                value
                for k in range(dtheta)
                for value in (ell_delta[k], var_delta[k])
            ]
            try:
                plot_hyperparameter_density_state(
                    live_hyperparameter_values,
                    live_hyperparameter_specs,
                    figures_directory,
                    filename="hyperparameter_density_live.png",
                    iteration=it + 1,
                )
            except (OSError, ValueError, KeyError, IndexError) as exc:
                print(f"Could not update live hyperparameter density: {exc}")

        if (
            live_monitor_enabled
            and live_monitor_path
            and ((it + 1) % live_monitor_interval == 0 or it + 1 == Nmcmc)
        ):
            live_indices = np.linspace(0, No - 1, min(No, 500), dtype=int)
            theta_current = np.tile(theta_fixed, (No, 1))[live_indices] + delta.T[live_indices]
            live_inputs = np.hstack([x_obs[live_indices], theta_current])
            live_mean, live_std = gp_eta.predict(live_inputs, return_std=True)
            live_uncertainty = np.sqrt(live_std**2 + sigma2)
            live_x = (
                x_obs[live_indices, 0]
                if x_obs.shape[1] == 1
                else live_indices.astype(float)
            )
            live_payload = {
                "iteration": int(it + 1),
                "total_iterations": int(Nmcmc),
                "x": live_x.tolist(),
                "observed": np.asarray(y_obs)[live_indices].reshape(-1).tolist(),
                "predicted": np.asarray(live_mean)[..., 0].reshape(-1).tolist()
                if np.asarray(live_mean).ndim > 1
                else np.asarray(live_mean).reshape(-1).tolist(),
                "uncertainty": np.asarray(live_uncertainty).reshape(-1).tolist(),
            }
            live_temp_path = f"{live_monitor_path}.tmp"
            with open(live_temp_path, "w") as stream:
                json.dump(live_payload, stream)
            os.replace(live_temp_path, live_monitor_path)

        # ---- diagnostics ----
        if (it+1) % 50 == 0:
            print(f"Iter {it+1}/{Nmcmc}")
            print(" sigma2 =", sigma2)
            print(" delta acceptance rates:",
                accept_delta / (it+1))
            print(" ell_delta:", ell_delta)
            print(" var_delta:", var_delta)

            print("--------------------------------------------------")


    print("\nSampler complete.")


    # ============================================================
    # 7. Posterior Predictive Check
    # ============================================================

    print("\nBuilding posterior predictive mean + uncertainty...\n")

    y_post_mean = np.zeros(No)
    y_post_var  = np.zeros(No)

    Nsamp = calibration_settings['posterior_predictive_samples']
    idx = np.random.choice(Nmcmc, Nsamp, replace=False)

    y_preds = np.zeros((Nsamp, No))

    for s_i, s in enumerate(idx):

        for i in range(No):

            delta_s = delta_chain[s, :, i]
            theta_star = theta_fixed + delta_s

            m_i, s2_i = eta_predict(x_obs[i], theta_star, gp_eta)

            # Draw one posterior predictive sample from emulator
            y_preds[s_i, i] = np.random.normal(m_i, np.sqrt(s2_i))
            

    # Posterior mean and variance
    y_post_mean = y_preds.mean(axis=0)
    y_post_std  = y_preds.std(axis=0)
    # print prediction mean and 95% credible interval
    print(f"x_obs[{i}] = {x_obs[i,0]:.3f}, y_obs = {y_obs[i]:.3f}, post pred mean = {y_post_mean[i]:.3f}, 95% CI = [{y_post_mean[i] - 2*np.sqrt(y_post_var[i]):.3f}, {y_post_mean[i] + 2*np.sqrt(y_post_var[i]):.3f}]")

    # ============================================================
    # 8. Convert back to physical units for plotting
    # ============================================================


    x_obs_col = [col for col in obs_data.columns if col.startswith('x_')][0]
    y_obs_col = [col for col in obs_data.columns if col.startswith('xi_')][0]
    x_sim_col = [col for col in model_data.columns if col.startswith('x_')][0]
    y_sim_col = [col for col in model_data.columns if col.startswith('y_')][0]
    theta_cols = [col for col in model_data.columns if col.startswith('theta_')]

    x_obs_min = obs_data[x_obs_col].min()
    x_obs_max = obs_data[x_obs_col].max()

    x_sim_min = model_data[x_sim_col].min()
    x_sim_max = model_data[x_sim_col].max()

    x_obs_phys = x_obs * (x_obs_max - x_obs_min) + x_obs_min
    x_sim_phys = x_sim * (x_sim_max - x_sim_min) + x_sim_min


    y_obs_mean = obs_data[y_obs_col].mean()
    y_obs_std = obs_data[y_obs_col].std()

    y_sim_mean = model_data[y_sim_col].mean()
    y_sim_std = model_data[y_sim_col].std()

    y_obs_phys = y_obs * y_obs_std + y_obs_mean
    y_sim_phys = y_sim * y_sim_std + y_sim_mean

    # denormalize posterior predictive (use obs scale to compare with obs)
    y_post_mean_phys = y_post_mean * y_obs_std + y_obs_mean
    y_post_std_phys = y_post_std * y_obs_std

    theta_min = model_data[theta_cols].min().values
    theta_max = model_data[theta_cols].max().values
    theta_range = theta_max - theta_min

    theta_fixed_phys = np.asarray(
    theta_fixed * theta_range + theta_min
    )


    delta_chain_phys = delta_chain * theta_range[:, None]

    # Sort x for smooth plotting (physical)
    x_sorted_idx = np.argsort(x_obs_phys.ravel())
    x_sorted_phys = x_obs_phys.ravel()[x_sorted_idx]

    delta_mean_phys = delta_chain_phys.mean(axis=0)
    delta_std_phys = delta_chain_phys.std(axis=0)

    # Posterior theta prediction in physical units
    theta_post_mean_phys = (
        theta_fixed_phys[:, None]
        + delta_mean_phys
    )

    theta_post_std_phys = delta_std_phys

    external_validation = validation_options.get("external_validation", {})
    if external_validation.get("enabled", False):
        external_data = pd.read_csv(external_validation["csv_path"])
        parameter_mapping = external_validation["parameter_columns"]
        missing_columns = [
            column
            for column in [
                external_validation["x_column"],
                external_validation["y_column"],
                *parameter_mapping.values(),
            ]
            if column not in external_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"External validation CSV is missing columns: {missing_columns}"
            )

        external_x_phys = external_data[external_validation["x_column"]].to_numpy(
            dtype=float
        )
        external_y_phys = (
            external_data[external_validation["y_column"]].to_numpy(dtype=float)
            * external_validation.get("y_scale", 1.0)
        )
        external_theta_phys = external_data[
            list(parameter_mapping.values())
        ].to_numpy(dtype=float)
        external_theta_normalized = (
            external_theta_phys - theta_min
        ) / theta_range
        external_x_normalized = (
            external_x_phys - x_sim_min
        ) / (x_sim_max - x_sim_min)
        external_correction_phys = np.column_stack(
            [
                np.interp(
                    external_x_phys,
                    x_obs_phys.ravel(),
                    delta_mean_phys[k],
                )
                for k in range(dtheta)
            ]
        )
        external_correction_normalized = (
            external_correction_phys / theta_range
        )
        external_predictions = []
        for row in range(len(external_data)):
            theta_input = external_theta_normalized[row] + (
                external_correction_normalized[row]
            )
            mean, _ = eta_predict(
                np.atleast_1d(external_x_normalized[row]),
                theta_input,
                gp_eta,
            )
            external_predictions.append(mean * y_sim_std + y_sim_mean)

        plot_external_validation(
            x_sim_phys,
            y_sim_phys,
            x_obs_phys,
            y_obs_phys,
            x_obs_phys,
            y_post_mean_phys,
            y_post_std_phys,
            external_x_phys,
            external_y_phys,
            external_predictions,
            theta_labels,
            external_theta_phys,
            external_correction_phys,
            observation_set_label=observation_set_label,
            simulation_set_label=simulation_set_label,
            application_domain_label=app_labels[0],
            observation_label=observation_labels[0],
            figures_directory=figures_directory,
            filename=external_validation.get(
                "figure_filename", "external_validation.png"
            ),
            prior_parameters=model_data[theta_cols].to_numpy(),
        )
        external_results_path = os.path.join(
            results_directory, "external_validation_predictions.csv"
        )
        os.makedirs(results_directory, exist_ok=True)
        external_output = external_data.copy()
        external_output["posterior_predictive"] = external_predictions
        for index, label in enumerate(theta_labels):
            external_output[f"{label}_learned_correction"] = (
                external_correction_phys[:, index]
            )
        external_output.to_csv(external_results_path, index=False)
        print(f"Saved external validation predictions to {external_results_path}")

    if figure_options.get('prior_posterior_density', False):
        posterior_start = min(max(burnin, 0), Nmcmc - 1)
        posterior_theta_samples = (
            theta_fixed_phys[None, :, None]
            + delta_chain_phys[posterior_start:]
        ).transpose(0, 2, 1).reshape(-1, dtheta)
        plot_prior_posterior_densities(
            model_data[theta_cols].to_numpy(),
            posterior_theta_samples,
            theta_labels,
            figures_directory,
        )

    if figure_options.get('holdout_sensitivity', False):
        holdout_metrics = holdout_sensitivity_refit(
            x_obs,
            y_obs,
            x_obs_phys,
            y_obs_phys,
            theta_fixed,
            gp_eta,
            dtheta,
            holdout_fractions=validation_options.get(
                'holdout_fractions', (0.1, 0.2, 0.3)
            ),
            n_repeats=validation_options.get('holdout_repeats', 20),
            n_mcmc=validation_options.get(
                'holdout_n_mcmc', max(1000, calibration_settings["N_mcmc"] // 5)
            ),
            burn_in=validation_options.get(
                'holdout_burn_in', max(250, burnin // 2)
            ),
            mh_scale_delta=calibration_settings["mh_scale_delta_prior"],
            allow_singular_covariance=allow_singular_covariance,
            random_state=validation_options.get('random_seed', 0),
            kappa_hyperpriors=kappa_hyperpriors,
        )
        full_scale = np.ptp(y_obs_phys)
        full_scale = full_scale if full_scale > 0 else 1.0
        holdout_metrics["full_nrmse"] = float(
            np.sqrt(np.mean((y_obs_phys - y_post_mean_phys) ** 2))
            / full_scale
        )
        for result in holdout_metrics["holdout_results"].values():
            result["sensitivity"] = result["nrmse"] - holdout_metrics["full_nrmse"]
        if figure_options.get('holdout_predictions', False):
            plot_holdout_predictions(
                holdout_metrics['cases'],
                theta_labels,
                figures_directory,
                full_x=x_obs_phys,
                full_y_true=y_obs_phys,
                full_y_pred=y_post_mean_phys,
                full_y_pred_std=y_post_std_phys,
                full_parameter_corrections=delta_mean_phys.T,
                filename=validation_options.get(
                    'holdout_predictions_filename',
                    'holdout_predictions.png',
                ),
            )
        if figure_options.get('holdout_nrmse_sensitivity', False):
            plot_holdout_nrmse_sensitivity(
                holdout_metrics,
                figures_directory,
                filename=validation_options.get(
                    'holdout_nrmse_filename',
                    'holdout_nrmse_sensitivity.png',
                ),
                holdout_cases=holdout_metrics['cases'],
                full_x=x_obs_phys,
                full_y_true=y_obs_phys,
                full_y_pred=y_post_mean_phys,
                full_y_pred_std=y_post_std_phys,
            )
        os.makedirs(results_directory, exist_ok=True)
        holdout_json_path = os.path.join(
            results_directory, 'holdout_sensitivity_metrics.json'
        )
        with open(holdout_json_path, 'w') as f:
            json.dump(holdout_metrics, f, indent=4)
        print(f"Saved holdout sensitivity metrics to {holdout_json_path}")

    if figure_options.get("kappa_hyperprior_sensitivity", False):
        hyperprior_metrics = kappa_hyperprior_sensitivity(
            x_obs,
            y_obs,
            theta_fixed,
            gp_eta,
            dtheta,
            kappa_hyperpriors,
            factors=validation_options.get(
                "kappa_hyperprior_factors", (0.5, 1.0, 2.0)
            ),
            n_repeats=validation_options.get(
                "kappa_hyperprior_repeats", 5
            ),
            n_mcmc=validation_options.get(
                "kappa_hyperprior_n_mcmc",
                max(1000, calibration_settings["N_mcmc"] // 5),
            ),
            burn_in=validation_options.get(
                "kappa_hyperprior_burn_in",
                max(250, burnin // 2),
            ),
            mh_scale_delta=calibration_settings["mh_scale_delta_prior"],
            allow_singular_covariance=allow_singular_covariance,
            random_state=validation_options.get("random_seed", 0),
        )
        plot_kappa_hyperprior_sensitivity(
            hyperprior_metrics,
            kappa_labels,
            theta_labels,
            figures_directory,
            filename=validation_options.get(
                "kappa_hyperprior_sensitivity_filename",
                "kappa_hyperprior_sensitivity.png",
            ),
        )
        os.makedirs(results_directory, exist_ok=True)
        hyperprior_json_path = os.path.join(
            results_directory,
            "kappa_hyperprior_sensitivity_metrics.json",
        )
        with open(hyperprior_json_path, "w") as f:
            json.dump(hyperprior_metrics, f, indent=4)
        print(f"Saved kappa hyperprior sensitivity metrics to {hyperprior_json_path}")

    # ============================================================
    # Cross-validation metrics
    # ============================================================

    theta_mse = []
    theta_nrmse = []

    y_mse = None
    y_nrmse = None

    if cross_validation_settings['conduct_cross_validation']:

        known_theta_form = cross_validation_settings.get(
            "known_theta_form"
        )

        known_theta_form_params = (
            cross_validation_settings.get(
                "known_theta_form_params",
                {}
            )
        )

        form_config = (
            known_theta_form_params.get(
                known_theta_form,
                {}
            )
        )

        # ========================================================
        # Theta metrics
        # ========================================================

        for k in range(dtheta):

            theta_true = build_known_theta_field(
                known_theta_form,
                form_config,
                x_sorted_phys,
                k
            )

            theta_pred = theta_post_mean_phys[
                k,
                x_sorted_idx
            ]

            mse_k = np.mean(
                (theta_true - theta_pred)**2
            )

            rmse_k = np.sqrt(mse_k)

            theta_range_k = (
                np.max(theta_true)
                - np.min(theta_true)
            )

            if theta_range_k < 1e-12:
                theta_range_k = 1.0

            nrmse_k = (
                rmse_k / theta_range_k
            )

            theta_mse.append(mse_k)
            theta_nrmse.append(nrmse_k)

            print(
                f"[CV] Theta "
                f"({theta_labels[k]}): "
                f"MSE={mse_k:.6f}, "
                f"NRMSE={nrmse_k:.6f}"
            )

        # ========================================================
        # Posterior predictive metrics
        # ========================================================

        y_true = y_obs_phys.ravel()

        y_pred = y_post_mean_phys.ravel()

        y_mse = np.mean(
            (y_true - y_pred)**2
        )

        y_rmse = np.sqrt(y_mse)

        y_range = (
            np.max(y_true)
            - np.min(y_true)
        )

        y_nrmse = (
            y_rmse / (y_range + 1e-12)
        )

        print(
            f"[CV] Posterior predictive: "
            f"MSE={y_mse:.6f}, "
            f"NRMSE={y_nrmse:.6f}"
        )

    # ============================================================
    # Save JSON
    # ============================================================

    if cross_validation_settings['conduct_cross_validation']:

        cv_results = {

            "known_theta_form": known_theta_form,

            "posterior_predictive_mse": float(y_mse),

            "posterior_predictive_nrmse": float(
                y_nrmse
            ),

            "theta_mse": {},

            "theta_nrmse": {}
        }

        for k, mse_k in enumerate(theta_mse):

            param_name = (
                theta_labels[k]
                if k < len(theta_labels)
                else f"theta_{k}"
            )

            cv_results["theta_mse"][
                param_name
            ] = float(mse_k)

            cv_results["theta_nrmse"][
                param_name
            ] = float(theta_nrmse[k])

        cv_json_path = os.path.join(
            results_directory,
            "cross_validation_metrics.json"
        )

        os.makedirs(results_directory, exist_ok=True)

        with open(cv_json_path, "w") as f:
            json.dump(cv_results, f, indent=4)

        print(
            f"Saved cross-validation metrics to "
            f"{cv_json_path}"
        )

    # ============================================================
    # 3. Delta GP diagnostics: mean + band across x
    # ============================================================

    # Sort x for smooth plotting
    x_sorted_idx = np.argsort(x_obs.ravel())
    x_sorted = x_obs.ravel()[x_sorted_idx]

    # Delta posterior mean/std at each x
    delta_mean = delta_chain.mean(axis=0)   # shape (dtheta, No)
    delta_std  = delta_chain.std(axis=0)


    # ============================================================
    # 9. Plot Posterior Predictive Fit and other figures
    # ============================================================
    if figure_options['posterior_predict_normalized']:
        print('Plotting posterior predictive check (normalized)...')
        plt.figure(figsize=(8,6))

        # Observations
        plt.scatter(
            x_obs, y_obs,
            label=f"{observation_set_label} data",
            marker="o",
            s=60
        )

        # Simulation evaluations
        plt.scatter(
            x_sim, y_sim,
            label=f"{simulation_set_label} runs",
            marker="x",
            alpha=0.7
        )

        # Posterior mean prediction
        plt.plot(
            x_obs,
            y_post_mean,
            label="Posterior mean η(x,θ+δ)",
            linewidth=2
        )

        # Uncertainty band
        plt.fill_between(
            x_obs.ravel(),
            y_post_mean - 2*y_post_std,
            y_post_mean + 2*y_post_std,
            alpha=0.3,
            label="±2 posterior std"
        )

        plt.title(
            f"Posterior Predictive Fit with {simulation_set_label} Runs"
        )
        plt.xlabel("x (normalized domain)")
        plt.ylabel("y (normalized)")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt_path = os.path.join(figures_directory, f"posterior_predictive_check_normalized.png")
        plt.savefig(plt_path, dpi=150)
        print(f"Saved posterior predictive check figure to {plt_path}")
        plt.close()

    if figure_options['posterior_predict_physical']:
        print('Plotting posterior predictive check (physical)...')

        plt.figure(figsize=(8, 6))

        plt.scatter(
            x_obs_phys, y_obs_phys,
            label=f"{observation_set_label} data (physical)",
            marker="o",
            s=60
        )

        plt.scatter(
            x_sim_phys, y_sim_phys,
            label=f"{simulation_set_label} runs (physical)",
            marker="x",
            alpha=0.7
        )

        plt.plot(
            x_obs_phys,
            y_post_mean_phys,
            label="Posterior mean η(x,θ+δ) (physical)",
            linewidth=2
        )

        plt.fill_between(
            x_obs_phys.ravel(),
            y_post_mean_phys - 2 * y_post_std_phys,
            y_post_mean_phys + 2 * y_post_std_phys,
            alpha=0.3,
            label="±2 posterior std"
        )

        plt.title(
            f"Posterior Predictive Fit with {simulation_set_label} Runs "
            "(Physical Units)"
        )
        plt.xlabel("x (physical domain)")
        plt.ylabel("y (physical)")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt_path = os.path.join(figures_directory, f"posterior_predictive_check_physical.png")
        plt.savefig(plt_path, dpi=150)
        print(f"Saved posterior predictive check figure to {plt_path}")
        plt.close()

    if figure_options['delta_posterior_normalized']:
        print('Plotting delta posterior samples (normalized)...')
        # ============================================================
        # 4. Subplots: One delta_k(x) per calibration parameter
        # ============================================================

        fig, axes = plt.subplots(
            dtheta, 1,
            figsize=(9, 3*dtheta),
            sharex=True
        )

        if dtheta == 1:
            axes = [axes]

        param_names = theta_labels # customize if needed

        for k in range(dtheta):

            ax = axes[k]

            mean_k = delta_mean[k, x_sorted_idx]
            std_k  = delta_std[k, x_sorted_idx]

            # Posterior mean
            ax.plot(
                x_sorted,
                mean_k,
                label=f"δ{k+1}(x) posterior mean",
                linewidth=2
            )

            # Credible band
            ax.fill_between(
                x_sorted,
                mean_k - 2*std_k,
                mean_k + 2*std_k,
                alpha=0.3,
                label="±2 std"
            )

            # Overlay a few sample trajectories
            for j in range(5):
                s = np.random.choice(Nmcmc)
                sample_k = delta_chain[s, k, x_sorted_idx]
                ax.plot(x_sorted, sample_k, alpha=0.3)

            ax.axhline(0, color="black", linestyle="--", linewidth=1)

            name = param_names[k] if k < len(param_names) else f"theta{k}"
            ax.set_title(
                f"Discrepancy Field {_math_label(kappa_labels[k])} "
                f"for parameter: {_math_label(name)}"
            )
            ax.set_ylabel(_math_label(kappa_labels[k]))
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("x (normalized domain)")

        plt.tight_layout()
        plt_path = os.path.join(figures_directory, f"delta_posterior_normalized.png")
        plt.savefig(plt_path, dpi=150)
        print(f"Saved delta posterior normalized figure to {plt_path}")
        plt.close()


    if figure_options['delta_posterior_physical']:
        print('Plotting delta posterior samples (physical)...')

        # ============================================================
        # Delta GP diagnostics in physical theta units
        # ============================================================

        fig, axes = plt.subplots(
            dtheta, 1,
            figsize=(9, 3*dtheta),
            sharex=True
        )

        if dtheta == 1:
            axes = [axes]

        param_names = theta_labels

        for k in range(dtheta):
            ax = axes[k]

            mean_k = delta_mean_phys[k, x_sorted_idx]
            std_k = delta_std_phys[k, x_sorted_idx]

            # Plot delta_theta itself as a deviation from the fixed theta^0 = 0
            # reference, rather than the absolute calibrated theta field.
            ax.plot(
                x_sorted_phys,
                mean_k,
                label=f"{_math_label(kappa_labels[k])} posterior mean",
                linewidth=2
            )

            ax.fill_between(
                x_sorted_phys,
                mean_k - 2 * std_k,
                mean_k + 2 * std_k,
                alpha=0.3,
                label="±2 std"
            )
            prior_offset_min = theta_min[k] - theta_fixed_phys[k]
            prior_offset_max = theta_max[k] - theta_fixed_phys[k]
            ax.axhspan(
                prior_offset_min,
                prior_offset_max,
                color="tab:orange",
                alpha=0.12,
                label=r"Theta prior range normalized about $\theta^0$",
            )
            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
                label=(
                    r"$\delta_\theta=0$ baseline "
                    rf"($\theta^0_{{\rm phys}}={theta_fixed_phys[k]:.4g}$)"
                ),
            )
            ax.text(
                0.98,
                0.96,
                rf"$\theta^0_{{\rm phys}}={theta_fixed_phys[k]:.4g}$",
                transform=ax.transAxes,
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

            # ============================================================
            # Overlay known theta field for cross-validation
            # ============================================================

            if cross_validation_settings['conduct_cross_validation']:

                known_theta_form = cross_validation_settings.get(
                    "known_theta_form"
                )

                known_theta_form_params = cross_validation_settings.get(
                    "known_theta_form_params",
                    {}
                )

                form_config = known_theta_form_params.get(
                    known_theta_form,
                    {}
                )

                theta_true = build_known_theta_field(
                    known_theta_form,
                    form_config,
                    x_sorted_phys,
                    k
                ) - theta_fixed_phys[k]

                ax.plot(
                    x_sorted_phys,
                    theta_true,
                    linestyle="--",
                    linewidth=2,
                    label=rf"$\theta_{{{k+1}}}^{{true}}(x)-\theta^0_{{\rm phys}}$"
                )

            # for j in range(5):
            #     s = np.random.choice(Nmcmc)
            #     sample_k = delta_chain_phys[s, k, x_sorted_idx]
            #     ax.plot(x_sorted_phys, sample_k, alpha=0.3)

            name = param_names[k] if k < len(param_names) else f"theta{k}"
            ax.set_title(
                f"Physical discrepancy field for: {_math_label(name)}"
            )
            ax.set_ylabel(
                f"{_math_label(kappa_labels[k])} "
                r"(relative to $\theta^0$)"
            )
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel(_math_label(app_labels[0]))
        plt.tight_layout()
        plt_path = os.path.join(figures_directory, f"delta_posterior_physical.png")
        plt.savefig(plt_path, dpi=150)
        print(f"Saved delta posterior physical figure to {plt_path}")
        plt.close()



    if figure_options['acceptance_trajectory']:
        print('Plotting acceptance trajectory...')
        plot_delta_acceptance_trajectory(accept_trace,window=50, figures_directory=figures_directory)


    if figure_options['step_size_plot']:
        print('Plotting step size trajectory...')
        plot_delta_jump_sizes(delta_chain, figures_directory=figures_directory)


    

    if results_options['save_model_csv']:
        print('Saving model data to CSV...')
        # save in results path
        # os.chdir(results_options['results_path'])
        export_emulator_csv(model_data, "model_data.csv",path=results_options['results_path'])

    if results_options['save_model_json']:
        print('Saving model data to JSON...')
        # os.chdir(results_options['results_path'])
        export_emulator_json(model_data, "model_data.json",path=results_options['results_path'])
    return


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json")
