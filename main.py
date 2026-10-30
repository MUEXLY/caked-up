import sys
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve
from scipy.stats import multivariate_normal, norm
from scipy.stats import invgamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from functs import *
from shrinkage import *
from plottingFuncts import *
from plottingFuncts import _plot_label
import json

def main():

    # ============================================================
    # 1a. Load config settings
    # ============================================================

    if len(sys.argv) < 2:
        raise ValueError("Usage: python main.py <config_path>")

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.load(f)

    # store calibration settings
    calibration_settings = config["calibration_settings"]

    # load settings for shrinkage models
    shrinkage_settings = calibration_settings["shrinkage_models"]

    # load settings for orthogonalization of delta_eta with respect to sensitivities G
    orthogonalization_settings = calibration_settings["orthogonalization_settings"]

    # store input settings
    input_settings = config["input_settings"]

    # determine if conducting cross-validation
    cross_validation_settings = config["cross_validation_settings"]
    if cross_validation_settings["conduct_cross_validation"]:
        print("Cross-validation enabled. Comparing results with known ground truth...")
        print(f"Known theta form: {cross_validation_settings['known_theta_form']}")
        print(f"Known delta form: {cross_validation_settings['known_delta_form']}")

    # store output settings
    output_settings = config["output_settings"]

    # results_path = output_settings["results_path"]

    figure_options = output_settings["figure_options"]

    results_options = output_settings["results_options"]
    live_monitor_enabled = bool(output_settings.get("live_monitor", False))
    live_monitor_path = output_settings.get("live_monitor_path")
    live_monitor_interval = max(int(output_settings.get("live_monitor_interval", 25)), 1)
    validation_options = config.get("validation_options", {})

    # Plot labels are kept separate from the calibration settings when desired.
    # A dictionary is also accepted for backwards-compatible single-file configs.
    plot_settings = config.get("plot_settings", {})
    if isinstance(plot_settings, str):
        plot_settings_path = plot_settings
        if not os.path.isabs(plot_settings_path):
            plot_settings_path = os.path.join(
                os.path.dirname(os.path.abspath(config_path)),
                plot_settings_path,
            )
        with open(plot_settings_path, "r") as f:
            plot_settings = json.load(f)
    elif "plot_settings_path" in config:
        plot_settings_path = config["plot_settings_path"]
        if not os.path.isabs(plot_settings_path):
            plot_settings_path = os.path.join(
                os.path.dirname(os.path.abspath(config_path)),
                plot_settings_path,
            )
        with open(plot_settings_path, "r") as f:
            plot_settings = json.load(f)

    # ------------------------------------------------------------
    # Flexible results path handling (HPC + Optuna compatible)
    # ------------------------------------------------------------
    if "results_path" in output_settings and output_settings["results_path"] is not None:
        # Use path provided by config (Optuna or manual override)
        results_path = output_settings["results_path"]
    else:
        # Fallback to legacy HPC behavior
        config_id = os.path.basename(config_path).replace(".json", "")
        results_path = os.path.join("results", config_id)

    os.makedirs(results_path, exist_ok=True)

    # Figures path
    if "figures_path" in output_settings and output_settings["figures_path"] is not None:
        figure_path = output_settings["figures_path"]
    else:
        figure_path = os.path.join(results_path, "figures")

    os.makedirs(figure_path, exist_ok=True)

    # ============================================================
    # 1b. Load the model and observation data, normalize, and prepare for calibration
    # ============================================================
    # Load data
    # Create a DataFrame to store model data
    # The DataFrame can have any number of rows and any number of columns
    # The columns will be identified by header names x_{label}, theta_{label}, y_{label}, etc.
    # x will be read from modelData/appDomain.txt
    # theta will be read from modelData/thetaVals.txt
    # y will be read from modelData/modelPredictions.txt
    # Each row will correspond to a different model run

    model_data = pd.DataFrame()
    model_data_path = input_settings["model_data_path"]

    #populate x values from appDomain.txt
    # The first line of appDomain.txt contains the header names
    # The subsequent lines contain the x values
    with open(f'{model_data_path}appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(input_settings['input_delimiter'])
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            model_data[col_name] = [float(line.strip().split(input_settings['input_delimiter'])[i]) for line in lines[1:]]

    # #populate theta values from thetaVals.txt
    # The first line of thetaVals.txt contains the header names
    # The subsequent lines contain the theta values
    with open(f'{model_data_path}thetaVals.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(input_settings['input_delimiter'])
        for i, header in enumerate(headers):
            col_name = f'theta_{header}'
            model_data[col_name] = [float(line.strip().split(input_settings['input_delimiter'])[i]) for line in lines[1:]]

    #populate y values from modelPredictions.txt
    # The first line of modelPredictions.txt contains the header names
    # The subsequent lines contain the y values
    # If there are blank lines at the end of the file, ignore
    with open(f'{model_data_path}modelPredictions.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(input_settings['input_delimiter'])
        for i, header in enumerate(headers):
            col_name = f'zeta_{header}'
            model_data[col_name] = [float(line.strip().split(input_settings['input_delimiter'])[i]) for line in lines[1:]]

    # Create a dataframe to store the observation data
    # The DataFrame can have any number of rows and any number of columns
    # The columns will be identified by header names x_{label}, y_{label}, etc.
    # x will be read from obsData/appDomain.txt
    # y will be read from obsData/observationData.txt
    # Each row will correspond to a different observation
    obs_data = pd.DataFrame()
    observation_data_path = input_settings["obs_data_path"]

    with open(f'{observation_data_path}appDomain.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(input_settings['input_delimiter'])
        for i, header in enumerate(headers):
            col_name = f'x_{header}'
            obs_data[col_name] = [float(line.strip().split(input_settings['input_delimiter'])[i]) for line in lines[1:]]

    with open(f'{observation_data_path}observationData.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        headers = lines[0].strip().split(input_settings['input_delimiter'])
        for i, header in enumerate(headers):
            col_name = f'zeta_{header}'
            obs_data[col_name] = [float(line.strip().split(input_settings['input_delimiter'])[i]) for line in lines[1:]]


    # if cross-validation is enabled, make sure that the number of thetas read from the data match the number of cross-validation thetas
    if cross_validation_settings["conduct_cross_validation"]:
        n_theta = len([col for col in model_data.columns if col.startswith('theta_')])
        known_theta_form = cross_validation_settings["known_theta_form"]
        if known_theta_form == "constant":
            n_cv_theta = len(cross_validation_settings["known_theta_form_params"]["constant"]["values"])
        elif known_theta_form == "trig_funct":
            n_cv_theta = len(cross_validation_settings["known_theta_form_params"]["trig_funct"]["functions"])
        elif known_theta_form == "linear":
            n_cv_theta = len(cross_validation_settings["known_theta_form_params"]["linear"]["m"])
        else:
            raise ValueError(f"Unknown known_theta_form: {known_theta_form}")
        if n_theta != n_cv_theta:
            raise ValueError(f"Number of thetas in data ({n_theta}) does not match number of cross-validation thetas ({n_cv_theta}). Please check your data and config file.")

    #labels for bookkeeping
    model_domain = model_data[[col for col in model_data.columns if col.startswith('x_')]]
    calib_params = model_data[[col for col in model_data.columns if col.startswith('theta_')]]

    theta_labels = [col.replace('theta_', '') for col in calib_params.columns]
    app_labels = [col.replace('x_', '') for col in model_domain.columns]

    print(f'Normalizing input data (simulator_space)...')

    # create standardized/normalized dataframes
    model_normalized = model_data.copy()
    observation_normalized = obs_data.copy()

    y_mean = model_data[[col for col in model_data.columns if col.startswith('zeta_')]].mean().iloc[0]
    y_std  = model_data[[col for col in model_data.columns if col.startswith('zeta_')]].std().iloc[0]
    # Use the same normalization procedure for both the model data and the observation data to ensure they are on the same scale for calibration
    # Normalize to model space

    # transform the simulator inputs (x,theta) to the unit hypercube [0,1] using min-max scaling
    # Normalize x columns to [0,1]
    for col in model_data.columns:
        if col.startswith('x_'):
            min_x = model_data[col].min()
            max_x = model_data[col].max()
            model_normalized[col] = (model_data[col] - min_x) / (max_x - min_x)
    
    # Normalize theta columns to [0,1]
    for col in model_data.columns:
        if col.startswith('theta_'):
            min_theta = model_data[col].min()
            max_theta = model_data[col].max()
            model_normalized[col] = (model_data[col] - min_theta) / (max_theta - min_theta)

    # transform the simulator outputs to be zero mean and unit variance
    for col in model_data.columns:
        if col.startswith('zeta_'):
            # mean_val = model_data[col].mean()
            # std_val = model_data[col].std()
            model_normalized[col] = (model_data[col] - y_mean) / y_std
            #save the simulator output mean and std for later use in transforming the discrepancy back to physical units
            # model_normalized[f'{col}_mean'] = y_mean
            # model_normalized[f'{col}_std'] = y_std
            
            
            
    # repeat the procedure for the observation data
    for col in obs_data.columns:
        if col.startswith('x_'):
            # min_val = obs_data[col].min()
            # max_val = obs_data[col].max()
            observation_normalized[col] = (obs_data[col] - min_x) / (max_x - min_x)
            
    for col in obs_data.columns:
        if col.startswith('zeta_'):
            # mean_val = obs_data[col].mean()
            # std_val = obs_data[col].std()
            observation_normalized[col] = (obs_data[col] - y_mean) / y_std
            
    #print normalized dataframes for debugging

    print("\nNormalized Model Data:")
    print(model_normalized.head())
    print("\nNormalized Observation Data:")
    print(observation_normalized.head())

    if cross_validation_settings.get("holdout_data", False):

        holdout_paths = cross_validation_settings["holdout_data_paths"]

        appDomain_holdout = np.loadtxt(
            holdout_paths["appDomain_holdout"]
        )

        observation_holdout = np.loadtxt(
            holdout_paths["observationData_holdout"]
        )

        x_holdout_realspace = as_domain_matrix(appDomain_holdout)

        y_holdout_realspace = np.asarray(observation_holdout).reshape(-1)

        x_holdout_norm = np.column_stack([
            (x_holdout_realspace[:, j] - model_domain.iloc[:, j].min())
            / (model_domain.iloc[:, j].max() - model_domain.iloc[:, j].min())
            for j in range(x_holdout_realspace.shape[1])
        ])

        y_holdout_norm = (y_holdout_realspace - y_mean) / y_std


        holdout_data_realspace = pd.DataFrame({
            'x': x_holdout_realspace,
            'y': y_holdout_realspace
        })

        holdout_data_normalized = pd.DataFrame({
            **{
                f"x_{j}": x_holdout_norm[:, j]
                for j in range(x_holdout_norm.shape[1])
            },
            'y': (y_holdout_realspace - y_mean) / y_std
        })


    # create a reverse-normalization dictionary to store all the necessary information to transform the normalized discrepancy back to physical units for interpretation and visualization
    reverse_normalization = {
        "x": {
            col: {
                "min": float(model_data[col].min()),
                "max": float(model_data[col].max())
            }
            for col in model_data.columns if col.startswith("x_")
        },
        "theta": {
            col: {
                "min": float(model_data[col].min()),
                "max": float(model_data[col].max())
            }
            for col in model_data.columns if col.startswith("theta_")
        },
        "y": {
            col: {
                "mean": float(y_mean),
                "std": float(y_std)
            }
            for col in model_data.columns if col.startswith("zeta_") 
        }
    }

    if results_options['normalization_parameters']:
        print(f'Saving normalization parameters to {results_path}...')
        with open(os.path.join(results_path, "normalization_parameters.json"), "w") as f:
            json.dump(reverse_normalization, f, indent=4)
        print(f"Saved normalization parameters to {os.path.join(results_path, 'normalization_parameters.json')}")

    # print the reverse normalization dictionary for debugging
    print("\nReverse Normalization Dictionary:")
    print(json.dumps(reverse_normalization, indent=4))

    # if rawData figure is to be output, run generate_raw_data_figures(model_data, obs_data, output_settings, results_path)
    if figure_options['data_priors_realspace']:
        print(f'Generating raw data figures in real space...')
        generate_rawData_figure(
            model_data, obs_data, figure_path,
            figure_name="raw_data_realspace.png",
            suptitle="Raw Data (Physical Units)",
            plot_settings=plot_settings,
        )

    if figure_options['data_priors_normalized']:
        print(f'Generating raw data figures in normalized space...')
        generate_rawData_figure(
            model_normalized, observation_normalized, figure_path,
            figure_name="raw_data_normalized.png",
            suptitle="Raw Data (Normalized Space)",
            plot_settings=plot_settings,
        )

    x_sim = as_domain_matrix(model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]].values)
    theta_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('theta_')]].values
    y_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('zeta_') and not col.endswith('_mean') and not col.endswith('_std')]].values

    x_obs = as_domain_matrix(observation_normalized[[col for col in observation_normalized.columns if col.startswith('x_')]].values)
    y_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('zeta_')]].values

    print("Simulator design shape:", x_sim.shape, theta_sim.shape)
    print("Observation shape:", x_obs.shape, y_obs.shape)

    dtheta = theta_sim.shape[1]
    No = len(x_obs)
    theta_prior_min_phys = model_data[
        [f"theta_{label}" for label in theta_labels]
    ].min().to_numpy(dtype=float)
    theta_prior_max_phys = model_data[
        [f"theta_{label}" for label in theta_labels]
    ].max().to_numpy(dtype=float)

    # ============================================================
    # 1c. Instantiate shrinkage prior objects based on config
    # ============================================================

    kappa_prior_config = shrinkage_settings["kappa"]
    kappa_priors = []
    for k in range(dtheta):
        dimension_config = copy.deepcopy(kappa_prior_config)
        dimension_overrides = kappa_prior_config.get("by_dimension", [])
        if k < len(dimension_overrides):
            active_prior = dimension_config["shrinkage_prior"]
            dimension_config[active_prior].update(dimension_overrides[k])
        kappa_priors.append(create_prior(dimension_config))

    print(f"Instantiated kappa_priors of type {kappa_priors[0].name} with initial state: {kappa_priors[0].get_state()}")
    
    delta_eta_prior=create_prior(shrinkage_settings["delta_eta"])
    print(f"Instantiated delta_eta_prior of type {delta_eta_prior.name} with initial state: {delta_eta_prior.get_state()}")



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
        normalize_y=False,
        n_restarts_optimizer=5
    )

    gp_eta.fit(Z_sim, y_sim)

    print("\nTrained emulator kernel:")
    print(gp_eta.kernel_)

    parameter_indices = np.arange(x_sim.shape[1], Z_sim.shape[1])
    sensitivity_labels = [
        _plot_label(plot_settings, "calibration_parameter_labels", k, label)
        for k, label in enumerate(theta_labels)
    ]
    if figure_options.get("emulator_sensitivity_boxplot", False):
        plot_emulator_sensitivity_boxplot(
            gp_eta,
            Z_sim,
            parameter_indices=parameter_indices,
            parameter_labels=sensitivity_labels,
            figures_directory=figure_path,
            filename=figure_options.get(
                "emulator_sensitivity_boxplot_filename",
                "emulator_sensitivity_boxplot.png",
            ),
        )
    if figure_options.get("emulator_sensitivity_pca", False):
        plot_emulator_sensitivity_pca(
            gp_eta,
            Z_sim,
            parameter_indices=parameter_indices,
            parameter_labels=sensitivity_labels,
            figures_directory=figure_path,
            filename=figure_options.get(
                "emulator_sensitivity_pca_filename",
                "emulator_sensitivity_pca.png",
            ),
        )

    # save the following prior information: y_prior_mean,y_prior_var,delta_eta_mean,delta_eta_std,
    y_prior_mean = gp_eta.predict(Z_sim)
    y_prior_var = np.diag(gp_eta.predict(Z_sim, return_cov=True)[1])

    # if output_settings['eta_emulator_prior'] is true, run plot_emulator_prior(x_sim, y_sim, y_prior_mean, y_prior_var, model_data, model_normalized, output_settings['figures_path'], results_path)
    if figure_options['eta_emulator_prior']:
        print(f'Generating emulator prior figure...')
        plot_emulator_prior(
            y_prior_mean, y_prior_var, model_data, model_normalized,
            reverse_normalization, figure_path, results_path,
            plot_settings=plot_settings,
        )

    # intialize storage for deta eta
    delta_eta_mean = np.zeros(No)
    delta_eta_std = np.zeros(No)

    # ============================================================
    # 3. Initialize θ (fixed plug-in estimate) and conduct PCA
    # ============================================================
    theta_fixed_arr =[]
    theta_fixed_phys_arr = []
    theta_settings = calibration_settings['theta_settings']
    theta_init=theta_settings['theta_initialization']

    if theta_init == "fixed":

        fixed_settings = theta_settings["fixed_settings"]

        if fixed_settings.get("use_mean", True):
            theta_fixed = theta_sim.mean(axis=0)

        else:
            fixed_values = np.asarray(fixed_settings["fixed_values"])

            # convert physical -> normalized
            theta_fixed = np.zeros(len(theta_labels))

            for i, label in enumerate(theta_labels):
                min_val = model_data[f"theta_{label}"].min()
                max_val = model_data[f"theta_{label}"].max()

                theta_fixed[i] = (
                    (fixed_values[i] - min_val)
                    / (max_val - min_val)
                )

        theta_fixed_arr.append(theta_fixed)

        print("\nFixed theta used (normalized):")
        print(theta_fixed)

        print("\nFixed theta used (physical units):")

        for i, label in enumerate(theta_labels):

            min_val = model_data[f"theta_{label}"].min()
            max_val = model_data[f"theta_{label}"].max()

            theta_phys = theta_fixed[i] * (max_val - min_val) + min_val

            theta_fixed_phys_arr.append(theta_phys)

            print(f"{label}: {theta_phys}")

    elif theta_init == "compositional":

        No = len(x_obs)

        theta_fixed = np.zeros((No, dtheta))

        print("\nCompositional theta initialized.")

        for k, label in enumerate(theta_labels):

            # interpolate constitutive theta field from simulation points
            theta_fixed[:, k] = interpolate_domain_field(
                x_sim, theta_sim[:, k], x_obs
            )

            min_val = model_data[f"theta_{label}"].min()
            max_val = model_data[f"theta_{label}"].max()

            theta_phys = (
                theta_fixed[:, k] * (max_val - min_val)
                + min_val
            )

            theta_fixed_phys_arr.append(theta_phys)

            print(f"{label}: compositional field loaded")

    else:
        raise NotImplementedError(
            "Only fixed/compositional theta initialization is currently implemented."
        )
    # PCA implementation
    
    # compute jacobian across all x
    J_all = np.array([compute_jacobian(x_obs[i], get_theta_at_obs(theta_fixed, i), gp_eta)
                  for i in range(No)])   # shape (No, dtheta)
    
    U, S, Vt = np.linalg.svd(J_all, full_matrices=False)

    # principal directions in theta space
    W = Vt.T   # shape (dtheta, dtheta)

    # ============================================================
    # 4a. Initialize discrepancy \kappa(x) latent vectors
    # ============================================================

    kappa_z = np.zeros((dtheta, No))  # shape (dtheta, No)
    # kappa_theta = np.zeros((dtheta, No))
    kappa_theta=W @ kappa_z  # shape (dtheta, No)``


    # Initial hyperparameters for each \kappa (i.e. each theta dimension)
    # ell_kappa = calibration_settings['kappa_prior_vals']['ell'] * np.ones(dtheta)
    # # var_delta = 0.05 * np.ones(dtheta)
    # var_kappa = calibration_settings['kappa_prior_vals']['var'] * np.ones(dtheta) # start with smaller variance to encourage more conservative initial discrepancy fields, which can help stabilize early sampling

    # Initial noise variance
    sigma2 = calibration_settings['sigma2_prior_val']**2


    # ============================================================
    # 4b. Initialize additive discrepancy δ_eta(x)
    # ============================================================

    delta_eta = np.zeros(No)

    # hyperparameters for δ_eta GP
    # ell_eta = calibration_settings['delta_eta_prior_vals']['ell']
    # var_eta = calibration_settings['delta_eta_prior_vals']['var']

    # ============================================================
    # 5. Sampler config
    # ============================================================
    Nmcmc = calibration_settings['N_mcmc']
    mh_scale = calibration_settings['mh_scale_kappa_prior']
    mh_scale_kappa=np.ones(dtheta) * mh_scale

    # kappa_prior_ell=calibration_settings['kappa_priors']['ell']
    # kappa_prior_var=calibration_settings['kappa_priors']['var']

    mh_scales=calibration_settings['mh_scales']

    a_sigma = calibration_settings['sigma2_prior_vals']['a']
    b_sigma = calibration_settings['sigma2_prior_vals']['b']

    # Storage
    kappa_theta_chain = np.zeros((Nmcmc, dtheta, No))
    theta_star_chain = np.zeros((Nmcmc, dtheta, No))
    delta_eta_chain = np.zeros((Nmcmc, No))
    sigma2_chain = np.zeros(Nmcmc)

    accept_kappa = np.zeros(dtheta)

    # --- acceptance trace storage ---
    accept_trace = [[] for _ in range(dtheta)]

    # GP hyperparameter diagnostics. The first trace row records the values
    # before any MCMC update; subsequent rows record each sampler iteration.
    hyperparameter_labels = []
    hyperparameter_prior_configs = []
    hyperparameter_initial_values = []
    hyperparameter_sources = []
    for k, prior in enumerate(kappa_priors):
        if prior.name == "gp":
            hyperparameter_labels.append(f"kappa_{k}")
            hyperparameter_prior_configs.append(prior.config)
            hyperparameter_initial_values.append([prior.ell, prior.var])
            hyperparameter_sources.append(("kappa", k))
    if delta_eta_prior.name == "gp":
        hyperparameter_labels.append("delta_eta")
        hyperparameter_prior_configs.append(delta_eta_prior.config)
        hyperparameter_initial_values.append([delta_eta_prior.ell, delta_eta_prior.var])
        hyperparameter_sources.append(("delta_eta", None))

    hyperparameter_initial_values = np.asarray(hyperparameter_initial_values, dtype=float)
    hyperparameter_trace = np.full(
        (Nmcmc + 1, len(hyperparameter_sources), 2), np.nan
    )
    for h, (source, index) in enumerate(hyperparameter_sources):
        prior = kappa_priors[index] if source == "kappa" else delta_eta_prior
        hyperparameter_trace[0, h] = [prior.ell, prior.var]

    density_traces = []
    density_specs = []
    live_density_stride = max(
        1,
        int(figure_options.get("hyperprior_density_animation_stride", 100)),
    )

    def register_density_hyperparameters(prior, label):
        if prior.name == "gp":
            density_specs.extend([
                {
                    "label": label,
                    "name": r"$\ell$",
                    "parameter": "ell",
                    "kind": "lognormal",
                    "prior": prior.config["ell_prior"],
                },
                {
                    "label": label,
                    "name": r"$\mathrm{var}$",
                    "parameter": "var",
                    "kind": "lognormal",
                    "prior": prior.config["var_prior"],
                },
            ])
            density_traces.extend([[prior.ell], [prior.var]])
        elif prior.name in {"lasso", "fused_lasso"}:
            density_specs.append({
                "label": label,
                "name": r"$\lambda$",
                "parameter": "lambda",
                "kind": "gamma",
                "prior": prior.config["lambda_prior"],
            })
            density_traces.append([prior.lam])
        elif prior.name == "spike_slab":
            density_specs.append({
                "label": label,
                "name": r"$\pi$",
                "parameter": "pi",
                "kind": "beta",
                "prior": prior.config["pi_prior"],
            })
            density_traces.append([prior.pi])
        elif prior.name == "horseshoe":
            density_specs.append({
                "label": label,
                "name": r"$\tau$",
                "kind": "halfcauchy",
                "prior": {"scale": 1.0},
            })
            density_traces.append([
                prior.tau.get(
                    0 if label == "delta_eta" else int(label.rsplit("_", 1)[-1]),
                    prior.tau_init,
                )
            ])

    for k, prior in enumerate(kappa_priors):
        register_density_hyperparameters(prior, f"kappa_{k}")
    register_density_hyperparameters(delta_eta_prior, "delta_eta")

    # --- MH scaling adaptation settings ---
    burnin = calibration_settings['burn_in']
    adapt_interval = calibration_settings['adapt_interval']
    target_accept = calibration_settings['target_accept']


    # store scale history (optional)
    mh_scale_trace = []

    # ============================================================
    # 6. Run Embedded Discrepancy Sampler
    # ============================================================

    print("\nRunning embedded δ-MH sampler...\n")

    for it in range(Nmcmc):

        # ---- update kappa fields ----
        for k in range(dtheta):
            kappa_z, acc = kappa_priors[k].sample_kappa_field(
                k,
                kappa_z,
                delta_eta,
                theta_fixed,
                x_obs,
                y_obs,
                gp_eta,
                sigma2,
                mh_scale_kappa[k]
            )
            accept_kappa[k] += acc
            accept_trace[k].append(acc)

            # kappa_theta[k] -= np.mean(kappa_theta[k])

        
            
        # ---- adapt mh_scale_kappa during burn-in ----
        if it < burnin and (it + 1) % adapt_interval == 0:

            gamma = 0.05  # adaptation speed

            for k in range(dtheta):

                # acceptance rate over last window
                recent_accept_k = np.mean(accept_trace[k][-adapt_interval:])

                # log-scale update
                mh_scale_kappa[k] *= np.exp(gamma * (recent_accept_k - target_accept))

                print(f"[ADAPT] Iter {it+1}, k={k}: acc={recent_accept_k:.3f}, scale={mh_scale_kappa[k]:.4f}")
                print(f"[ADAPT] Updated mh_scale_kappa for theta_{k} = {mh_scale_kappa[k]:.4f}")



        # ---- update \kappa hyperparameters ----
        # for k in range(dtheta):
        #     ell_kappa[k], var_kappa[k], _ = mh_update_delta_hyperparams(
        #         kappa_z[k],
        #         ell_kappa[k],
        #         var_kappa[k],
        #         x_obs,
        #         kappa_prior_ell,
        #         kappa_prior_var,
        #         mh_scales,
        #         calibration_settings['allow_singular_cov']
        #     )

        for k in range(dtheta):
            kappa_priors[k].update_hyperparameters(
                k=k,
                z=kappa_z[k],
                x_obs=x_obs,
                mh_scales=mh_scales,
                allow_singular_cov=calibration_settings['allow_singular_cov']
            )

        # ---- compute sensitivities ---- (@ theta fixed)
        G=compute_sensitivities(x_obs, theta_fixed, gp_eta, orthogonalization_settings['theta_epsilon'])

        # ---- update δ_eta_raw (Gibbs) ----
        # delta_eta_raw = gibbs_delta_eta(
        #     x_obs,
        #     y_obs,
        #     theta_fixed,
        #     kappa_theta,
        #     gp_eta,
        #     sigma2,
        #     delta_eta_prior.ell,
        #     delta_eta_prior.var
        # )

        delta_eta_raw = delta_eta_prior.sample_delta_eta_field(
            delta_eta,
            x_obs,
            y_obs,
            theta_fixed,
            kappa_theta,
            gp_eta,
            sigma2
        )


        # orthogonalize delta_eta with respect to the sensitivities G

        if orthogonalization_settings['orthogonalize_delta_eta']:
            delta_eta=orthogonalize_delta_eta(delta_eta_raw, G)
        else:
            delta_eta = delta_eta_raw

        delta_eta_prior.update_hyperparameters(
            k=None,
            z=delta_eta,
            x_obs=x_obs,
            mh_scales=mh_scales,
            allow_singular_cov=calibration_settings['allow_singular_cov']
        )

        for h, (source, index) in enumerate(hyperparameter_sources):
            prior = kappa_priors[index] if source == "kappa" else delta_eta_prior
            hyperparameter_trace[it + 1, h] = [prior.ell, prior.var]

        density_state_index = 0
        for prior in [*kappa_priors, delta_eta_prior]:
            if prior.name == "gp":
                density_traces[density_state_index].append(prior.ell)
                density_traces[density_state_index + 1].append(prior.var)
                density_state_index += 2
            elif prior.name in {"lasso", "fused_lasso"}:
                density_traces[density_state_index].append(prior.lam)
                density_state_index += 1
            elif prior.name == "spike_slab":
                density_traces[density_state_index].append(prior.pi)
                density_state_index += 1
            elif prior.name == "horseshoe":
                index = (
                    0 if prior is delta_eta_prior
                    else kappa_priors.index(prior)
                )
                density_traces[density_state_index].append(
                    prior.tau.get(index, prior.tau_init)
                )
                density_state_index += 1

        if (
            figure_options.get("hyperprior_density_animation", False)
            and density_specs
            and ((it + 1) % live_density_stride == 0 or it + 1 == Nmcmc)
        ):
            try:
                plot_hyperparameter_density_state(
                    [trace[-1] for trace in density_traces],
                    density_specs,
                    figure_path,
                    filename="hyperparameter_density_live.png",
                    iteration=it + 1,
                )
            except (OSError, ValueError, KeyError, IndexError) as exc:
                print(f"Could not update live hyperparameter density: {exc}")

        # ---- Gibbs update σ² ----
        sigma2 = gibbs_sigma2(
        y_obs,
        x_obs,
        theta_fixed,
        kappa_theta,
        delta_eta,   # NEW
        gp_eta,
        a_sigma,
        b_sigma
        )
        
        # CRITICAL: map back AFTER updating all components
        kappa_theta = W @ kappa_z

        # ---- diagnostic ----
        # print("theta_fixed.shape:", np.shape(theta_fixed))
        # print("kappa_theta.shape:", np.shape(kappa_theta))
        # print("x_obs.shape:", np.shape(x_obs))
        # print("theta_sim.shape:", np.shape(theta_sim))

        # ---- store ----
        kappa_theta_chain[it] = kappa_theta
        if theta_fixed.ndim == 1:
            theta_star_chain[it] = theta_fixed[:, None] + kappa_theta
        else:
            theta_star_chain[it] = theta_fixed.T + kappa_theta
        delta_eta_chain[it] = delta_eta
        sigma2_chain[it] = sigma2

        if (
            live_monitor_enabled
            and live_monitor_path
            and ((it + 1) % live_monitor_interval == 0 or it + 1 == Nmcmc)
        ):
            live_indices = np.linspace(
                0, No - 1, min(No, 500), dtype=int
            )
            theta_base = (
                np.tile(theta_fixed, (No, 1))
                if theta_fixed.ndim == 1
                else theta_fixed
            )
            theta_current = theta_base[live_indices] + kappa_theta.T[live_indices]
            live_inputs = np.hstack([x_obs[live_indices], theta_current])
            live_mean, live_std = gp_eta.predict(live_inputs, return_std=True)
            live_mean = live_mean + delta_eta[live_indices]
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
                "observed": y_obs[live_indices, 0].tolist(),
                "predicted": live_mean.tolist(),
                "uncertainty": live_uncertainty.tolist(),
            }
            live_temp_path = f"{live_monitor_path}.tmp"
            with open(live_temp_path, "w") as stream:
                json.dump(live_payload, stream)
            os.replace(live_temp_path, live_monitor_path)

        # ---- diagnostics ----
        if (it+1) % 50 == 0:
            print(f"Iter {it+1}/{Nmcmc}")
            print(" sigma2 =", sigma2)
            print(" kappa acceptance rates:",
                accept_kappa / (it+1))
            for h, label in enumerate(hyperparameter_labels):
                print(
                    f" {label} hyperparameters:",
                    f"ell={hyperparameter_trace[it + 1, h, 0]:.6g},",
                    f"var={hyperparameter_trace[it + 1, h, 1]:.6g}",
                )
            if orthogonalization_settings['orthogonalize_delta_eta']:
                proj = G @ np.linalg.solve(G.T @ G, G.T @ delta_eta)
                print("Projection norm (should be near 0):", np.linalg.norm(proj))
                
            # print("||kappa_z||      =", np.linalg.norm(kappa_z))
            # print("||W @ kappa_z||  =", np.linalg.norm(W @ kappa_z))
            # print("||kappa_theta||  =", np.linalg.norm(kappa_theta))

            print("--------------------------------------------------")


    print("\nSampler complete.")

    if figure_options.get("prior_posterior_comparison", False):
        theta_columns = [
            col for col in model_data.columns if col.startswith("theta_")
        ]
        theta_scales = np.array([
            reverse_normalization["theta"][col]["max"]
            - reverse_normalization["theta"][col]["min"]
            for col in theta_columns
        ])
        theta_offsets = np.array([
            reverse_normalization["theta"][col]["min"]
            for col in theta_columns
        ])
        theta_posterior_physical = (
            theta_star_chain * theta_scales[None, :, None]
            + theta_offsets[None, :, None]
        ).transpose(0, 2, 1).reshape(-1, dtheta)
        plot_prior_posterior_comparison(
            theta_prior_physical=model_data[theta_columns].to_numpy(),
            theta_posterior_physical=theta_posterior_physical,
            theta_labels=theta_labels,
            hyperparameter_traces=density_traces,
            hyperparameter_specs=density_specs,
            figures_directory=figure_path,
            filename=figure_options.get(
                "prior_posterior_comparison_filename",
                "prior_posterior_comparison.png",
            ),
            burn_in=burnin,
        )

    if figure_options.get("calibration_prediction_animation", False):
        animation_space = figure_options.get(
            "calibration_prediction_animation_space", "physical"
        ).lower()
        if animation_space not in {"physical", "normalized"}:
            raise ValueError(
                "calibration_prediction_animation_space must be "
                "'physical' or 'normalized'"
            )

        if animation_space == "physical":
            x_column = next(
                col for col in model_data.columns if col.startswith("x_")
            )
            x_stats = reverse_normalization["x"][x_column]
            theta_columns = [
                col for col in model_data.columns if col.startswith("theta_")
            ]
            animation_kwargs = {
                "y_scale": y_std,
                "y_offset": y_mean,
                "x_scale": x_stats["max"] - x_stats["min"],
                "x_offset": x_stats["min"],
                "theta_scales": np.array([
                    reverse_normalization["theta"][col]["max"]
                    - reverse_normalization["theta"][col]["min"]
                    for col in theta_columns
                ]),
                "theta_offsets": np.array([
                    reverse_normalization["theta"][col]["min"]
                    for col in theta_columns
                ]),
            }
        else:
            animation_kwargs = {}

        plot_calibration_prediction_animation(
            x_obs=x_obs,
            y_obs=y_obs,
            kappa_theta_chain=kappa_theta_chain,
            delta_eta_chain=delta_eta_chain,
            theta_fixed=theta_fixed,
            gp_eta=gp_eta,
            figures_directory=figure_path,
            filename=figure_options.get(
                "calibration_prediction_animation_filename",
                "calibration_prediction_updates.gif",
            ),
            frame_stride=figure_options.get(
                "calibration_prediction_animation_stride", 100
            ),
            max_posterior_samples=figure_options.get(
                "calibration_prediction_animation_samples", 200
            ),
            theta_labels=theta_labels,
            space_label=animation_space,
            **animation_kwargs,
        )

    if figure_options.get("hyperprior_updates", False) and len(hyperparameter_sources):
        animation_result = plot_hyperparameter_updates(
            hyperparameter_trace=hyperparameter_trace,
            hyperparameter_priors=hyperparameter_prior_configs,
            initial_values=hyperparameter_initial_values,
            parameter_labels=hyperparameter_labels,
            figures_directory=figure_path,
            filename=figure_options.get(
                "hyperprior_updates_filename", "hyperparameter_updates.png"
            ),
            animation_filename=figure_options.get(
                "hyperprior_updates_animation_filename",
                "hyperparameter_updates.gif",
            ),
            make_animation=figure_options.get(
                "hyperprior_updates_animation", False
            ),
            animation_stride=figure_options.get(
                "hyperprior_updates_animation_stride", 100
            ),
        )

    if figure_options.get("hyperprior_density_animation", False) and density_specs:
        plot_hyperparameter_density_animation(
            hyperparameter_traces=density_traces,
            hyperparameter_specs=density_specs,
            figures_directory=figure_path,
            filename=figure_options.get(
                "hyperprior_density_animation_filename",
                "hyperparameter_density_updates.gif",
            ),
            animation_stride=figure_options.get(
                "hyperprior_density_animation_stride", 100
            ),
        )

    # if results_options['parameter_traces'] is true, save all parameter traces (kappa_theta_chain, delta_eta_chain, sigma2_chain) to results_path in a single .txt file with appropriate headers
    if results_options['parameter_traces']:
        print(f'Saving parameter traces to {results_path}...')
        # os.makedirs(results_path, exist_ok=True)

        # flatten kappa_theta_chain for saving
        Nmcmc, dtheta, No = kappa_theta_chain.shape
        kappa_flat = kappa_theta_chain.reshape(Nmcmc, dtheta*No)

        # create header
        kappa_headers = [f'kappa_theta_{k}_{i}' for k in range(dtheta) for i in range(No)]
        delta_eta_headers = [f'delta_eta_{i}' for i in range(No)]
        headers = kappa_headers + delta_eta_headers + ['sigma2']

        # concatenate all chains for saving
        output_data = np.hstack([
            kappa_flat,
            delta_eta_chain,
            sigma2_chain.reshape(-1,1)])
        
        np.savetxt(os.path.join(results_path, "parameter_traces.txt"), output_data, header="\t".join(headers), delimiter="\t")
        print(f"Saved parameter traces to {os.path.join(results_path, 'parameter_traces.txt')}")

    # Fit supported shrinkage-hyperparameter posteriors for sequential runs.
    posterior_start = min(max(int(burnin) + 1, 0), Nmcmc)
    posterior_hyperparameters = {}
    for spec, trace in zip(density_specs, density_traces):
        samples = np.asarray(trace[posterior_start:], dtype=float)
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if samples.size == 0 or spec["kind"] == "halfcauchy":
            continue

        parameter = spec["parameter"]
        kind = spec["kind"]
        if kind == "lognormal":
            log_samples = np.log(samples)
            log_mean = float(np.mean(log_samples))
            log_std = float(np.std(log_samples, ddof=1)) if samples.size > 1 else 0.0
            fitted = {"mu": log_mean, "sigma": max(log_std, 1e-8)}
            mean = float(np.exp(log_mean + 0.5 * log_std**2))
            family = "gp"
        elif kind == "gamma":
            sample_mean = float(np.mean(samples))
            sample_var = float(np.var(samples, ddof=1)) if samples.size > 1 else 0.0
            fallback = spec["prior"]
            fitted = (
                {"shape": sample_mean**2 / sample_var, "rate": sample_mean / sample_var}
                if sample_var > 1e-12 else dict(fallback)
            )
            mean = fitted["shape"] / fitted["rate"]
            family = "gamma"
        elif kind == "beta":
            bounded_samples = np.clip(samples, 1e-8, 1 - 1e-8)
            sample_mean = float(np.mean(bounded_samples))
            sample_var = float(np.var(bounded_samples, ddof=1)) if samples.size > 1 else 0.0
            max_var = sample_mean * (1 - sample_mean)
            sample_var = min(sample_var, max_var * (1 - 1e-8))
            fallback = spec["prior"]
            if sample_var > 1e-12:
                concentration = max_var / sample_var - 1
                fitted = {
                    "alpha": sample_mean * concentration,
                    "beta": (1 - sample_mean) * concentration,
                }
            else:
                fitted = dict(fallback)
            mean = fitted["alpha"] / (fitted["alpha"] + fitted["beta"])
            family = "beta"
        else:
            continue

        record = posterior_hyperparameters.setdefault(
            spec["label"], {"kind": family, "parameters": {}, "means": {}}
        )
        record["parameters"][parameter] = fitted
        record["means"][parameter] = mean

    posterior_path = os.path.join(results_path, "hyperparameter_posterior.json")
    with open(posterior_path, "w") as f:
        json.dump(posterior_hyperparameters, f, indent=2)
    print(f"Saved shrinkage hyperparameter posterior to {posterior_path}")

    # ============================================================
    # 7. Posterior Predictive Check
    # ============================================================

    print("\nBuilding posterior predictive mean + uncertainty...\n")

    y_post_mean = np.zeros(No)
    y_post_var  = np.zeros(No)

    Nsamp = calibration_settings['posterior_predictive_samples']
    idx = np.random.choice(Nmcmc, Nsamp, replace=False)

    for i in range(No):

        preds = []

        for s in idx:

            kappa_theta_s = kappa_theta_chain[s, :, i]
            theta_star = get_theta_at_obs(theta_fixed, i) + kappa_theta_s
            delta_eta_s = delta_eta_chain[s, i]


            m_i, s2_i = eta_predict(x_obs[i], theta_star, gp_eta)

            preds.append(m_i+delta_eta_s)

        preds = np.array(preds)
        

        y_post_mean[i] = preds.mean()
        y_post_var[i]  = preds.var()

        delta_eta_mean[i] = delta_eta_chain[idx, i].mean()
        delta_eta_std[i] = delta_eta_chain[idx, i].std()

        # # print prediction mean and 95% credible interval
        # print(f"x_obs[{i}] = {x_obs[i,0]:.3f}, y_obs = {y_obs[i]:.3f}, post pred mean = {y_post_mean[i]:.3f}, 95% CI = [{y_post_mean[i] - 2*np.sqrt(y_post_var[i]):.3f}, {y_post_mean[i] + 2*np.sqrt(y_post_var[i]):.3f}]")
        # print(f" delta_eta mean = {delta_eta_mean[i]:.3f}, std = {delta_eta_std[i]:.3f}")

        # if results_options['posterior_predictive_samples'] is true, save the posterior predictive samples for each observation to results_path in a file named posterior_predictive_samples.txt with appropriate headers
       
    # Save posterior predictive mean and std
    if results_options['posterior_predictive_samples']:
        pp_stats = np.column_stack([y_post_mean, np.sqrt(y_post_var)])
        headers = ['y_post_mean', 'y_post_std']
        np.savetxt(os.path.join(results_path, "posterior_predictive_mean_std.txt"), pp_stats, header="\t".join(headers), delimiter="\t")
        print(f"Saved posterior predictive mean and std to {os.path.join(results_path, 'posterior_predictive_mean_std.txt')}")

    # ============================================================
    # 8. Store all normalized and physical-space results to dataframes
    # ============================================================

    # Calculate kappa mean and std across MCMC samples
    kappa_mean = kappa_theta_chain[idx, :, :].mean(axis=0)  # shape: (dtheta, No)
    kappa_std = kappa_theta_chain[idx, :, :].std(axis=0)    # shape: (dtheta, No)
    
    # Build results dataframe with dynamic kappa columns
    results_data = {
        "zeta_obs": y_obs.flatten(),
        # "x_sim": model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]].values.flatten(),
        # "xi_sim": model_normalized[[col for col in model_normalized.columns if col.startswith('y_') and not col.endswith('_mean') and not col.endswith('_std')]].values.flatten(),
        "y_post_mean": y_post_mean,
        "y_post_std": np.sqrt(y_post_var),
        "delta_eta_mean": delta_eta_mean,
        "delta_eta_std": delta_eta_std,
    }
    for j, x_col in enumerate([col for col in model_normalized.columns if col.startswith("x_")]):
        results_data[f"x_obs_{x_col.replace('x_', '')}"] = x_obs[:, j]
    # Preserve the legacy single-domain column as the first coordinate.
    results_data["x_obs"] = x_obs[:, 0]
    
    # Add kappa mean and std for each theta dimension
    for k in range(dtheta):
        results_data[f"kappa_{k}_mean"] = kappa_mean[k, :]
        results_data[f"kappa_{k}_std"] = kappa_std[k, :]
    

    # print the size of all entries in results_data for debugging
    for key, value in results_data.items():
        print(f"{key}: {value.shape}")

    results_normalized = pd.DataFrame(results_data)
    
    # Reverse normalization for real-space results using the reverse-normalization dictionary
    x_cols = [col for col in model_normalized.columns if col.startswith("x_")]
    theta_cols = [col for col in model_normalized.columns if col.startswith("theta_")]    
    # Build physical space results dataframe
    results_data_physical = {}
    
    # Reverse-normalize x columns
    for x_col in x_cols:
        x_stats = reverse_normalization["x"][x_col]
        x_min, x_max = x_stats["min"], x_stats["max"]
        x_label = x_col.replace('x_', '')
        x_norm_col = f"x_obs_{x_label}"
        x_phys = results_normalized[x_norm_col].values * (x_max - x_min) + x_min
        results_data_physical[f"x_obs_{x_label}"] = x_phys
    
    # Observation data (zeta columns) - reverse-normalize using y stats (same normalization applied)
    zeta_cols = [col for col in observation_normalized.columns if col.startswith("zeta_")]
    # y_col = y_cols[0]  # Use first y column stats for observations
    y_col = [col for col in model_normalized.columns if col.startswith("zeta_")][0]
    y_stats = reverse_normalization["y"][y_col]
    y_mu, y_sd = y_stats["mean"], y_stats["std"]
    
    for zeta_col in zeta_cols:
        y_label = zeta_col.replace("zeta_", "")
        results_data_physical[f"zeta_obs_{y_label}"] = results_normalized["zeta_obs"].values * y_sd + y_mu
    
    # Posterior predictive and discrepancy (apply y normalization to all y outputs)
    results_data_physical["y_post_mean"] = results_normalized["y_post_mean"] * y_sd + y_mu
    results_data_physical["y_post_std"] = results_normalized["y_post_std"] * y_sd
    results_data_physical["delta_eta_mean"] = results_normalized["delta_eta_mean"] * y_sd
    results_data_physical["delta_eta_std"] = results_normalized["delta_eta_std"] * y_sd
    
    # Add kappa columns (convert from normalized theta space to physical theta space)
    for k, theta_col in enumerate(theta_cols):
        theta_stats = reverse_normalization["theta"][theta_col]
        theta_min, theta_max = theta_stats["min"], theta_stats["max"]
        scale = theta_max - theta_min
        
        results_data_physical[f"kappa_{k}_mean"] = results_normalized[f"kappa_{k}_mean"] * scale
        results_data_physical[f"kappa_{k}_std"] = results_normalized[f"kappa_{k}_std"] * scale
    
    results_physical = pd.DataFrame(results_data_physical)

    # print the first few rows of the results dataframes for debugging
    print("\nNormalized Results:")
    print(results_normalized.head())
    print("\nPhysical Results:")
    print(results_physical.head())

    external_validation = validation_options.get("external_validation", {})
    if external_validation.get("enabled", False):
        csv_path = external_validation.get("csv_path")
        if not csv_path:
            raise ValueError(
                "validation_options.external_validation.csv_path is required "
                "when external validation is enabled"
            )
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), csv_path)
        external_data = pd.read_csv(csv_path)

        x_columns = [col for col in model_data.columns if col.startswith("x_")]
        theta_columns = [col for col in model_data.columns if col.startswith("theta_")]
        y_column = [col for col in model_data.columns if col.startswith("zeta_")][0]
        observation_y_column = [
            col for col in obs_data.columns if col.startswith("zeta_")
        ][0]
        plot_domain_index = int(external_validation.get("plot_domain_index", 0))
        if plot_domain_index >= len(x_columns):
            raise ValueError("plot_domain_index exceeds the number of domain coordinates")

        domain_columns = external_validation.get("domain_columns")
        if domain_columns is None:
            domain_columns = [external_validation.get("x_column")]
        if len(domain_columns) != len(x_columns):
            raise ValueError(
                "external_validation.domain_columns must contain one CSV column "
                "per application-domain coordinate"
            )
        required_columns = [
            external_validation.get("y_column"),
            *domain_columns,
            *external_validation.get("parameter_columns", {}).values(),
        ]
        missing_columns = [
            column for column in required_columns
            if not column or column not in external_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"External validation CSV is missing columns: {missing_columns}"
            )

        x_external_phys = external_data[domain_columns].to_numpy(dtype=float)
        y_external_phys = (
            external_data[external_validation["y_column"]].to_numpy(dtype=float)
            * external_validation.get("y_scale", 1.0)
        )
        parameter_mapping = external_validation["parameter_columns"]
        if len(parameter_mapping) != dtheta:
            raise ValueError(
                "external_validation.parameter_columns must contain one CSV "
                "column per calibration parameter"
            )
        external_theta_phys = external_data[
            list(parameter_mapping.values())
        ].to_numpy(dtype=float)

        x_stats = reverse_normalization["x"]
        x_external_norm = np.column_stack([
            (x_external_phys[:, j] - x_stats[x_columns[j]]["min"])
            / (x_stats[x_columns[j]]["max"] - x_stats[x_columns[j]]["min"])
            for j in range(len(x_columns))
        ])
        theta_stats = reverse_normalization["theta"]
        theta_min = np.array([theta_stats[col]["min"] for col in theta_columns])
        theta_max = np.array([theta_stats[col]["max"] for col in theta_columns])
        theta_range = theta_max - theta_min
        external_theta_norm = (external_theta_phys - theta_min) / theta_range

        x_obs_phys_matrix = np.column_stack([
            results_physical[f"x_obs_{col.replace('x_', '')}"].values
            for col in x_columns
        ])
        x_plot_obs = x_obs_phys_matrix[:, plot_domain_index]
        x_plot_external = x_external_phys[:, plot_domain_index]
        sort_idx = np.argsort(x_plot_obs)

        kappa_phys = np.column_stack([
            results_physical[f"kappa_{k}_mean"].values
            for k in range(dtheta)
        ])
        delta_phys = results_physical["delta_eta_mean"].values
        delta_std_phys = results_physical["delta_eta_std"].values
        external_correction_phys = np.column_stack([
            np.interp(x_plot_external, x_plot_obs[sort_idx], kappa_phys[sort_idx, k])
            for k in range(dtheta)
        ])
        external_delta_phys = np.interp(
            x_plot_external, x_plot_obs[sort_idx], delta_phys[sort_idx]
        )
        external_delta_std_phys = np.interp(
            x_plot_external, x_plot_obs[sort_idx], delta_std_phys[sort_idx]
        )
        if theta_init == "fixed":
            calibration_base_phys = np.tile(
                np.asarray(theta_fixed_phys_arr, dtype=float),
                (len(x_plot_external), 1),
            )
        else:
            calibration_base_phys = np.column_stack([
                np.interp(
                    x_plot_external,
                    x_plot_obs[sort_idx],
                    np.asarray(theta_fixed_phys_arr[k])[sort_idx],
                )
                for k in range(dtheta)
            ])
        posterior_parameters_phys = calibration_base_phys + external_correction_phys

        external_predictions = []
        for row in range(len(external_data)):
            theta_input = external_theta_norm[row] + (
                external_correction_phys[row] / theta_range
            )
            mean, _ = eta_predict(x_external_norm[row], theta_input, gp_eta)
            external_predictions.append(mean * y_sd + y_mu + external_delta_phys[row])

        plot_external_validation(
            model_data[x_columns].to_numpy()[:, plot_domain_index],
            model_data[y_column].to_numpy(),
            x_obs_phys_matrix[:, plot_domain_index],
            results_physical[
                f"zeta_obs_{observation_y_column.replace('zeta_', '')}"
            ].values,
            x_plot_obs,
            results_physical["y_post_mean"].values,
            results_physical["y_post_std"].values,
            x_plot_external,
            y_external_phys,
            external_predictions,
            theta_labels,
            external_theta_phys,
            external_correction_phys,
            figure_path,
            observation_set_label=plot_settings.get("observation_data_label", "Observed"),
            simulation_set_label=plot_settings.get("simulation_data_label", "Simulator"),
            application_domain_label=_plot_label(
                plot_settings, "application_domain_labels", plot_domain_index, "x"
            ),
            observation_label=plot_settings.get("observation_value_label", "y"),
            filename=external_validation.get(
                "figure_filename", "external_validation.png"
            ),
            prior_parameters=model_data[theta_columns].to_numpy(),
            posterior_parameters=posterior_parameters_phys,
            external_discrepancy=external_delta_phys,
            external_discrepancy_std=external_delta_std_phys,
        )

        external_output = external_data.copy()
        external_output["posterior_predictive"] = external_predictions
        external_output["additive_discrepancy_mean"] = external_delta_phys
        external_output["additive_discrepancy_std"] = external_delta_std_phys
        for k, label in enumerate(theta_labels):
            external_output[f"{label}_learned_correction"] = (
                external_correction_phys[:, k]
            )
        os.makedirs(results_path, exist_ok=True)
        external_output.to_csv(
            os.path.join(results_path, "external_validation_predictions.csv"),
            index=False,
        )

    if results_options['output_normalized_results']:
        print(f'Saving normalized results to {results_path}...')
        np.savez(os.path.join(results_path, "results_normalized.npz"), **results_normalized)
        print(f"Saved normalized results to {os.path.join(results_path, 'results_normalized.npz')}")

    if results_options['output_physical_results']:
        print(f'Saving physical results to {results_path}...')
        np.savez(os.path.join(results_path, "results_physical.npz"), **results_physical)
        print(f"Saved physical results to {os.path.join(results_path, 'results_physical.npz')}")

    # if evaluating against holdout data, interpolate the kappa and delta fields to the holdout x values and save the results to a separate file
    if cross_validation_settings.get("holdout_data", False):

        #initialize the holdout posterior predictive mean array
        y_holdout_post_mean = np.zeros(len(x_holdout_norm))
        y_holdout_post_var = np.zeros(len(x_holdout_norm))

        # initialize the holdout posterior predictive std array
        y_holdout_post_std = np.zeros(len(x_holdout_norm))

        # initialize the holdout kappa and delta arrays
        kappa_holdout = np.zeros((dtheta, len(x_holdout_norm)))
        delta_holdout = np.zeros(len(x_holdout_norm))

        # interpolate the kappa and delta fields to the holdout x values
        for k in range(dtheta):

            kappa_holdout[k] = interpolate_domain_field(
                x_obs, kappa_mean[k], x_holdout_norm
            )

        delta_holdout = interpolate_domain_field(
            x_obs, delta_eta_mean, x_holdout_norm
        )

        if theta_fixed.ndim == 2:
            theta_holdout_base = np.column_stack([
                interpolate_domain_field(x_obs, theta_fixed[:, k], x_holdout_norm)
                for k in range(dtheta)
            ])
        else:
            theta_holdout_base = theta_fixed

        # compute the posterior predictive mean for the holdout points
        for i in range(len(x_holdout_norm)):

            theta_star = (
                theta_holdout_base[i] if theta_fixed.ndim == 2 else theta_holdout_base
            ) + kappa_holdout[:, i]

            m_i, s2_i = eta_predict(
                x_holdout_norm[i],
                theta_star,
                gp_eta
            )

            y_holdout_post_mean[i] = (
                m_i
                + delta_holdout[i]
            )

            # compute the posterior predictive variance for the holdout points
            y_holdout_post_var[i] = (
                s2_i
                + delta_eta_prior.var
                + sigma2
            )

        # differentiate normalized vs physical holdout posterior predictive mean and std
        # generate the dictionaries of holdout results for normalized and physical space
        holdout_results_normalized = {
            "x": x_holdout_norm,
            "y": y_holdout_norm,
            "y_holdout_post_mean": y_holdout_post_mean,
            "y_holdout_post_var": y_holdout_post_var,
            "y_holdout_post_std": y_holdout_post_std,
            "kappa_holdout": kappa_holdout,
            "delta_holdout": delta_holdout
        }

        holdout_results_physical = {
            "x": x_holdout_realspace,
            "y": y_holdout_realspace,
            "y_holdout_post_mean": y_holdout_post_mean * y_sd + y_mu,
            "y_holdout_post_var": y_holdout_post_var * y_sd**2,
            "y_holdout_post_std": y_holdout_post_std * y_sd,
            "kappa_holdout": kappa_holdout * (theta_max - theta_min),  # scale to physical units
            "delta_holdout": delta_holdout * y_sd  # scale to physical units
        }



    # ============================================================
    # 9. Plot Posterior Predictive Fit (normalized and physical) and other diagnostics
    # ============================================================

    if figure_options['PCA_plot']:
        print(f'Generating PCA plot...')
        plot_pca_diagnostics(S,W,x_obs,kappa_theta_chain,theta_labels,max_modes=dtheta, figure_name="PCA_diagnostics.png", figure_directory=figure_path)

    if figure_options['acceptance_trajectory']:
        print(f'Generating kappa acceptance trajectory figure...')
        plot_delta_acceptance_trajectory(accept_trace, calibration_settings['burn_in'], figure_path)

    if figure_options['step_size_plot']:
        print(f'Generating kappa jump size figure...')
        plot_delta_jump_sizes(kappa_theta_chain, figure_path)

    # print a note determining if the results will be compared against known ground truth based on the cross-validation settings in the config file
    if cross_validation_settings["conduct_cross_validation"]:
        print(f"Cross-validation enabled. Posterior predictive checks will be compared against known ground truth.")

        if cross_validation_settings["known_theta_form"] is not None:
            known_theta_form = cross_validation_settings["known_theta_form"]
            print(f"Known theta form: {known_theta_form}")
            known_theta_params = cross_validation_settings["known_theta_form_params"][known_theta_form]
            if known_theta_form == "constant":
                known_theta_values = known_theta_params["values"]
                print(f"Known theta values: {known_theta_values}")
            if known_theta_form == "linear":
                known_theta_slopes = known_theta_params["m"]
                print(f"Known theta slopes: {known_theta_slopes}")
                known_theta_intercepts = known_theta_params["b"]
                print(f"Known theta intercepts: {known_theta_intercepts}")
            if known_theta_form == "trig_funct":
                known_theta_functions = known_theta_params["functions"]
                print(f"Known theta functions: {known_theta_functions}")

        if cross_validation_settings["known_delta_form"] is not None:
            known_delta_form = cross_validation_settings["known_delta_form"]
            print(f"Known delta form: {known_delta_form}")
            known_delta_params = cross_validation_settings["known_delta_form_params"][known_delta_form]
            print(f"Known delta parameters: {known_delta_params}")


    if figure_options['posterior_predict_physical']:
        print(f'Generating posterior predictive check figure (physical units)...')
        # y_sim_phys = model_data["y_sim_y"].values
        # y_sd = model_normalized["y_sim_y_std"].iloc[0]

        # reconduct the prior in real-space (reverse-normalize the emulator predictions) for visualization purposes
        x_col = [col for col in model_normalized.columns if col.startswith('x_')][0]
        x_label = x_col.replace('x_', '')
        x_obs_phys = np.column_stack([
            results_physical[f"x_obs_{label}"].values
            for label in app_labels
        ])
        x_obs_norm = x_obs.copy()
        # Z_obs=np.hstack([x_obs_norm.reshape(-1,1), 
                        #  np.tile(theta_fixed_arr, 
                        #          (x_obs_norm.shape[0], 1))])
        theta_settings = calibration_settings.get("theta_settings", {})
        theta_init = theta_settings.get("theta_initialization", "fixed")

        if theta_init == "fixed":

            theta_block = np.tile(theta_fixed_arr, (len(x_obs_norm), 1))

        elif theta_init == "compositional":

            theta_block = theta_fixed

        else:
            raise ValueError(f"Unknown theta initialization: {theta_init}")

        Z_obs = np.hstack([
            x_obs_norm,
            theta_block
        ])
        y_prior_mean_norm = gp_eta.predict(Z_obs)
        y_prior_mean_phys = y_prior_mean_norm * y_sd + y_mu
        y_prior_var_norm = np.diag(gp_eta.predict(Z_obs, return_cov=True)[1])
        y_prior_var_phys = y_prior_var_norm * (y_sd ** 2)

        plot_discrepancy_diagnostics(
            x_obs_phys,
            results_physical[f'zeta_obs_{y_label}'].values,
            model_data[[col for col in model_data.columns if col.startswith('x')]],
            model_data[[col for col in model_data.columns if col.startswith('zeta')]],
            y_prior_mean_phys,  # reverse-normalize prior mean
            y_prior_var_phys,  # reverse-normalize prior variance
            results_physical['delta_eta_mean'].values,
            results_physical['delta_eta_std'].values,
            results_physical['y_post_mean'].values,
            results_physical['y_post_std'].values ** 2,
            theta_fixed,
            theta_fixed_phys_arr,
            gp_eta,
            results_physical[[f"kappa_{k}_mean" for k in range(dtheta)]].values.T,
            results_physical[[f"kappa_{k}_std"  for k in range(dtheta)]].values.T,
            idx,
            dtheta,
            calibration_settings,
            cross_validation_settings,
            holdout_data=holdout_results_physical if cross_validation_settings.get("holdout_data", False) else None,
            figure_name="posterior_predictive_check_physical.png",
            figure_path=figure_path,
            suptitle='Posterior Predictive Check (Physical Units)',
            plot_settings=plot_settings,
            theta_labels=theta_labels,
            theta_prior_min=theta_prior_min_phys,
            theta_prior_max=theta_prior_max_phys,
            theta_fixed_plot=theta_fixed_phys_arr,
        )


    if figure_options['posterior_predict_normalized']:
        print(f'Generating posterior predictive check figure (normalized)...')

        #diagnostic print
        print("theta_fixed_arr:", theta_fixed_arr)
        print("np.shape(theta_fixed_arr):", np.shape(theta_fixed_arr))
        print("theta_fixed.shape:", np.shape(theta_fixed))


        theta_block_norm = (
            np.tile(theta_fixed, (len(x_obs), 1))
            if theta_fixed.ndim == 1 else theta_fixed
        )
        y_prior_obs_norm = gp_eta.predict(np.hstack([x_obs, theta_block_norm]))
        y_prior_obs_var_norm = np.diag(
            gp_eta.predict(np.hstack([x_obs, theta_block_norm]), return_cov=True)[1]
        )

        plot_discrepancy_diagnostics(
            x_obs,
            results_normalized['zeta_obs'].values,
            model_normalized[[col for col in model_normalized.columns if col.startswith('x')]],
            model_normalized[[col for col in model_normalized.columns if col.startswith('zeta')]],
            y_prior_obs_norm,
            y_prior_obs_var_norm,
            results_normalized['delta_eta_mean'].values,
            results_normalized['delta_eta_std'].values,
            results_normalized['y_post_mean'].values,
            results_normalized['y_post_std'].values ** 2,
            theta_fixed,
            theta_fixed_phys_arr,
            gp_eta,
            kappa_mean,
            kappa_std,
            idx,
            dtheta,
            calibration_settings,
            {**cross_validation_settings, "conduct_cross_validation": False},
            holdout_data=holdout_results_normalized if cross_validation_settings.get("holdout_data", False) else None,
            figure_path=figure_path,
            figure_name="posterior_predictive_check_normalized.png",
            suptitle='Posterior Predictive Check (Normalized Space)',
            plot_settings=plot_settings,
            theta_labels=theta_labels,
            theta_prior_min=np.zeros(dtheta),
            theta_prior_max=np.ones(dtheta),
            theta_fixed_plot=theta_fixed,
        )

    if figure_options['variance_decomposition']:

        print("Generating variance decomposition diagnostics...")

        var_results = discrepancy_variance_decomposition(
            x_obs=x_obs,
            y_obs=y_obs,
            theta_fixed=theta_fixed,
            kappa_theta_chain=kappa_theta_chain,
            delta_eta_chain=delta_eta_chain,
            kappa_mean=kappa_mean,
            delta_eta_mean=delta_eta_mean,
            gp_eta=gp_eta,
            x_obs_input=x_obs,
            Nsamp=calibration_settings['posterior_predictive_samples'],
            plot=True,
            figure_path=figure_path,
            save_name="variance_decomposition.png",
            suptitle="Variance Decomposition of Discrepancy"
        )

    if figure_options['relative_contributions']:

        print("Generating relative contribution diagnostics...")

        contrib = compute_relative_contributions(
            x_obs=x_obs,
            theta_fixed=theta_fixed,
            kappa_theta_chain=kappa_theta_chain,
            delta_eta_chain=delta_eta_chain,
            gp_eta=gp_eta,
            Nsamp=calibration_settings['posterior_predictive_samples']
        )

        plot_relative_contributions(
            x=x_obs_phys,
            contrib=contrib,
            figure_path=figure_path,
            save_name="relative_contributions.png",
            suptitle="Relative Contributions of κ and δη"
        )

    with open(f"{results_path}/used_config.json", "w") as f:
        json.dump(config, f, indent=2)


    # If cross-validation is enabled, compute the loss between:
    # 1. The known ground truth of y vs. the posterior mean
    # 2. The known ground truth of theta vs. the posterior mean of theta (theta_fixed + kappa_mean)
    # 3. The known ground truth of delta vs. the posterior mean of delta_eta
    # Print the results to a loss_output.json file and plot the loss convergence across MCMC iterations if the option is enabled in the config file
    # ============================================================
    # Cross-validation metrics (physical units)
    # ============================================================

    if cross_validation_settings["conduct_cross_validation"]:

        print("Computing cross-validation losses (physical space)...")

        # ========================================================
        # Known theta form setup
        # ========================================================

        known_theta_form = cross_validation_settings[
            "known_theta_form"
        ]

        known_theta_params = (
            cross_validation_settings[
                "known_theta_form_params"
            ][known_theta_form]
        )

        # ========================================================
        # 1. Posterior predictive y metrics
        # ========================================================

        y_true = obs_data[
            [
                col for col in obs_data.columns
                if col.startswith('zeta_')
            ]
        ].values.flatten()

        y_pred = results_physical[
            'y_post_mean'
        ].values.flatten()

        y_mse = np.mean(
            (y_true - y_pred) ** 2
        )

        y_rmse = np.sqrt(y_mse)

        y_range = (
            np.max(y_true)
            - np.min(y_true)
        )

        y_nrmse = y_rmse / (y_range + 1e-12)

        print(f"y MSE:   {y_mse:.6f}")
        print(f"y NRMSE:    {y_nrmse:.6f}")

        # ========================================================
        # 2. Theta field metrics
        # ========================================================

        theta_mse = {}
        theta_nrmse = {}

        for k in range(dtheta):

            # ----------------------------------------------------
            # Build true theta field
            # ----------------------------------------------------

            if known_theta_form == "constant":

                theta_true_k = (
                    np.ones(No)
                    * known_theta_params["values"][k]
                )

            elif known_theta_form == "trig_funct":

                func = known_theta_params[
                    "functions"
                ][k]

                x_vals = x_obs_phys[:, 0]

                if func == "sin":
                    theta_true_k = np.sin(x_vals)

                elif func == "cos":
                    theta_true_k = np.cos(x_vals)

                else:
                    raise ValueError(
                        f"Unknown trig function: {func}"
                    )
                
            elif known_theta_form == "linear":

                m = known_theta_params["m"][k]
                b = known_theta_params["b"][k]

                x_vals = x_obs_phys[:, 0]

                theta_true_k = m * x_vals + b

            else:
                raise ValueError(
                    f"Unknown known_theta_form: "
                    f"{known_theta_form}"
                )

            # ----------------------------------------------------
            # Predicted theta field
            # ----------------------------------------------------

            theta_settings = calibration_settings.get("theta_settings", {})
            theta_init = theta_settings.get("theta_initialization", "fixed")

            if theta_init == "fixed":

                theta_pred_k = (
                    theta_fixed[k]
                    + kappa_mean[k, :]
                )

            elif theta_init == "compositional":

                theta_pred_k = (
                    theta_fixed[:, k]
                    + kappa_mean[k, :]
                )

            else:
                raise ValueError(f"Unknown theta initialization: {theta_init}")

            # ----------------------------------------------------
            # MSE
            # ----------------------------------------------------

            mse_k = np.mean(
                (theta_true_k - theta_pred_k) ** 2
            )

            rmse_k = np.sqrt(mse_k)

            theta_range_k = (
                np.max(theta_true_k)
                - np.min(theta_true_k)
            )

            if theta_range_k < 1e-12:
                theta_range_k = 1.0

            nrmse_k = rmse_k / theta_range_k

            theta_mse[f"theta_{k}"] = float(mse_k)

            theta_nrmse[f"theta_{k}"] = float(nrmse_k)

            print(
                f"Theta {k}: "
                f"MSE={mse_k:.6f}, "
                f"NRMSE={nrmse_k:.6f}"
            )

        theta_mse_total = np.mean(
            list(theta_mse.values())
        )

        theta_nrmse_total = np.mean(
            list(theta_nrmse.values())
        )

        # ========================================================
        # 3. Additive discrepancy metrics
        # ========================================================

        if cross_validation_settings[
            "known_delta_form"
        ] is not None:

            known_delta_form = (
                cross_validation_settings[
                    "known_delta_form"
                ]
            )

            known_delta_params = (
                cross_validation_settings[
                    "known_delta_form_params"
                ][known_delta_form]
            )

            x_val = x_obs_phys[:, 0]

            if known_delta_form == "power_law":

                coeff = known_delta_params["coeff"]

                exponent = known_delta_params[
                    "exponent"
                ]

                delta_true = (
                    coeff
                    * (np.abs(x_val) ** exponent)
                )

            elif known_delta_form == "trig_funct":

                functions = known_delta_params[
                    "function"
                ]

                delta_true = np.zeros_like(x_val)

                for func in functions:

                    if func == "sin":
                        delta_true += np.sin(x_val)

                    elif func == "cos":
                        delta_true += np.cos(x_val)

            elif known_delta_form == "linear":

                m = known_delta_params["m"]

                b = known_delta_params["b"]

                delta_true = m * x_val + b

            elif known_delta_form == "polynomial":

                coeffs = known_delta_params["coeffs"]

                delta_true = np.polyval(coeffs, x_val)

            else:
                raise ValueError(
                    f"Unknown known_delta_form: "
                    f"{known_delta_form}"
                )
            

            # ----------------------------------------------------
            # Convert delta prediction to physical units
            # ----------------------------------------------------

            delta_pred_norm = (
                results_normalized[
                    'delta_eta_mean'
                ]
                .values
                .flatten()
            )

            delta_pred_phys = (
                delta_pred_norm * y_std
                + y_mean
            )

            delta_mse = np.mean(
                (delta_true - delta_pred_phys) ** 2
            )

            delta_rmse = np.sqrt(delta_mse)

            delta_range = (
                np.max(delta_true)
                - np.min(delta_true)
            )

            if delta_range < 1e-12:
                delta_range = 1.0

            delta_nrmse = (
                delta_rmse / delta_range
            )

            print(
                f"Delta: "
                f"MSE={delta_mse:.6f}, "
                f"NRMSE={delta_nrmse:.6f}"
            )

        else:

            delta_mse = None
            delta_nrmse = None

        # ========================================================
        # Net metrics
        # ========================================================

        net_loss = y_mse + theta_mse_total

        net_nrmse = (
            y_nrmse
            + theta_nrmse_total
        )

        if delta_mse is not None:
            net_loss += delta_mse

        if delta_nrmse is not None:
            net_nrmse += delta_nrmse


        # ========================================================
        # Holdout metrics
        # ========================================================

        if cross_validation_settings.get("holdout_data", False):

            y_holdout_true = holdout_results_physical['y'].flatten()

            y_holdout_pred = (
                holdout_results_physical['y_holdout_post_mean'].flatten()
            )

            y_holdout_mse = np.mean(
                (y_holdout_true - y_holdout_pred) ** 2
            )

            y_holdout_rmse = np.sqrt(y_holdout_mse)

            y_holdout_range = (
                np.max(y_holdout_true)
                - np.min(y_holdout_true)
            )

            if y_holdout_range < 1e-12:
                y_holdout_range = 1.0

            y_holdout_nrmse = (
                y_holdout_rmse / y_holdout_range
            )

            print(
                f"Holdout y: "
                f"MSE={y_holdout_mse:.6f}, "
                f"NRMSE={y_holdout_nrmse:.6f}"
            )
        # ========================================================
        # Save JSON
        # ========================================================

        loss_dict = {

            "y_mse": float(y_mse),
            "y_nrmse": float(y_nrmse),

            "theta_mse": theta_mse,
            "theta_nrmse": theta_nrmse,

            "theta_mse_total": float(
                theta_mse_total
            ),

            "theta_nrmse_total": float(
                theta_nrmse_total
            ),

            "delta_mse": (
                float(delta_mse)
                if delta_mse is not None
                else None
            ),

            "delta_nrmse": (
                float(delta_nrmse)
                if delta_nrmse is not None
                else None
            ),

            "net_loss": float(net_loss),
            "net_nrmse": float(net_nrmse),


            "holdout_y_mse": (
                float(y_holdout_mse)
                if cross_validation_settings.get(
                    "holdout_data", False
                )
                else None
            ),

            "holdout_y_nrmse": (
                float(y_holdout_nrmse)
                if cross_validation_settings.get(
                    "holdout_data", False
                )
                else None
            )
        }

        with open(
            os.path.join(
                results_path,
                "cross_validation_losses.json"
            ),
            "w"
        ) as f:

            json.dump(loss_dict, f, indent=4)

        print(
            f"Saved cross-validation losses to "
            f"{os.path.join(results_path, 'cross_validation_losses.json')}"
        )

    return()

if __name__ == "__main__":
    main()
