import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve
from scipy.stats import multivariate_normal, norm
from scipy.stats import invgamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from functs import *
from plottingFuncts import *
import json

def main():

    if len(sys.argv) < 2:
        raise ValueError("Usage: python main.py <config_path>")

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.load(f)

    # store calibration settings
    calibration_settings = config["calibration_settings"]

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
        generate_rawData_figure(model_data, obs_data, figure_path, figure_name="raw_data_realspace.png", suptitle="Raw Data (Physical Units)")

    if figure_options['data_priors_normalized']:
        print(f'Generating raw data figures in normalized space...')
        generate_rawData_figure(model_normalized, observation_normalized, figure_path, figure_name="raw_data_normalized.png", suptitle="Raw Data (Normalized Space)")

    x_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]].values.reshape(-1,1)
    theta_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('theta_')]].values
    y_sim = model_normalized[[col for col in model_normalized.columns if col.startswith('zeta_') and not col.endswith('_mean') and not col.endswith('_std')]].values

    x_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('x_')]].values.reshape(-1,1)
    y_obs = observation_normalized[[col for col in observation_normalized.columns if col.startswith('zeta_')]].values

    print("Simulator design shape:", x_sim.shape, theta_sim.shape)
    print("Observation shape:", x_obs.shape, y_obs.shape)

    dtheta = theta_sim.shape[1]
    No = len(x_obs)

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

    # save the following prior information: y_prior_mean,y_prior_var,delta_eta_mean,delta_eta_std,
    y_prior_mean = gp_eta.predict(Z_sim)
    y_prior_var = np.diag(gp_eta.predict(Z_sim, return_cov=True)[1])

    # if output_settings['eta_emulator_prior'] is true, run plot_emulator_prior(x_sim, y_sim, y_prior_mean, y_prior_var, model_data, model_normalized, output_settings['figures_path'], results_path)
    if figure_options['eta_emulator_prior']:
        print(f'Generating emulator prior figure...')
        plot_emulator_prior(y_prior_mean, y_prior_var, model_data, model_normalized, reverse_normalization, figure_path, results_path)

    # intialize storage for deta eta
    delta_eta_mean = np.zeros(No)
    delta_eta_std = np.zeros(No)

    # ============================================================
    # 3. Initialize θ (fixed plug-in estimate) and conduct PCA
    # ============================================================
    theta_fixed_arr =[]
    theta_fixed_phys_arr = []

    if calibration_settings['theta_initialization'] == 'fixed':
        theta_fixed = theta_sim.mean(axis=0)
        theta_fixed_arr.append(theta_fixed)

        print("\nFixed theta used (normalized):")
        print(theta_fixed)
        print("\nFixed theta used (physical units):")
        for i, label in enumerate(theta_labels):
            # print(f"{label} (normalized) = {theta_fixed[i]:.3f}")
            min_val = model_data[f'theta_{label}'].min()
            max_val = model_data[f'theta_{label}'].max()
            theta_fixed_phys = theta_fixed[i] * (max_val - min_val) + min_val
            print(f"{label}: {theta_fixed_phys}")
            theta_fixed_phys_arr.append(theta_fixed_phys)
    else:
        raise NotImplementedError("Only fixed theta initialization is currently implemented.")
    
    # PCA implementation
    
    # compute jacobian across all x
    J_all = np.array([compute_jacobian(x_obs[i], theta_fixed, gp_eta)
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
    ell_kappa = calibration_settings['kappa_prior_vals']['ell'] * np.ones(dtheta)
    # var_delta = 0.05 * np.ones(dtheta)
    var_kappa = calibration_settings['kappa_prior_vals']['var'] * np.ones(dtheta) # start with smaller variance to encourage more conservative initial discrepancy fields, which can help stabilize early sampling

    # Initial noise variance
    sigma2 = calibration_settings['sigma2_prior_val']**2


    # ============================================================
    # 4b. Initialize additive discrepancy δ_eta(x)
    # ============================================================

    delta_eta = np.zeros(No)

    # hyperparameters for δ_eta GP
    ell_eta = calibration_settings['delta_eta_prior_vals']['ell']
    var_eta = calibration_settings['delta_eta_prior_vals']['var']

    # ============================================================
    # 5. Sampler config
    # ============================================================
    Nmcmc = calibration_settings['N_mcmc']
    mh_scale = calibration_settings['mh_scale_kappa_prior']
    mh_scale_kappa=np.ones(dtheta) * mh_scale

    kappa_prior_ell=calibration_settings['kappa_priors']['ell']
    kappa_prior_var=calibration_settings['kappa_priors']['var']

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

        # ---- update δₖ fields ----
        for k in range(dtheta):
            kappa_z, acc = mh_update_delta_k(
                k,
                kappa_z,
                delta_eta,
                theta_fixed,
                ell_kappa[k],
                var_kappa[k],
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
        for k in range(dtheta):
            ell_kappa[k], var_kappa[k], _ = mh_update_delta_hyperparams(
                kappa_z[k],
                ell_kappa[k],
                var_kappa[k],
                x_obs,
                kappa_prior_ell,
                kappa_prior_var,
                mh_scales,
                calibration_settings['allow_singular_cov']
            )

        # ---- compute sensitivities ---- (@ theta fixed)
        G=compute_sensitivities(x_obs, theta_fixed, gp_eta, orthogonalization_settings['theta_epsilon'])

        # ---- update δ_eta_raw (Gibbs) ----
        delta_eta_raw = gibbs_delta_eta(
            x_obs,
            y_obs,
            theta_fixed,
            kappa_theta,
            gp_eta,
            sigma2,
            ell_eta,
            var_eta
        )

        # orthogonalize delta_eta with respect to the sensitivities G

        if orthogonalization_settings['orthogonalize_delta_eta']:
            delta_eta=orthogonalize_delta_eta(delta_eta_raw, G)
        else:
            delta_eta = delta_eta_raw

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

        # ---- store ----
        kappa_theta_chain[it] = kappa_theta
        theta_star_chain[it] = theta_fixed[:, None] + kappa_theta
        delta_eta_chain[it] = delta_eta
        sigma2_chain[it] = sigma2

        # ---- diagnostics ----
        if (it+1) % 50 == 0:
            print(f"Iter {it+1}/{Nmcmc}")
            print(" sigma2 =", sigma2)
            print(" kappa acceptance rates:",
                accept_kappa / (it+1))
            print(" ell_kappa:", ell_kappa)
            print(" var_kappa:", var_kappa)
            if orthogonalization_settings['orthogonalize_delta_eta']:
                proj = G @ np.linalg.solve(G.T @ G, G.T @ delta_eta)
                print("Projection norm (should be near 0):", np.linalg.norm(proj))
                

            print("--------------------------------------------------")


    print("\nSampler complete.")

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
            theta_star = theta_fixed + kappa_theta_s
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
        "x_obs": x_obs.flatten(),
        "zeta_obs": y_obs.flatten(),
        # "x_sim": model_normalized[[col for col in model_normalized.columns if col.startswith('x_')]].values.flatten(),
        # "xi_sim": model_normalized[[col for col in model_normalized.columns if col.startswith('y_') and not col.endswith('_mean') and not col.endswith('_std')]].values.flatten(),
        "y_post_mean": y_post_mean,
        "y_post_std": np.sqrt(y_post_var),
        "delta_eta_mean": delta_eta_mean,
        "delta_eta_std": delta_eta_std,
    }
    
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
        x_phys = results_normalized["x_obs"].values * (x_max - x_min) + x_min
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

    if results_options['output_normalized_results']:
        print(f'Saving normalized results to {results_path}...')
        np.savez(os.path.join(results_path, "results_normalized.npz"), **results_normalized)
        print(f"Saved normalized results to {os.path.join(results_path, 'results_normalized.npz')}")

    if results_options['output_physical_results']:
        print(f'Saving physical results to {results_path}...')
        np.savez(os.path.join(results_path, "results_physical.npz"), **results_physical)
        print(f"Saved physical results to {os.path.join(results_path, 'results_physical.npz')}")



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
        x_obs_phys = results_physical[f'x_obs_{x_label}'].values
        x_min = reverse_normalization["x"][x_col]["min"]
        x_max = reverse_normalization["x"][x_col]["max"]
        x_obs_norm = (x_obs_phys - x_min) / (x_max - x_min)
        Z_obs=np.hstack([x_obs_norm.reshape(-1,1), 
                         np.tile(theta_fixed_arr, 
                                 (x_obs_norm.shape[0], 1))])
        y_prior_mean_norm = gp_eta.predict(Z_obs)
        y_prior_mean_phys = y_prior_mean_norm * y_sd + y_mu
        y_prior_var_norm = np.diag(gp_eta.predict(Z_obs, return_cov=True)[1])
        y_prior_var_phys = y_prior_var_norm * (y_sd ** 2)

        plot_discrepancy_diagnostics(
            results_physical[f'x_obs_{x_label}'].values,
            results_physical[f'zeta_obs_{y_label}'].values,
            model_data[[col for col in model_data.columns if col.startswith('x')]],
            model_data[[col for col in model_data.columns if col.startswith('zeta')]],
            y_prior_mean_phys,  # reverse-normalize prior mean
            y_prior_var_phys,  # reverse-normalize prior variance
            results_physical['delta_eta_mean'].values,
            results_physical['delta_eta_std'].values,
            results_physical['y_post_mean'].values,
            results_physical['y_post_std'].values ** 2,
            theta_fixed_arr,
            theta_fixed_phys_arr,
            gp_eta,
            results_physical[[f"kappa_{k}_mean" for k in range(dtheta)]].values.T,
            results_physical[[f"kappa_{k}_std"  for k in range(dtheta)]].values.T,
            idx,
            dtheta,
            cross_validation_settings,
            figure_path,
            figure_name="posterior_predictive_check_physical.png",
            suptitle='Posterior Predictive Check (Physical Units)'
        )


    if figure_options['posterior_predict_normalized']:
        print(f'Generating posterior predictive check figure (normalized)...')
        plot_discrepancy_diagnostics(
            results_normalized['x_obs'].values,
            results_normalized['zeta_obs'].values,
            model_normalized[[col for col in model_normalized.columns if col.startswith('x')]],
            model_normalized[[col for col in model_normalized.columns if col.startswith('zeta')]],
            y_prior_mean,
            y_prior_var,
            results_normalized['delta_eta_mean'].values,
            results_normalized['delta_eta_std'].values,
            results_normalized['y_post_mean'].values,
            results_normalized['y_post_std'].values ** 2,
            theta_fixed_arr,
            theta_fixed_phys_arr,
            gp_eta,
            kappa_mean,
            kappa_std,
            idx,
            dtheta,
            {**cross_validation_settings, "conduct_cross_validation": False},
            figure_path,
            figure_name="posterior_predictive_check_normalized.png",
            suptitle='Posterior Predictive Check (Normalized Space)'
        )

   

        with open(f"{results_path}/used_config.json", "w") as f:
            json.dump(config, f, indent=2)


    # If cross-validation is enabled, compute the loss between:
    # 1. The known ground truth of y vs. the posterior mean
    # 2. The known ground truth of theta vs. the posterior mean of theta (theta_fixed + kappa_mean)
    # 3. The known ground truth of delta vs. the posterior mean of delta_eta
    # Print the results to a loss_output.json file and plot the loss convergence across MCMC iterations if the option is enabled in the config file
    if cross_validation_settings["conduct_cross_validation"]:
        print(f"Computing cross-validation losses (real space)...")

        # Create arrays for the known forms based on the settings in the config file

        # 1. y loss (e.g. MSE)
        y_true = obs_data[[col for col in obs_data.columns if col.startswith('zeta_')]].values.flatten()
        y_pred = results_physical['y_post_mean'].values
        y_mse = np.mean((y_true - y_pred) ** 2)

        # 2. theta loss (e.g. MSE across all theta dimensions)
        known_theta_form = cross_validation_settings["known_theta_form"]
        known_theta_params = cross_validation_settings["known_theta_form_params"][known_theta_form]
        
        if known_theta_form == "constant":
            theta_true = np.array(known_theta_params["values"])
        elif known_theta_form == "trig_funct":
            # Compute trig function values at first observation
            x_val = x_obs[0, 0]
            functions = known_theta_params["functions"]
            theta_true = np.array([
                np.sin(x_val) if func == "sin" else np.cos(x_val)
                for func in functions
            ])
        else:
            raise ValueError(f"Unknown known_theta_form: {known_theta_form}")
        
        theta_pred = theta_fixed_arr[0] + kappa_mean[:, 0]  # posterior mean of theta at first observation
        theta_mse = np.mean((theta_true - theta_pred) ** 2)

        # 3. delta loss (e.g. MSE)
        if cross_validation_settings["known_delta_form"] is not None:
            known_delta_form = cross_validation_settings["known_delta_form"]
            known_delta_params = cross_validation_settings["known_delta_form_params"][known_delta_form]
            
            if known_delta_form == "power_law":
                coeff = known_delta_params["coeff"]
                exponent = known_delta_params["exponent"]
                delta_true = coeff * (np.abs(x_obs.flatten()) ** exponent)
            elif known_delta_form == "trig_funct":
                x_val = x_obs.flatten()
                functions = known_delta_params["function"]
                delta_true = np.zeros_like(x_val)
                for func in functions:
                    if func == "sin":
                        delta_true += np.sin(x_val)
                    elif func == "cos":
                        delta_true += np.cos(x_val)
            else:
                raise ValueError(f"Unknown known_delta_form: {known_delta_form}")
            
            delta_pred = results_normalized['delta_eta_mean'].values
            delta_mse = np.mean((delta_true - delta_pred) ** 2)
        else:
            delta_mse = None

        # Compute net loss (sum of MSEs)
        net_loss = y_mse + theta_mse
        if delta_mse is not None:
            net_loss += delta_mse

        # Save losses to a json file
        loss_dict = {
            "y_mse": float(y_mse),
            "theta_mse": float(theta_mse),
            "delta_mse": float(delta_mse) if delta_mse is not None else None,
            "net_loss": float(net_loss)
        }
        with open(os.path.join(results_path, "cross_validation_losses.json"), "w") as f:
            json.dump(loss_dict, f, indent=4)
        print(f"Saved cross-validation losses to {os.path.join(results_path, 'cross_validation_losses.json')}")
        print(f"y_mse: {y_mse:.6f}, theta_mse: {theta_mse:.6f}, delta_mse: {delta_mse}, net_loss: {net_loss:.6f}")


    return()

if __name__ == "__main__":
    main()