# Functions for preforming the calibration 
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.stats import multivariate_normal, norm
from scipy.stats import invgamma

def rbf_kernel(X, Y, ell=1.0, var=1.0):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sqdist = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return var * np.exp(-0.5 * sqdist / ell**2)

def log_likelihood(y, x, theta, delta, eta, sigma2):
    """
    y_i ~ N( eta(x_i, theta + delta(x_i)), sigma2 )
    """
    y_pred = np.zeros_like(y)

    for i in range(len(x)):
        theta_star = theta + delta[:, i]
        y_pred[i] = eta(x[i], theta_star)

    resid = y - y_pred
    return -0.5 * (
        np.sum(resid**2) / sigma2
        + len(y) * np.log(2 * np.pi * sigma2)
    )

def log_prior_delta(delta_k, K_delta_inv):
    return -0.5 * delta_k.T @ K_delta_inv @ delta_k

def gp_log_density(delta_k, K):
    """
    Stable log N(0,K) evaluation.
    """
    No = len(delta_k)
    L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, delta_k))

    logdet = 2*np.sum(np.log(np.diag(L)))

    return -0.5 * (delta_k @ alpha + logdet + No*np.log(2*np.pi))


def log_prior_hyperparams(ell, var, prior_ell, prior_var):
    """
    Log-prior for kernel hyperparameters (ell, var).

    prior_ell = {"mu": ..., "sigma": ...} on log(ell)
    prior_var = {"mu": ..., "sigma": ...} on log(var)
    """

    log_ell = np.log(ell)
    log_var = np.log(var)

    lp_ell = norm.logpdf(
        log_ell,
        loc=prior_ell["mu"],
        scale=prior_ell["sigma"]
    )

    lp_var = norm.logpdf(
        log_var,
        loc=prior_var["mu"],
        scale=prior_var["sigma"]
    )

    return lp_ell + lp_var

def mh_update_delta_hyperparams(
    delta_k, ell, var, x_md,
    prior_ell, prior_var,
    mh_scales,allow_singular=True
):
    log_ell_prop = np.log(ell) + mh_scales["log_ell_kappa"] * np.random.randn()
    log_var_prop = np.log(var) + mh_scales["log_var_kappa"] * np.random.randn()

    ell_prop = np.exp(log_ell_prop)
    var_prop = np.exp(log_var_prop)

    K_curr = rbf_kernel(x_md, x_md, ell=ell, var=var)+ 1e-8*np.eye(len(x_md))
    K_prop = rbf_kernel(x_md, x_md, ell=ell_prop, var=var_prop) + 1e-8*np.eye(len(x_md))

    logp_curr = (
        multivariate_normal.logpdf(delta_k, mean=np.zeros(len(delta_k)), cov=K_curr, allow_singular=True)
        + log_prior_hyperparams(ell, var, prior_ell, prior_var)
    )

    logp_prop = (
        multivariate_normal.logpdf(delta_k, mean=np.zeros(len(delta_k)), cov=K_prop, allow_singular=True)
        + log_prior_hyperparams(ell_prop, var_prop, prior_ell, prior_var)
    )

    if np.log(np.random.rand()) < (logp_prop - logp_curr):
        return ell_prop, var_prop, True
    else:
        return ell, var, False
    
def gibbs_sigma2(y_obs, x_obs, theta, delta_theta, delta_eta, gp_eta, a, b):

    resid = np.zeros_like(y_obs)

    for i in range(len(y_obs)):
        theta_i = get_theta_at_obs(theta, i)
        theta_star = theta_i + delta_theta[:, i]
        m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
        resid[i] = y_obs[i] - (m_i + delta_eta[i])

    a_post = a + len(y_obs)/2
    b_post = b + 0.5*np.sum(resid**2)

    return invgamma.rvs(a_post, scale=b_post)

def gibbs_delta_eta(x_obs, y_obs, theta, delta_theta, gp_eta, sigma2, ell_eta, var_eta):

    No = len(x_obs)

    # --- build covariance ---
    K = rbf_kernel(x_obs, x_obs, ell=ell_eta, var=var_eta)
    K += 1e-8 * np.eye(No)

    # --- compute residual ---
    r = np.zeros(No)

    for i in range(No):
        theta_star = theta + delta_theta[:, i]
        m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
        # print(f"m_i: {m_i:.3f}, y_obs[i]: {y_obs[i]}")
        r[i] = y_obs[i][0] - m_i

    # --- posterior ---
    K_inv = np.linalg.inv(K)
    Sigma_post = np.linalg.inv(K_inv + (1/sigma2)*np.eye(No))

    mu_post = Sigma_post @ ((1/sigma2) * r)

    return np.random.multivariate_normal(mu_post, Sigma_post)

def eta_predict(x, theta_star, gp_eta):

    x = np.atleast_1d(np.asarray(x))
    theta_star = np.atleast_1d(np.asarray(theta_star))

    z = np.hstack([x, theta_star]).reshape(1, -1)

    m, s2 = gp_eta.predict(z, return_std=True)

    return m[0], s2[0]**2


def log_likelihood_embedded(y_obs, x_obs, theta, delta_theta,delta_eta, gp_eta, sigma2):
    """
    y_i ~ N( m_i , sigma2 + s_i^2 )
    where emulator provides (m_i, s_i^2)
    """
    N = len(y_obs)
    loglike = 0.0

    for i in range(N):

        theta_i = get_theta_at_obs(theta, i)
        theta_star = theta_i + delta_theta[:, i]
        # m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)

        m_i, s2_i = eta_predict(
            x_obs[i],
            theta_star,
            gp_eta
        )

        total_var = sigma2 + s2_i

        resid = y_obs[i] - (m_i + delta_eta[i])

        loglike += -0.5 * (
            np.log(2*np.pi*total_var)
            + resid**2 / total_var
        )

    return loglike

# def mh_update_delta_k(
#     k, delta_theta, delta_eta, theta,
#     ell_k, var_k,
#     x_obs, y_obs,
#     gp_eta,
#     sigma2,
#     mh_scale
# ):
#     No = len(x_obs)

#     # --- GP prior covariance ---
#     K = rbf_kernel(x_obs, x_obs, ell=ell_k, var=var_k) + 1e-8*np.eye(No)
#     L = np.linalg.cholesky(K)

#     # --- proposal ---
#     proposal = delta_theta[k] + mh_scale * (L @ np.random.randn(No))

#     delta_prop = delta_theta.copy()
#     delta_prop[k] = proposal

#     # --- log posterior current ---
#     logpost_curr = (
#         log_likelihood_embedded(y_obs, x_obs, theta, delta_theta, delta_eta, gp_eta, sigma2)
#         + gp_log_density(delta_theta[k], K)
#     )

#     # --- log posterior proposed ---
#     logpost_prop = (
#         log_likelihood_embedded(y_obs, x_obs, theta, delta_prop, delta_eta, gp_eta, sigma2)
#         + gp_log_density(proposal, K)
#     )

#     # print("mean proposal jump:", np.linalg.norm(delta_prop - delta))

#     log_alpha = logpost_prop - logpost_curr

#     # print(f"log posterior current: {logpost_curr:.3f}, proposed: {logpost_prop:.3f}, log alpha: {log_alpha:.3f}")

#     if np.log(np.random.rand()) < log_alpha:
#         delta_theta[k] = proposal
#         return delta_theta, True
#     else:
#         return delta_theta, False
    

# orthogonalization functions
# def compute_sensitivities(x_obs, theta_fixed, gp_eta, eps=1e-2):
#     No = len(x_obs)
#     dtheta = len(theta_fixed)

#     G = np.zeros((No, dtheta))

#     for i, x in enumerate(x_obs):
#         for k in range(dtheta):

#             theta_plus = theta_fixed.copy()
#             theta_minus = theta_fixed.copy()

#             theta_plus[k] += eps
#             theta_minus[k] -= eps

#             m_plus, _ = eta_predict(x, theta_plus, gp_eta)
#             m_minus, _ = eta_predict(x, theta_minus, gp_eta)

#             G[i, k] = (m_plus - m_minus) / (2 * eps)

#     return G

def compute_sensitivities(x_obs, theta_fixed, gp_eta, eps=1e-2):
    """
    Compute dη/dθ at each observation.

    Parameters
    ----------
    theta_fixed : ndarray
        Either
            (dtheta,)      for fixed initialization
        or
            (No, dtheta)   for compositional initialization.
    """

    No = len(x_obs)

    theta_fixed = np.asarray(theta_fixed)

    if theta_fixed.ndim == 1:
        dtheta = len(theta_fixed)
    else:
        dtheta = theta_fixed.shape[1]

    G = np.zeros((No, dtheta))

    for i, x in enumerate(x_obs):

        theta_i = get_theta_at_obs(theta_fixed, i)

        for k in range(dtheta):

            theta_plus = theta_i.copy()
            theta_minus = theta_i.copy()

            theta_plus[k] += eps
            theta_minus[k] -= eps

            m_plus, _ = eta_predict(x, theta_plus, gp_eta)
            m_minus, _ = eta_predict(x, theta_minus, gp_eta)

            G[i, k] = (m_plus - m_minus) / (2 * eps)

    return G

def orthogonalize_delta_eta(delta_eta, G, jitter=1e-8):
    # G: (No, dtheta)
    # delta_eta: (No,)

    GTG = G.T @ G + jitter * np.eye(G.shape[1])
    P = G @ np.linalg.solve(GTG, G.T)

    delta_eta_orth = delta_eta - P @ delta_eta

    return delta_eta_orth

def compute_jacobian(x, theta, gp, eps=1e-4):
    dtheta = len(theta)
    J = np.zeros(dtheta)

    base, _ = eta_predict(x, theta, gp)

    for k in range(dtheta):
        theta_perturb = theta.copy()
        theta_perturb[k] += eps

        pert, _ = eta_predict(x, theta_perturb, gp)
        J[k] = (pert - base) / eps

    return J

def compute_relative_contributions(
    x_obs,
    theta_fixed,
    kappa_theta_chain,
    delta_eta_chain,
    gp_eta,
    Nsamp=200
):
    """
    Computes additive contribution decomposition:

        y = η(x, θ_fixed)
            + Δη_κ
            + δη

    where:
        Δη_κ = η(x, θ_fixed + κ) - η(x, θ_fixed)

    Returns posterior means/stds for each contribution.
    """

    Nmcmc, dtheta, No = kappa_theta_chain.shape

    idx = np.random.choice(Nmcmc, Nsamp, replace=False)

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    base = np.zeros((Nsamp, No))
    kappa_effect = np.zeros((Nsamp, No))
    eta_effect = np.zeros((Nsamp, No))
    full = np.zeros((Nsamp, No))

    for s_i, s in enumerate(idx):

        for i in range(No):

            # ----------------------------------------------------
            # Baseline emulator
            # ----------------------------------------------------
            m_base, _ = eta_predict(
                x_obs[i],
                get_theta_at_obs(theta_fixed, i),
                gp_eta
            )

            # ----------------------------------------------------
            # κ-shifted emulator
            # ----------------------------------------------------
            theta_star = (
                get_theta_at_obs(theta_fixed, i)
                + kappa_theta_chain[s, :, i]
            )

            m_theta, _ = eta_predict(
                x_obs[i],
                theta_star,
                gp_eta
            )

            # ----------------------------------------------------
            # Contributions
            # ----------------------------------------------------
            base[s_i, i] = m_base

            # κ-induced change
            kappa_effect[s_i, i] = (
                m_theta - m_base
            )

            # additive discrepancy
            eta_effect[s_i, i] = (
                delta_eta_chain[s, i]
            )

            # total prediction
            full[s_i, i] = (
                m_theta
                + delta_eta_chain[s, i]
            )

    return {

        "base_mean": base.mean(axis=0),
        "base_std": base.std(axis=0),

        "kappa_mean": kappa_effect.mean(axis=0),
        "kappa_std": kappa_effect.std(axis=0),

        "eta_mean": eta_effect.mean(axis=0),
        "eta_std": eta_effect.std(axis=0),

        "full_mean": full.mean(axis=0),
        "full_std": full.std(axis=0),
    }

def get_theta_at_obs(theta_fixed, i):
    """
    Returns the normalized theta vector for observation i.

    Parameters
    ----------
    theta_fixed : ndarray
        Either shape (dtheta,) for fixed calibration or
        (No, dtheta) for compositional calibration.
    i : int

    Returns
    -------
    ndarray
        Shape (dtheta,)
    """
    theta_fixed = np.asarray(theta_fixed)

    if theta_fixed.ndim == 1:
        return theta_fixed

    return theta_fixed[i]