import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree
from functs import *

class ShrinkagePrior:

    name="shrinkage_base"

    def __init__(self, config):
        self.config = config

    def log_prior(self, z, x=None, k=None):
        raise NotImplementedError
    
    def sample_kappa_field(self, k, kappa_theta, delta_eta, theta, x_obs, y_obs, gp_eta, sigma2, mh_scale):
        raise NotImplementedError

    def sample_delta_eta_field(self, delta_eta, x_obs, y_obs, theta, kappa_theta, gp_eta, sigma2):
        raise NotImplementedError

    def update_hyperparameters(self, *args):
        pass

    def get_state(self):
        return {}

    
class GPPrior(ShrinkagePrior):

    name = "gp"

    def __init__(self, config):

        super().__init__(config)

        self.ell = config["ell_init"]
        self.var = config["var_init"]

        self.ell_prior = config["ell_prior"]
        self.var_prior = config["var_prior"]

    # def log_prior(self, z, x=None, config=None):

    #     K = build_covariance(
    #         x,
    #         self.ell,
    #         self.var
    #     )

    #     return multivariate_normal.logpdf(
    #         z,
    #         np.zeros(len(z)),
    #         K,
    #         allow_singular=True
    #     )
    # def log_prior(self, z, x=None):
    #     """
    #     Stable log N(0, K) prior using Cholesky decomposition (matches gp_log_density).
    #     """
    #     z = np.asarray(z)

    #     K = build_covariance(
    #         x,
    #         self.ell,
    #         self.var
    #     )

    #     # Cholesky factorization
    #     L = np.linalg.cholesky(K)

    #     # solve K^{-1} z via triangular solves
    #     alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))

    #     # log determinant
    #     logdet = 2.0 * np.sum(np.log(np.diag(L)))

    #     n = z.shape[0]

    #     return -0.5 * (z @ alpha + logdet + n * np.log(2 * np.pi))

    def log_prior(self, z, x=None):
        K = rbf_kernel(x, x, ell=self.ell, var=self.var) + 1e-8*np.eye(len(x))
        L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        logdet = 2*np.sum(np.log(np.diag(L)))

        return -0.5 * (z @ alpha)   # <<< DROP logdet TEMPORARILY

    def sample_kappa_field(
        self,
        k,
        kappa_theta,
        delta_eta,
        theta,
        x_obs,
        y_obs,
        gp_eta,
        sigma2,
        mh_scale
        ):
        
        No = len(x_obs)

        # --- GP prior covariance ---
        K = rbf_kernel(x_obs, x_obs, ell=self.ell, var=self.var) + 1e-8*np.eye(No)
        L = np.linalg.cholesky(K)

        # --- proposal ---
        proposal = kappa_theta[k] + mh_scale * (L @ np.random.randn(No))

        delta_prop = kappa_theta.copy()
        delta_prop[k] = proposal

        # --- log posterior current ---
        logpost_curr = (
            log_likelihood_embedded(y_obs, x_obs, theta, kappa_theta, delta_eta, gp_eta, sigma2)
            + self.log_prior(kappa_theta[k], x_obs)
        )

        # --- log posterior proposed ---
        logpost_prop = (
            log_likelihood_embedded(y_obs, x_obs, theta, delta_prop, delta_eta, gp_eta, sigma2)
            + self.log_prior(proposal, x_obs)
        )

        # diagnostic prints
        # print("proposal norm:", np.linalg.norm(proposal))

        # print("mean proposal jump:", np.linalg.norm(delta_prop - kappa_theta))

        # print("curr likelihood:",
        # log_likelihood_embedded(
        #     y_obs, x_obs, theta,
        #     kappa_theta, delta_eta,
        #     gp_eta, sigma2
        # ))

        # print("curr prior:",
        #     self.log_prior(kappa_theta[k], x_obs))

        # print("prop likelihood:",
        #     log_likelihood_embedded(
        #         y_obs, x_obs, theta,
        #         delta_prop, delta_eta,
        #         gp_eta, sigma2
        #     ))

        # print("prop prior:",
        #     self.log_prior(proposal, x_obs))
   
        log_alpha = logpost_prop - logpost_curr

        # print("log_alpha =", log_alpha)
        # print("isfinite?", np.isfinite(log_alpha))

        # print(f"log posterior current: {logpost_curr:.3f}, proposed: {logpost_prop:.3f}, log alpha: {log_alpha:.3f}")

        # print("prior diff check:",
        # np.linalg.norm(
        #     self.log_prior(proposal, x_obs) -
        #     self.log_prior(kappa_theta[k], x_obs)
        # ))

        # print("old style diff:",
        # gp_log_density(proposal, K) -
        # gp_log_density(kappa_theta[k], K))


        if np.log(np.random.rand()) < log_alpha:
            kappa_theta[k] = proposal
            # print("ACCEPTED")
            # print("new field norm:", np.linalg.norm(proposal))
            return kappa_theta, True
        else:
            return kappa_theta, False

    def sample_delta_eta_field(
        self,
        delta_eta,
        x_obs,
        y_obs,
        theta,
        kappa_theta,
        gp_eta,
        sigma2
        ):
            
        No = len(x_obs)

        # --- build covariance ---
        K = rbf_kernel(x_obs, x_obs, ell=self.ell, var=self.var)
        K += 1e-8 * np.eye(No)

        # --- compute residual ---
        r = np.zeros(No)

        for i in range(No):
            theta_i=get_theta_at_obs(theta, i)
            theta_star = theta_i + kappa_theta[:, i]
            m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            # print(f"m_i: {m_i:.3f}, y_obs[i]: {y_obs[i]}")
            r[i] = y_obs[i][0] - m_i

        # --- posterior ---
        K_inv = np.linalg.inv(K)
        Sigma_post = np.linalg.inv(K_inv + (1/sigma2)*np.eye(No))

        mu_post = Sigma_post @ ((1/sigma2) * r)

        return np.random.multivariate_normal(mu_post, Sigma_post)

    def update_hyperparameters(
    self,
    k,
    z,
    x_obs,
    mh_scales,
    allow_singular_cov
        ):

        old_ell = self.ell
        old_var = self.var

        self.ell, self.var, _ = (
            mh_update_delta_hyperparams(
                z,
                self.ell,
                self.var,
                x_obs,
                self.ell_prior,
                self.var_prior,
                mh_scales,
                allow_singular_cov
            )
        )
        # print(
        # f"updated GP prior:"
        # f" ell {old_ell:.4f}->{self.ell:.4f},"
        # f" var {old_var:.4f}->{self.var:.4f}"
        # )

    
    def get_state(self):

        return {
            "ell": self.ell,
            "var": self.var
        }
    
class LassoPrior(ShrinkagePrior):

    name = "lasso"

    def __init__(self, config):

        super().__init__(config)

        self.lam = config["lambda_init"]

        self.lambda_prior = config["lambda_prior"]

    def log_prior(self, z, x=None):

        n = len(z)

        return (
            n*np.log(self.lam/2)
            - self.lam*np.sum(np.abs(z))
        )
    
    def sample_kappa_field(self,
        k,
        kappa_theta,
        delta_eta,
        theta,
        x_obs,
        y_obs,
        gp_eta,
        sigma2,
        mh_scale
    ):
        proposal = kappa_theta[k] + mh_scale*np.random.randn(len(x_obs))

        delta_prop = kappa_theta.copy()
        delta_prop[k] = proposal

        logpost_curr = (
            log_likelihood_embedded(y_obs, x_obs, theta, kappa_theta, delta_eta, gp_eta, sigma2)
            + self.log_prior(kappa_theta[k])
        )

        logpost_prop = (
            log_likelihood_embedded(y_obs, x_obs, theta, delta_prop, delta_eta, gp_eta, sigma2)
            + self.log_prior(proposal)
        )

        log_alpha = logpost_prop-logpost_curr

        if np.log(np.random.rand()) < log_alpha:
            kappa_theta[k] = proposal
            accepted = True
        else:
            accepted = False

        return kappa_theta, accepted
    
    def sample_delta_eta_field(
        self,
        delta_eta,
        x_obs,
        y_obs,
        theta,
        kappa_theta,
        gp_eta,
        sigma2
        ):
            
        No = len(x_obs)

        # --- build covariance ---
        K = rbf_kernel(x_obs, x_obs, ell=self.ell, var=self.var)
        K += 1e-8 * np.eye(No)

        # --- compute residual ---
        r = np.zeros(No)

        for i in range(No):
            theta_i=get_theta_at_obs(theta, i)
            theta_star = theta_i + kappa_theta[:, i]
            m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            # print(f"m_i: {m_i:.3f}, y_obs[i]: {y_obs[i]}")
            r[i] = y_obs[i][0] - m_i

        # --- posterior ---
        K_inv = np.linalg.inv(K)
        Sigma_post = np.linalg.inv(K_inv + (1/sigma2)*np.eye(No))

        mu_post = Sigma_post @ ((1/sigma2) * r)

        return np.random.multivariate_normal(mu_post, Sigma_post)
        

    def update_hyperparameters(
    self,
    k,
    z,
    x_obs=None,
    mh_scales=None,
    allow_singular_cov=None,
    ):

        a = self.lambda_prior["shape"]
        b = self.lambda_prior["rate"]

        n = z.size

        self.lam = np.random.gamma(
            shape=a+n,
            scale=1/(b+np.sum(np.abs(z)))
        )

    def get_state(self):

        return {
            "lambda": self.lam
        }
    
class FusedLassoPrior(ShrinkagePrior):

    name = "fused_lasso"

    def __init__(self, config):

        super().__init__(config)

        self.lam = config["lambda_init"]
        self.lambda_prior = config["lambda_prior"]
        self.edges = None  # to be set by build_graph

    def log_prior(self, z, x=None):

        if self.edges is None:
            self.build_graph(x)

        penalty = self.graph_penalty(z)

        m = len(self.edges)

        return (
            m*np.log(self.lam/2)
            - self.lam*penalty
        )

    def sample_kappa_field(
        self,
        k,
        kappa_theta,
        delta_eta,
        theta,
        x_obs,
        y_obs,
        gp_eta,
        sigma2,
        mh_scale
    ):

        proposal = (
            kappa_theta[k]
            + mh_scale*np.random.randn(len(x_obs))
        )

        delta_prop = kappa_theta.copy()
        delta_prop[k] = proposal

        logpost_curr = (
            log_likelihood_embedded(
                y_obs,
                x_obs,
                theta,
                kappa_theta,
                delta_eta,
                gp_eta,
                sigma2
            )
            + self.log_prior(kappa_theta[k], x_obs)
        )

        logpost_prop = (
            log_likelihood_embedded(
                y_obs,
                x_obs,
                theta,
                delta_prop,
                delta_eta,
                gp_eta,
                sigma2
            )
            + self.log_prior(proposal, x_obs)
        )

        log_alpha = logpost_prop - logpost_curr

        if np.log(np.random.rand()) < log_alpha:
            kappa_theta[k] = proposal
            accepted = True
        else:
            accepted = False

        return kappa_theta, accepted


    def sample_delta_eta_field(
        self,
        delta_eta,
        x_obs,
        y_obs,
        theta,
        kappa_theta,
        gp_eta,
        sigma2
    ):

        # identical to LassoPrior
        No = len(x_obs)

        K = rbf_kernel(x_obs, x_obs,
                       ell=self.ell,
                       var=self.var)

        K += 1e-8*np.eye(No)

        r = np.zeros(No)

        for i in range(No):

            theta_i = get_theta_at_obs(theta, i)
            theta_star = theta_i + kappa_theta[:, i]

            m_i, _ = eta_predict(
                x_obs[i],
                theta_star,
                gp_eta
            )

            r[i] = y_obs[i][0] - m_i

        K_inv = np.linalg.inv(K)

        Sigma_post = np.linalg.inv(
            K_inv + (1/sigma2)*np.eye(No)
        )

        mu_post = Sigma_post @ ((1/sigma2)*r)

        return np.random.multivariate_normal(
            mu_post,
            Sigma_post
        )


    def update_hyperparameters(
        self,
        k,
        z,
        x_obs=None,
        mh_scales=None,
        allow_singular_cov=None,
    ):

        a = self.lambda_prior["shape"]
        b = self.lambda_prior["rate"]

        if self.edges is None:
            self.build_graph(x_obs)

        penalty = self.graph_penalty(z)

        m = len(self.edges)

        a = self.lambda_prior["shape"]
        b = self.lambda_prior["rate"]

        self.lam = np.random.gamma(
            shape=a+m,
            scale=1/(b+penalty)
        )

    def build_graph(self, x, k=2):

        tree = KDTree(x)

        _, nbrs = tree.query(x, k=k+1)

        edges = set()

        for i in range(len(x)):
            for j in nbrs[i][1:]:      # skip self
                edges.add(tuple(sorted((i, j))))

        self.edges = list(edges)

    def graph_penalty(self, z):

        penalty = 0.0

        for i, j in self.edges:
            penalty += abs(z[i] - z[j])

        return penalty

    def get_state(self):

        return {
            "lambda": self.lam
        }
    
class HorseshoePrior(ShrinkagePrior):

    name = "horseshoe"

    def __init__(self, config):

        super().__init__(config)

        self.tau_init = config["tau_init"]
        self.tau_prior = config["tau_prior"]

        # One independent hierarchy per embedded parameter
        self.tau = {}
        self.local_scale = {}
        self.local_aux = {}
        self.global_aux = {}


    def initialize(self, k, z):

        if k in self.tau:
            return

        n = len(z)

        self.tau[k] = self.tau_init

        self.local_scale[k] = np.ones(n)
        self.local_aux[k] = np.ones(n)
        self.global_aux[k] = 1.0

    def _sample_inverse_gamma(self, shape, scale):
        return 1.0 / np.random.gamma(shape, 1.0 / scale)
    
    def log_prior(self, k, z, x=None):

        self.initialize(k, z)

        sigma = self.tau[k] * self.local_scale[k]

        return np.sum(
            -0.5*np.log(2*np.pi*sigma**2)
            -0.5*(z/sigma)**2
        )
    
    def update_local_scales(self, k, z):

        # print("tau =", self.tau[k])

        # print("lambda min =", self.local_scale[k].min())

        # print("lambda max =", self.local_scale[k].max())

        # print("sum =", np.sum(z**2/self.local_scale[k]**2))

        # print("||z|| =", np.linalg.norm(z))
        # print("max |z| =", np.max(np.abs(z)))

        tau2 = self.tau[k]**2

        for i in range(len(z)):

            lam2 = self._sample_inverse_gamma(

                shape=1.0,

                scale=(z[i]**2)/(2*tau2)
                    + 1/self.local_aux[k][i]
            )

            self.local_scale[k][i] = np.sqrt(lam2)

    def update_global_scale(self, k, z):

        p = len(z)

        tau2 = self._sample_inverse_gamma(

            shape=(p+1)/2,

            scale=(
                0.5*np.sum(
                    z**2/self.local_scale[k]**2
                )
                + 1/self.global_aux[k]
            )
        )

        self.tau[k] = np.sqrt(tau2)

        self.global_aux[k] = self._sample_inverse_gamma(

            shape=1.0,

            scale=1 + 1/tau2
        )

    def log_half_cauchy(x, scale=1.0):

        if np.any(x <= 0):
            return -np.inf

        return np.sum(
            np.log(2)
            - np.log(np.pi*scale)
            - np.log(1 + (x/scale)**2)
        )
    
    def sample_kappa_field(self,
        k,
        kappa_theta,
        delta_eta,
        theta,
        x_obs,
        y_obs,
        gp_eta,
        sigma2,
        mh_scale
    ):
        
        self.initialize(k, kappa_theta[k])

        proposal = (
            kappa_theta[k]
            + mh_scale
            * np.sqrt(self.local_scale[k])
            * np.random.randn(len(kappa_theta[k]))
        )

        delta_prop = kappa_theta.copy()
        delta_prop[k] = proposal

        logpost_curr = (
            log_likelihood_embedded(y_obs, x_obs, theta, kappa_theta, delta_eta, gp_eta, sigma2)
            + self.log_prior(k, kappa_theta[k])
        )

        logpost_prop = (
            log_likelihood_embedded(y_obs, x_obs, theta, delta_prop, delta_eta, gp_eta, sigma2)
            + self.log_prior(k, proposal)
        )

        log_alpha = logpost_prop-logpost_curr

        if np.log(np.random.rand()) < log_alpha:
            kappa_theta[k] = proposal
            accepted = True
        else:
            accepted = False

        return kappa_theta, accepted
        

    
    def sample_delta_eta_field(
        self,
        delta_eta,
        x_obs,
        y_obs,
        theta,
        kappa_theta,
        gp_eta,
        sigma2
        ):
            
        No = len(x_obs)

        # --- build covariance ---
        K = rbf_kernel(x_obs, x_obs, ell=self.ell, var=self.var)
        K += 1e-8 * np.eye(No)

        # --- compute residual ---
        r = np.zeros(No)

        for i in range(No):
            theta_i = get_theta_at_obs(theta, i)
            theta_star = theta_i + kappa_theta[:, i]
            m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            # print(f"m_i: {m_i:.3f}, y_obs[i]: {y_obs[i]}")
            r[i] = y_obs[i][0] - m_i

        # --- posterior ---
        K_inv = np.linalg.inv(K)
        Sigma_post = np.linalg.inv(K_inv + (1/sigma2)*np.eye(No))

        mu_post = Sigma_post @ ((1/sigma2) * r)

        return np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_hyperparameters(
        self,
        k,
        z,
        x_obs=None,
        mh_scales=None,
        allow_singular_cov=None,
    ):

        self.initialize(k, z)

        self.update_local_scales(k, z)

        # update ν
        for i in range(len(z)):
            self.local_aux[k][i] = self._sample_inverse_gamma(
                shape=1.0,
                scale=1 + 1/(self.local_scale[k][i]**2)
            )

        # update τ and ξ
        self.update_global_scale(k, z)
    
    def get_state(self):

        return {

            "tau": self.tau,

            "lambda": self.local_scale,

            "nu": self.local_aux,

            "xi": self.global_aux
        }
    

class SpikeSlabPrior(ShrinkagePrior):

    name = "spike_slab"

    def __init__(self, config):

        super().__init__(config)

        self.pi = config["pi_init"]

        self.pi_prior = config["pi_prior"]

        self.sigma_spike = config["sigma_spike"]

        self.sigma_slab = config["sigma_slab"]

        self.gamma = {}

    def initialize(self, k, z):

        if k in self.gamma:
            return

        self.gamma[k] = np.ones(len(z), dtype=int)

    def log_prior(self, k, z, x=None):

        self.initialize(k, z)

        # print(type(self.gamma))
        # print(type(self.gamma[k]))
        # print(type(self.sigma_slab))
        # print(type(self.sigma_spike))

        sigma = np.where(
            self.gamma[k],
            self.sigma_slab,
            self.sigma_spike
        )

        return np.sum(
            -0.5*np.log(2*np.pi*sigma**2)
            -0.5*(z/sigma)**2
        )
    
    def update_indicators(self, k, z):

        self.initialize(k, z)

        for i in range(len(z)):

            slab_density = (
                self.pi
                * norm.pdf(
                    z[i],
                    0,
                    self.sigma_slab
                )
            )

            spike_density = (
                (1-self.pi)
                * norm.pdf(
                    z[i],
                    0,
                    self.sigma_spike
                )
            )

            p_slab = slab_density/(slab_density+spike_density)

            self.gamma[k][i] = (
                np.random.rand() < p_slab
            )

    def update_pi(self):

        alpha = self.pi_prior["alpha"]
        beta = self.pi_prior["beta"]

        total_active = 0
        total_count = 0

        for g in self.gamma.values():

            total_active += np.sum(g)

            total_count += len(g)

        self.pi = np.random.beta(
            alpha + total_active,
            beta + total_count - total_active
    )
        
    def sample_kappa_field(
    self,
    k,
    kappa_theta,
    delta_eta,
    theta,
    x_obs,
    y_obs,
    gp_eta,
    sigma2,
    mh_scale
    ):

        self.initialize(k, kappa_theta[k])

        proposal = (
            kappa_theta[k]
            + mh_scale*np.random.randn(
                len(kappa_theta[k])
            )
        )

        delta_prop = kappa_theta.copy()

        delta_prop[k] = proposal

        logpost_curr = (
            log_likelihood_embedded(
                y_obs,
                x_obs,
                theta,
                kappa_theta,
                delta_eta,
                gp_eta,
                sigma2
            )
            + self.log_prior(
                k,
                kappa_theta[k]
            )
        )

        logpost_prop = (
            log_likelihood_embedded(
                y_obs,
                x_obs,
                theta,
                delta_prop,
                delta_eta,
                gp_eta,
                sigma2
            )
            + self.log_prior(
                k,
                proposal
            )
        )

        log_alpha = (
            logpost_prop
            - logpost_curr
        )

        if np.log(np.random.rand()) < log_alpha:

            kappa_theta[k] = proposal

            return kappa_theta, True

        return kappa_theta, False
    
    def sample_delta_eta_field(
        self,
        delta_eta,
        x_obs,
        y_obs,
        theta,
        kappa_theta,
        gp_eta,
        sigma2
        ):
            
        No = len(x_obs)

        # --- build covariance ---
        K = rbf_kernel(x_obs, x_obs, ell=self.ell, var=self.var)
        K += 1e-8 * np.eye(No)

        # --- compute residual ---
        r = np.zeros(No)

        for i in range(No):
            theta_i = get_theta_at_obs(theta, i)
            theta_star = theta_i + kappa_theta[:, i]
            m_i, _ = eta_predict(x_obs[i], theta_star, gp_eta)
            # print(f"m_i: {m_i:.3f}, y_obs[i]: {y_obs[i]}")
            r[i] = y_obs[i][0] - m_i

        # --- posterior ---
        K_inv = np.linalg.inv(K)
        Sigma_post = np.linalg.inv(K_inv + (1/sigma2)*np.eye(No))

        mu_post = Sigma_post @ ((1/sigma2) * r)

        return np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_hyperparameters(
        self,
        k,
        z,
        x_obs=None,
        mh_scales=None,
        allow_singular_cov=None
    ):

        self.update_indicators(k, z)

        self.update_pi()
    
    
def get_state(self):

    return {
        "pi": self.pi,
        "gamma": self.gamma
    }
    

def create_prior(config):

    method = config["shrinkage_prior"]

    if method == "gp":
        return GPPrior(config["gp"])

    elif method == "lasso":
        return LassoPrior(config["lasso"])
    
    elif method == "fused_lasso":
        return FusedLassoPrior(config["fused_lasso"])

    elif method == "horseshoe":
        return HorseshoePrior(config["horseshoe"])

    elif method == "spike_slab":
        return SpikeSlabPrior(config["spike_slab"])

    else:
        raise ValueError(
            f"Unknown shrinkage prior {method}"
        )
    
def build_covariance(x, ell, var, jitter=1e-10):

    K = rbf_kernel(
        x,
        x,
        ell=ell,
        var=var
    )

    K += jitter * np.eye(len(x))

    return K