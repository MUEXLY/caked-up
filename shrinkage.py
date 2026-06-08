import numpy as np
from scipy.stats import multivariate_normal
from functs import *

class ShrinkagePrior:

    name="shrinkage_base"

    def __init__(self, config):
        self.config = config

    def log_prior(self, z):
        raise NotImplementedError

    def update_hyperparameters(self, *args):
        pass

    def get_state(self):
        return {}
    
class FieldPrior:

    name="field_base"
    
    def __init__(self, config):
        self.config = config

    def log_prior(self, z, x):
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

    def log_prior(self, z, x):

        K = build_covariance(
            x,
            self.ell,
            self.var
        )

        return multivariate_normal.logpdf(
            z,
            np.zeros(len(z)),
            K
        )
    
    def update_hyperparameters(
    self,
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

    def log_prior(self, z):

        return -self.lam * np.sum(np.abs(z))
    
    def get_state(self):

        return {
            "lambda": self.lam
        }
    
class HorseshoePrior(ShrinkagePrior):
    # NOTE:
    # This is currently a Cauchy approximation to a
    # true horseshoe prior.

    name = "horseshoe"

    def __init__(self, config):

        super().__init__(config)

        self.tau = config["tau_init"]

        self.tau_prior = config["tau_prior"]

    def log_prior(self, z):

        return np.sum(
            np.log(1/(self.tau * np.pi * (1 + (z/self.tau)**2)))
        )
    
    def get_state(self):

        return {
            "tau": self.tau
        }
    
class SpikeSlabPrior(ShrinkagePrior):

    name = "spike_slab"

    def __init__(self, config):

        super().__init__(config)

        self.pi = config["pi_init"]
        self.sigma2_slab = config["sigma2_slab_init"]

        self.pi_prior = config["pi_prior"]
        self.sigma2_slab_prior = config["sigma2_slab_prior"]

    def log_prior(self, z):

        logp_spike = np.log(1 - self.pi) - 0.5 * np.log(2 * np.pi * 1e-6) - (z**2) / (2 * 1e-6)
        logp_slab = np.log(self.pi) - 0.5 * np.log(2 * np.pi * self.sigma2_slab) - (z**2) / (2 * self.sigma2_slab)

        return np.sum(np.logaddexp(logp_spike, logp_slab))
    
    def get_state(self):

        return {
            "pi": self.pi,
            "sigma2_slab": self.sigma2_slab
        }
    

def create_prior(config):

    method = config["shrinkage_prior"]

    if method == "gp":
        return GPPrior(config["gp"])

    elif method == "lasso":
        return LassoPrior(config["lasso"])

    elif method == "horseshoe":
        return HorseshoePrior(config["horseshoe"])

    elif method == "spike_slab":
        return SpikeSlabPrior(config["spike_slab"])

    else:
        raise ValueError(
            f"Unknown shrinkage prior {method}"
        )
    
def build_covariance(x, ell, var):

    K = rbf_kernel(
        x,
        x,
        ell=ell,
        var=var
    )

    K += 1e-10 * np.eye(len(x))

    return K