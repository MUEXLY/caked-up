#imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os


# create a sample problem of an embedded constiuative model
# the model form should be f(g(x)) = y
# where f is the outer model and g is the inner model
# we will use a simple linear model for f and a nonlinear model for g
# the inner model will be a simple quadratic function
def g(x):
    return x**2 + 2*x + 1

# the outer model will be a simple linear function
def f(z):
    return 3*z + 2

# define a simulation form that has inner model and outer model discrepancy
# the inner model discrepancy will be a simple linear function
def inner_discrepancy(x):
    return 0.5*x + 1

# the outer model discrepancy will be a simple linear function
def outer_discrepancy(z):
    return 0.8*z + 0.5

def main():

    # now we can create a sample dataset
    x = np.linspace(-10, 10, 100)
    y = f(g(x))

    # now we can add some noise to the data
    noise = np.random.normal(0, 10, size=y.shape)
    y_noisy = y + noise


    # plot the true model, noisy observations, and the synthetic simulation with inner and outer model discrepancy
    # first we need to create the synthetic simulation data
    y_simulation = f(g(x) + inner_discrepancy(x)) + outer_discrepancy(f(g(x) + inner_discrepancy(x)))   

    # create discrete simulation data points for comparison with the noisy observations
    # have more simulation data points than the noisy observations
    x_simulation = np.linspace(-10, 10, 200)
    y_simulation_discrete = f(g(x_simulation) + inner_discrepancy(x_simulation)) + outer_discrepancy(f(g(x_simulation) + inner_discrepancy(x_simulation)))

    # create the folders if they don't exist

    if not os.path.exists('modelData'):
        os.makedirs('modelData')
    if not os.path.exists('observationData'):
        os.makedirs('observationData')

    # save the model data
    np.savetxt('modelData/appDomain.txt', x_simulation) # vector of x values for the model predictions
    np.savetxt('modelData/modelPredictions.txt', y_simulation_discrete) # vector of model predictions for the x values
    np.savetxt('modelData/thetaVals.txt', [g(x) + inner_discrepancy(x) for x in x_simulation]) # vector of utilized inner model values

    # save the observation data
    np.savetxt('observationData/appDomain.txt', x) # vector of x values
    np.savetxt('observationData/observationData.txt', y_noisy) # vector of noisy observations for the x values

    # create subsampled observation cases (50%, 25%, 10%) with 100% simulation data
    np.random.seed(42)
    cases = {'50pct': 0.50, '25pct': 0.25, '10pct': 0.10}

    for name, frac in cases.items():
        base = name
        obs_dir = os.path.join(base, 'observationData')
        mod_dir = os.path.join(base, 'modelData')
        os.makedirs(obs_dir, exist_ok=True)
        os.makedirs(mod_dir, exist_ok=True)

        # save full simulation data (100%)
        np.savetxt(os.path.join(mod_dir, 'appDomain.txt'), x_simulation)
        np.savetxt(os.path.join(mod_dir, 'modelPredictions.txt'), y_simulation_discrete)
        theta_vals = np.array([g(xi) + inner_discrepancy(xi) for xi in x_simulation])
        np.savetxt(os.path.join(mod_dir, 'thetaVals.txt'), theta_vals)

        # choose observation subset
        n_keep = max(1, int(len(x) * frac))
        keep_idx = np.sort(np.random.choice(len(x), n_keep, replace=False))
        hold_idx = np.setdiff1d(np.arange(len(x)), keep_idx)

        # save used observations
        np.savetxt(os.path.join(obs_dir, 'appDomain.txt'), x[keep_idx])
        np.savetxt(os.path.join(obs_dir, 'observationData.txt'), y_noisy[keep_idx])

        # save held-out points in one file per field
        np.savetxt(os.path.join(obs_dir, 'appDomain_holdout.txt'), x[hold_idx])
        np.savetxt(os.path.join(obs_dir, 'observationData_holdout.txt'), y_noisy[hold_idx])