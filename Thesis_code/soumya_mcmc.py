"""
fit_ydata.py fits the Sx ydata with a beta model

Author: Soumya Shreeram
Email: shreeram@mpe.mpg.de
Date created: 2nd August 2024
"""
import numpy as np
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import ticker
import os




"""
======
Models
======
"""

def schechter_function(theta, x):
    """ 
    theta :: 
        parameters of the schechter function
    x ::
        mass values (x-axis)
    """
    phi_s, m_s, alpha = theta
    model = phi_s * (x / m_s )**alpha * np.exp(- ( x/ m_s ) ) / m_s # times the bin size dM
    return model

def double_schechter_function(theta, x):
    """ 3 parameter model
    """
    phi_s, m_s, alpha, phi_s1, m_s1, alpha1 = theta
    
    # edit model accordingly
    model = (10**L0) * beta_model
    return model

"""
===================
Fitting to ydatas
===================
"""

def log_likelihood(theta, x, y, yerr, fit_with):
    """Convolve the beta ydata with the mean PSF ydata [kpc] of the galaxy sample

    Parameters
    ----------
    theta ::
        priors used for beta model

    """
    if fit_with.lower() in ['schechter_function']:
        model = schechter_function(theta, x)
    if fit_with.lower() in ['double_schechter_function']:
        model = double_schechter_function(theta, x)
    
    sigma2 = (yerr)**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_schechter_function(theta, prior_ranges):
    phi_s, m_s, alpha = theta
    
    phi_s_min, phi_s_max = prior_ranges[0]
    m_s_min, m_s_max = prior_ranges[1]
    alpha_min, alpha_max = prior_ranges[2]

    if (phi_s_min < phi_s < phi_s_max and m_s_min < m_s < m_s_max and alpha_min < alpha < alpha_max):
        return(0.0)
    return(-np.inf)

def log_prior_double_schechter_function(theta):
    phi_s, m_s, alpha, phi_s1, m_s1, alpha1 = theta

    if (0.<phi_s<1 and 10.0<m_s<11.6 and 0.<alpha<4.9 and 0.<phi_s1<1 and 10.0<m_s1<11.6 and 0.<alpha1<4.9):
        return(0.0)
    return(-np.inf)

def log_probability(theta, x, y, yerr, fit_with, prior_ranges):
    if fit_with in ['schechter_function']:
        lp = log_prior_schechter_function(theta, prior_ranges)
    if fit_with.lower() in ['double_schechter_function']:
        # the likelihood defining quality of fit
        lp = log_prior_double_schechter_function(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, fit_with)

def fit_schechter_function2data(xbin, ydata, ydata_err, fit_with, initial, prior_ranges, n_steps=1000):
    """
    Function to fit the schoecter function to data

    xbin, bins ::
        mass bin means, and bin values
    ydata ::
    ydata_err
    psf_kpc ::
        the mean PSF of the galaxy sample in kpc (physical scale)
    """
    np.random.seed(420)
    nll = lambda *args: -log_likelihood(*args)

    if fit_with in ['schechter_function']:
        # parameter initial guesses
        #initial = np.array([1.4, 17., .40])

        # parameter physical bounds
        #bounds=((0, 50), (10, 12), (-2, 0))

        # optimizes the best-guess input values to get "first estimation" of best-fit parameters
        soln = minimize(nll, initial,
                        args=(xbin, ydata, ydata_err, fit_with),
                        method='SLSQP', # minimizer optimizes over the bounds
                        bounds=prior_ranges)
        phi_s_ml, m_s_ml, alpha_ml = soln.x
        print(f"{phi_s_ml=:.2f}, {m_s_ml=:.2f}, {alpha_ml=:.2f}")

        pos_phi_s   = np.ravel(soln.x[0]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        pos_m_s   = np.ravel(soln.x[1]  + np.hstack([1e-2   * np.random.randn(150, 1 ) ]))
        pos_alpha = np.ravel(soln.x[2]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        
        pos = np.array([pos_phi_s, pos_m_s, pos_alpha]).T

    if fit_with.lower() in ['double_schechter_function']:
        # parameter initial guesses
        #initial = np.array([1.4, 17., .40, 1.4, 17., .40])

        # parameter physical bounds
        #bounds=((34., 39.), (0.01, 0.6), (0., 5.), (34., 39.), (0.01, 0.6), (0., 5.))

        # optimizes the best-guess input values to get "first estimation" of best-fit parameters
        soln = minimize(nll, initial,
                        args=(xbin, bins, ydata, ydata_err, fit_with, mean_r),
                        method='SLSQP', # minimizer optimizes over the bounds
                        bounds=prior_ranges)
        phi_s_ml, m_s_ml, alpha_ml, phi_s1_ml, m_s1_ml, alpha1_ml = soln.x
        print(f"{phi_s_ml=:.2f}, {m_s_ml=:.2f}, {alpha_ml=:.2f}")

        pos_phi_s   = np.ravel(soln.x[0]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        pos_m_s   = np.ravel(soln.x[1]  + np.hstack([1e-2   * np.random.randn(150, 1 ) ]))
        pos_alpha = np.ravel(soln.x[2]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        pos_phi_s1   = np.ravel(soln.x[3]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        pos_m_s1   = np.ravel(soln.x[4]  + np.hstack([1e-2   * np.random.randn(150, 1 ) ]))
        pos_alpha1 = np.ravel(soln.x[5]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        
        pos = np.array([pos_phi_s, pos_m_s, pos_alpha, pos_phi_s1, pos_m_s1, pos_alpha1]).T

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xbin, ydata, ydata_err, fit_with, prior_ranges))
        sampler.run_mcmc(pos, n_steps, progress=True)

        # all the results from the mcmc walks
        flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
        if fit_with.lower() in ['schechter_function']:
            labels = ["Phi_s", "m_s", "\alpha"]
        else:
            labels = ["Phi_s", "m_s", "\alpha", "Phi_s1", "m_s1", "\alpha_1"]

        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [40, 50, 60])
            q = np.diff(mcmc)
            txt = r"${{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            print(txt)
        return flat_samples


"""
Implementation
--------------
flat_samples = fit_schechter_function2data(xbin, bins, ydata, ydata_err, fit_with)
do_corner_plot(flat_samples)
"""

"""
==================
Plotting functions
==================
"""
def setup(ax, format_y: bool = True):
    """Set up common parameters for the Axes in the example."""
    # only show the bottom spine
    if format_y:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    else:
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.spines[['left', 'right', 'top']].set_visible(True)

    ax.tick_params(axis='x', which='both',  direction='in', length=6, width=1)
    ax.tick_params(axis='y', which='both',  direction='in', length=6, width=1)


def do_corner_plot(sample):
    if sample.shape[1] == 3:
        labels = [r"phi_s", r"m_s", r"alpha"]

    if sample.shape[1] == 6:
        labels = [r"$\log_{10} S_0$", r"$\rm \frac{r_c}{kpc}$", r"$\alpha$", r"$\beta$",  r"$\rm \frac{r_s}{kpc}$", r"$\epsilon$"]
    fsize = int(3*sample.shape[1])
    fig, ax = plt.subplots(sample.shape[1], sample.shape[1], figsize=(fsize, fsize))
    sample[:, 0] += 35.
    fig = corner.corner(
        sample, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 20},
        labelpad=0.05,
        smooth=1.5,
        max_n_ticks=2.,
        use_math_text=True,
        hist_bin_factor=1.4,
        fig=fig,
        plot_contours=True,
        hist_kwargs={"histtype": "stepfilled",
                     "color":"grey",
                     "alpha": 0.3,
                      "ec": 'k',
                      "lw": 3.},
        hist2d_kwargs={"bins": 100,
                       "color": 'g'}
    )
    for i in [0]:
        for j in np.arange(sample.shape[1]):
            setup(ax[j, i])

    for m in [sample.shape[1]-1]:
        for n in np.arange(sample.shape[1]):
            setup(ax[m, n], format_y=False)

    # Extract the axes
    ndim = sample.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # This is the empirical mean of the sample:
    value2 = np.median(sample, axis=0)

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.tick_params(axis='x', which='both',  direction='in', length=6, width=1)
        ax.axvline(value2[i], color="r")

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(value2[xi], value2[yi], "xr", ms=10., mec='r', mew=3.)