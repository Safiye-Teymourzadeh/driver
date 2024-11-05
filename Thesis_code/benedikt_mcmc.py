import numpy as np
from scipy.optimize import minimize
import emcee

class MCMC:
    def __init__(self, bounds: tuple, start_values: tuple, n_steps: int, masses: list, y_values: list, y_errors: list):
        self.bounds = bounds
        self.start_values = start_values
        self.n_steps = n_steps
        self.masses = masses
        self.y_values = y_values
        self.y_errors = y_errors
        
    def log_prior_schechter_function(self, theta):
        
        phi_s_min, phi_s_max = self.bounds[0]
        m_s_min, m_s_max = self.bounds[1]
        alpha_min, alpha_max = self.bounds[2]

        if (phi_s_min < theta[0] < phi_s_max and m_s_min < theta[1] < m_s_max and alpha_min < theta[2] < alpha_max):
            return(0.0)
        return(-np.inf)
        
    def log_probability(self, theta):
        lp = self.log_prior_schechter_function(theta)

        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def log_likelihood(self, theta):
        model = self.schechter_function(theta)
        sigma2 = (self.y_errors)**2
        return -0.5 * np.sum((self.y_values - model) ** 2 / sigma2 + np.log(sigma2))

    def schechter_function(self, theta):
        phi_s, m_s, alpha = theta
        return phi_s * (self.masses / m_s )**alpha * np.exp(- ( self.masses / m_s ) ) / m_s # times the bin size dM

    def fit_data(self):
        np.random.seed(420)

        nll = lambda *args: -self.log_likelihood(*args)
                
        soln = minimize(nll, np.array(self.start_values),
                        method='SLSQP', # minimizer optimizes over the bounds
                        bounds=self.bounds)
        phi_s_ml, m_s_ml, alpha_ml = soln.x
        print(f"{phi_s_ml=:.2f}, {m_s_ml=:.2f}, {alpha_ml=:.2f}")

        pos_phi_s   = np.ravel(soln.x[0]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        pos_m_s   = np.ravel(soln.x[1]  + np.hstack([1e-2   * np.random.randn(150, 1 ) ]))
        pos_alpha = np.ravel(soln.x[2]  + np.hstack([1e-3   * np.random.randn(150, 1 ) ]))
        
        pos = np.array([pos_phi_s, pos_m_s, pos_alpha]).T
        
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=())
        sampler.run_mcmc(pos, self.n_steps, progress=True)

        # all the results from the mcmc walks
        flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
        
        labels = ["Phi_s", "m_s", "\alpha"]
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [40, 50, 60])
            q = np.diff(mcmc)
            txt = r"${{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            print(txt)
        
        return flat_samples
