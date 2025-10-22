# scripts/3_bayesian_filter.py
import numpy as np
import pandas as pd
from scripts.utils import physics_state_transition

class ParticleFilter:
    # ... __init__ is unchanged ...
    def __init__(self, N=1000, sigma_state_rho=1e-3, sigma_state_D=1e-3, sigma_param_At=1e-7, sigma_param_alpha_t=1e-4):
        self.N = N
        self.sigma_state = np.array([sigma_state_rho, sigma_state_D])
        self.sigma_params = np.array([sigma_param_At, sigma_param_alpha_t])
        
        self.particles_state = np.zeros((self.N, 2))
        self.particles_params = np.zeros((self.N, 2))
        self.weights = np.ones(self.N) / self.N

    # ... initialize is unchanged ...
    def initialize(self, init_state, init_params_mean, init_params_cov):
        self.particles_state = np.tile(init_state, (self.N, 1))
        self.particles_params = np.random.multivariate_normal(init_params_mean, init_params_cov, self.N)
        self.weights.fill(1.0 / self.N)
        self.clip_params()

    def clip_params(self):
        """Apply physical constraints to parameters."""
        # --- START FIX: Tighter bounds ---
        # A_t (Paris law coefficient)
        self.particles_params[:, 0] = np.clip(self.particles_params[:, 0], 1e-9, 1e-3)
        # alpha_t (Paris law exponent)
        self.particles_params[:, 1] = np.clip(self.particles_params[:, 1], 1.0, 4.0)
        # --- END FIX ---

    # ... predict, update, resample, estimate are unchanged ...
    def predict(self, model_func, delta_n=1):
        """
        Predict step: Evolve parameters and then simulate state forward
        by delta_n cycles.
        """
        param_noise = np.random.normal(0, self.sigma_params, (self.N, 2))
        self.particles_params += param_noise
        self.clip_params() 
        
        state_pred = model_func(self.particles_state, self.particles_params, num_cycles=delta_n)

        state_noise = np.random.normal(0, self.sigma_state, (self.N, 2))
        self.particles_state = state_pred + state_noise
        
        self.particles_state[:, 0] = np.clip(self.particles_state[:, 0], 0, 1.0)
        self.particles_state[:, 1] = np.clip(self.particles_state[:, 1], 0, 1.0)

    def update(self, measurement):
        """
        Update step: Reweight particles based on measurement likelihood.
        """
        rho_meas, D_meas = measurement
        
        sigma_meas_rho = 0.05
        sigma_meas_D = 0.01
        
        diff = self.particles_state - np.array([rho_meas, D_meas])
        log_likelihood_rho = -0.5 * (diff[:, 0]**2 / sigma_meas_rho**2)
        log_likelihood_D = -0.5 * (diff[:, 1]**2 / sigma_meas_D**2)
        
        total_log_likelihood = log_likelihood_rho + log_likelihood_D
        
        log_likelihood_adj = total_log_likelihood - np.max(total_log_likelihood)
        likelihood = np.exp(log_likelihood_adj)
        
        self.weights *= likelihood
        
        total = np.sum(self.weights)
        if total > 0 and not np.isnan(total):
            self.weights /= total
        else:
            self.weights.fill(1.0 / self.N) 

        if 1.0 / np.sum(self.weights**2) < self.N / 2:
            self.resample()

    def resample(self):
        indices = np.random.choice(self.N, size=self.N, p=self.weights)
        self.particles_state = self.particles_state[indices]
        self.particles_params = self.particles_params[indices]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        state_est = np.average(self.particles_state, weights=self.weights, axis=0)
        params_est = np.average(self.particles_params, weights=self.weights, axis=0)
        return np.concatenate((state_est, params_est))