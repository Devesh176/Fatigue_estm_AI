# # 3_bayesian_filter.py
# import numpy as np
# import pandas as pd
# from scripts.utils import physics_state_transition
# class ParticleFilter:
#     def __init__(self, N=500, sigma_rho=1e-3, sigma_D=1e-3):
#         self.N = N
#         self.sigma_rho = sigma_rho
#         self.sigma_D = sigma_D
    
#     def initialize(self, init_state):
#         self.particles = np.tile(init_state, (self.N, 1))
#         self.weights = np.ones(self.N) / self.N

#     def predict(self, model_func, dt=1.0):
#         noise = np.random.normal(0, [self.sigma_rho, self.sigma_D], (self.N, 2))
#         self.particles = model_func(self.particles) + noise
#         self.particles = np.nan_to_num(self.particles, nan=0.0, posinf=1.0, neginf=0.0)


#     def update(self, measurement):
#         rho_meas, D_meas = measurement
#         if np.any(np.isnan(self.weights)):
#             print("⚠️ NaN detected in weights!")
#             print("Particles:", self.particles[:5])
#             print("Measurement:", measurement)
#             raise SystemExit

#         # 1. Compute Gaussian likelihood (using log to avoid underflow)
#         diff = self.particles - np.array([rho_meas, D_meas])
#         log_likelihood = -0.5 * np.sum(diff**2, axis=1)

#         # 2. Subtract max to prevent exp overflow
#         log_likelihood -= np.max(log_likelihood)

#         likelihood = np.exp(log_likelihood)
#         likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)

#         # 3. Update weights safely
#         self.weights *= likelihood
#         self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)

#         # 4. Normalize properly
#         total = np.sum(self.weights)
#         if total == 0 or np.isnan(total):
#             self.weights = np.ones(self.N) / self.N
#         else:
#             self.weights /= total

#         # 5. Resample
#         self.resample()


#     def resample(self):
#         idx = np.random.choice(self.N, size=self.N, p=self.weights)
#         self.particles = self.particles[idx]
#         self.weights.fill(1.0 / self.N)

#     def estimate(self):
#         return np.average(self.particles, weights=self.weights, axis=0)

# def micro_macro_model(x):
#     """Deterministic state evolution."""
#     rho, D = x[:, 0], x[:, 1]
#     d_rho = 0.002 * (1 - rho)
#     d_D = -0.001 * (1 - D)
#     return np.stack([rho + d_rho, D + d_D], axis=1)

# scripts/bayesian_filter.py
import numpy as np
import pandas as pd

class ParticleFilter:
    def __init__(self, N=500, sigma_state_rho=1e-3, sigma_state_D=1e-3, sigma_param_At=1e-7, sigma_param_alpha_t=1e-4):
        """
        Initialize the particle filter.
        N: Number of particles
        sigma_state_*: Process noise for the state variables (rho, D)
        sigma_param_*: "Artificial evolution" noise for parameters (A_t, alpha_t)
        """
        self.N = N
        self.sigma_state = np.array([sigma_state_rho, sigma_state_D])
        self.sigma_params = np.array([sigma_param_At, sigma_param_alpha_t])
        
        # We will have N particles, each with 2 states (rho, D) and 2 params (A_t, alpha_t)
        self.particles_state = np.zeros((self.N, 2))
        self.particles_params = np.zeros((self.N, 2))
        self.weights = np.ones(self.N) / self.N

    def initialize(self, init_state, init_params_mean, init_params_cov):
        """
        Initialize particle states and parameters.
        init_state: [rho_0, D_0]
        init_params_mean: [A_t_mean, alpha_t_mean]
        init_params_cov: Covariance matrix for parameter initialization
        """
        # Initialize states (all particles start at the same state)
        self.particles_state = np.tile(init_state, (self.N, 1))
        
        # Initialize parameters by sampling from a prior distribution
        self.particles_params = np.random.multivariate_normal(init_params_mean, init_params_cov, self.N)
        
        # Initialize weights
        self.weights.fill(1.0 / self.N)
        
        # Apply safety clips to initial parameters
        self.clip_params()

    def clip_params(self):
        """Apply physical constraints to parameters."""
        # A_t (Paris law coefficient) must be positive
        self.particles_params[:, 0] = np.clip(self.particles_params[:, 0], 1e-7, 1e-2)
        # alpha_t (Paris law exponent) is typically > 1
        self.particles_params[:, 1] = np.clip(self.particles_params[:, 1], 1.0, 5.0)

    def predict(self, model_func):
        """
        Predict step: Evolve parameters and then evolve state.
        This follows Algorithm 1, line 5.
        """
        
        # 1. Evolve parameters (Artificial Evolution / Random Walk) [cite: 124]
        param_noise = np.random.normal(0, self.sigma_params, (self.N, 2))
        self.particles_params += param_noise
        self.clip_params() # Re-apply constraints
        
        # 2. Evolve state using the physics model and the *new* parameters
        # The model function g(x_{n-1}, theta_n) [cite: 88, 89]
        full_particle_data = np.hstack((self.particles_state, self.particles_params))
        
        # Get the deterministic state prediction
        state_pred = model_func(full_particle_data)
        
        # 3. Add process noise v_n [cite: 88, 89]
        state_noise = np.random.normal(0, self.sigma_state, (self.N, 2))
        self.particles_state = state_pred + state_noise
        
        # Apply state constraints (rho: 0->1, D: 0->1)
        self.particles_state[:, 0] = np.clip(self.particles_state[:, 0], 0, 1.0)
        self.particles_state[:, 1] = np.clip(self.particles_state[:, 1], 0, 1.0)

    def update(self, measurement):
        """
        Update step: Reweight particles based on measurement likelihood.
        This follows Algorithm 1, line 6.
        """
        rho_meas, D_meas = measurement
        
        # Calculate likelihood p(y_n | x_n) [cite: 103, 104]
        # We assume a Gaussian measurement noise
        # Using log-likelihood for numerical stability
        
        # Define measurement noise (can be tuned)
        sigma_meas_rho = 0.05
        sigma_meas_D = 0.01
        
        diff = self.particles_state - np.array([rho_meas, D_meas])
        log_likelihood_rho = -0.5 * (diff[:, 0]**2 / sigma_meas_rho**2)
        log_likelihood_D = -0.5 * (diff[:, 1]**2 / sigma_meas_D**2)
        
        total_log_likelihood = log_likelihood_rho + log_likelihood_D
        
        # Convert back from log-space safely
        log_likelihood_adj = total_log_likelihood - np.max(total_log_likelihood)
        likelihood = np.exp(log_likelihood_adj)
        
        # Update weights: w_n = p(y_n | x_n) * w_{n-1} [cite: 120]
        self.weights *= likelihood
        
        # Normalize weights
        total = np.sum(self.weights)
        if total > 0 and not np.isnan(total):
            self.weights /= total
        else:
            # Reset weights if all particles die
            self.weights.fill(1.0 / self.N)

        # Resample if needed (effective sample size)
        if 1.0 / np.sum(self.weights**2) < self.N / 2:
            self.resample()

    def resample(self):
        """
        Resample particles using systematic resampling.
        """
        indices = np.random.choice(self.N, size=self.N, p=self.weights)
        
        # Resample both state and parameters
        self.particles_state = self.particles_state[indices]
        self.particles_params = self.particles_params[indices]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """
        Return the weighted average of state and parameters.
        """
        state_est = np.average(self.particles_state, weights=self.weights, axis=0)
        params_est = np.average(self.particles_params, weights=self.weights, axis=0)
        return np.concatenate((state_est, params_est))  