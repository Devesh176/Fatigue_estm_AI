# # 4_rul_prediction.py
# import numpy as np
# import pandas as pd
# from scripts.bayesian_filter import ParticleFilter
# def predict_probabilistic_RUL(particle_filter, current_cycle, rho_thresh=0.4, D_thresh=0.88, max_steps=100):
#     """
#     Predicts RUL by propagating each particle to the failure threshold.
#     This implements Algorithm 2 from the paper.
#     """ # [cite: 139]
    
#     particles_future = particle_filter.particles.copy()
#     N = particle_filter.N
#     eol_cycles = np.zeros(N)

#     for i in range(N):
#         particle_path = particles_future[i:i+1, :]
#         n_future = current_cycle
        
#         for step in range(max_steps):
#             # Propagate the single particle
#             particle_path = physics_state_transition(particle_path)
#             rho_f, D_f = particle_path[0, 0], particle_path[0, 1]
            
#             # Assuming each step represents 5000 cycles
#             n_future += 5000

#             # Check if failure threshold is reached
#             if rho_f >= rho_thresh or D_f <= D_thresh:
#                 break
        
#         eol_cycles[i] = n_future

#     # Calculate RUL using the weighted average of particle EOLs
#     mean_eol = np.average(eol_cycles, weights=particle_filter.weights)
#     rul = max(0, mean_eol - current_cycle)

#     # Calculate uncertainty (e.g., 5th and 95th percentiles)
#     sorted_indices = np.argsort(eol_cycles)
#     sorted_weights = particle_filter.weights[sorted_indices]
#     cumulative_weights = np.cumsum(sorted_weights)
    
#     lower_bound_idx = np.where(cumulative_weights >= 0.05)[0][0]
#     upper_bound_idx = np.where(cumulative_weights >= 0.95)[0][0]
    
#     eol_lower = eol_cycles[sorted_indices[lower_bound_idx]]
#     eol_upper = eol_cycles[sorted_indices[upper_bound_idx]]
    
#     rul_lower = max(0, eol_lower - current_cycle)
#     rul_upper = max(0, eol_upper - current_cycle)

#     return {
#         "RUL": rul,
#         "RUL_lower": rul_lower,
#         "RUL_upper": rul_upper
#     }

# def predict_RUL_series(df):
#     """
#     Runs the particle filter and predicts RUL at each time step.
#     """
#     pf = ParticleFilter(N=500)
#     pf.initialize(init_state=[0, 1]) # rho=0, D=1
    
#     rul_predictions = []
    
#     for _, row in df.iterrows():
#         current_cycle = row['cycle']
#         measurement = [row['rho_smooth'], row['D_smooth']]
        
#         # Run one step of the particle filter
#         pf.predict(physics_state_transition)
#         pf.update(measurement)
        
#         # Predict RUL with the updated particle distribution
#         rul_dict = predict_probabilistic_RUL(pf, current_cycle)
#         rul_predictions.append(rul_dict)
        
#     df_rul = pd.DataFrame(rul_predictions)
#     return pd.concat([df.reset_index(drop=True), df_rul], axis=1)

# 4_rul_prediction.py
import numpy as np
import pandas as pd
from scripts.utils import g1_paris_law_model, g2_stiffness_model # Import new models
from scripts.bayesian_filter import ParticleFilter
from scripts.utils import physics_state_transition # Use the same model

def predict_probabilistic_RUL(particle_filter, current_cycle, rho_thresh=0.4, D_thresh=0.88, max_steps=100, step_size_cycles=5000):
    """
    Predicts RUL by propagating each particle to the failure threshold.
    This implements Algorithm 2 from the paper.
    """
    
    # Get copies of all particle states and parameters
    particles_state_future = particle_filter.particles_state.copy()
    particles_params_future = particle_filter.particles_params.copy()
    N = particle_filter.N
    
    eol_cycles = np.full(N, -1.0) # Array to store EOL for each particle

    for i in range(N):
        # Get the state and params for this one particle
        rho, D = particles_state_future[i]
        A_t, alpha_t = particles_params_future[i]
        
        n_future = current_cycle
        
        for step in range(max_steps):
            # Check failure threshold *before* predicting
            if rho >= rho_thresh or D <= D_thresh:
                break
            
            # Propagate this single particle
            rho = g1_paris_law_model(rho, A_t, alpha_t)
            D = g2_stiffness_model(rho)
            
            n_future += step_size_cycles
        
        eol_cycles[i] = n_future

    # Calculate RUL using the weighted average of particle EOLs
    mean_eol = np.average(eol_cycles, weights=particle_filter.weights)
    rul = max(0, mean_eol - current_cycle)

    # Calculate uncertainty (5th and 95th percentiles)
    # We must use the *weights* to find the percentiles
    sorted_indices = np.argsort(eol_cycles)
    sorted_eols = eol_cycles[sorted_indices]
    sorted_weights = particle_filter.weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    
    lower_idx = np.where(cumulative_weights >= 0.05)[0][0]
    upper_idx = np.where(cumulative_weights >= 0.95)[0][0]
    
    eol_lower = sorted_eols[lower_idx]
    eol_upper = sorted_eols[upper_idx]
    
    rul_lower = max(0, eol_lower - current_cycle)
    rul_upper = max(0, eol_upper - current_cycle)

    return {
        "RUL": rul,
        "RUL_lower": rul_lower,
        "RUL_upper": rul_upper
    }

def predict_RUL_series(df, pf_config):
    """
    Runs the particle filter and predicts RUL at each time step.
    """
    # Create the filter from config
    pf = ParticleFilter(
        N=pf_config['N'],
        sigma_state_rho=pf_config['sigma_state_rho'],
        sigma_state_D=pf_config['sigma_state_D'],
        sigma_param_At=pf_config['sigma_param_At'],
        sigma_param_alpha_t=pf_config['sigma_param_alpha_t']
    )
    
    pf.initialize(
        init_state=pf_config['init_state'],
        init_params_mean=pf_config['init_params_mean'],
        init_params_cov=pf_config['init_params_cov']
    )
    
    results = []
    
    for _, row in df.iterrows():
        current_cycle = row['cycle']
        measurement = [row['rho_smooth'], row['D_smooth']]
        
        # Run one step of the particle filter
        pf.predict(physics_state_transition)
        pf.update(measurement)
        
        # Get weighted average estimates
        est_rho, est_D, est_At, est_alpha = pf.estimate()
        
        # Predict RUL with the updated particle distribution
        rul_dict = predict_probabilistic_RUL(pf, current_cycle)
        
        # Store all results
        rul_dict['cycle'] = current_cycle
        rul_dict['rho_filtered'] = est_rho
        rul_dict['D_filtered'] = est_D
        rul_dict['A_t_filtered'] = est_At
        rul_dict['alpha_t_filtered'] = est_alpha
        results.append(rul_dict)
        
    df_results = pd.DataFrame(results)
    return pd.merge(df.reset_index(drop=True), df_results, on='cycle')