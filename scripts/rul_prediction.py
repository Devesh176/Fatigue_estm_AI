# scripts/4_rul_prediction.py
import numpy as np
import pandas as pd
from scripts.utils import g1_linear_model, g2_stiffness_model, physics_state_transition # Import new model
from scripts.bayesian_filter import ParticleFilter 

def predict_probabilistic_RUL(particle_filter, current_cycle, rho_thresh=0.4, D_thresh=0.88, max_steps=100, step_size_cycles=50000):
    
    particles_state_future = particle_filter.particles_state.copy()
    particles_params_future = particle_filter.particles_params.copy() # Now (N, 1)
    N = particle_filter.N
    
    eol_cycles = np.full(N, -1.0) 

    for i in range(N):
        # --- START FIX: Use new 1-param model ---
        rho, D = particles_state_future[i]
        A_t = particles_params_future[i, 0] # Get the single param
        
        n_future = current_cycle
        
        for step in range(max_steps):
            if rho >= rho_thresh or D <= D_thresh:
                break
            
            # Use the new, simple, stable model
            rho = g1_linear_model(rho, A_t, delta_n=step_size_cycles)
            D = g2_stiffness_model(rho)
            
            n_future += step_size_cycles
        # --- END FIX ---
        
        eol_cycles[i] = n_future
    
    # (Rest of RUL calculation is unchanged)
    mean_eol = np.average(eol_cycles, weights=particle_filter.weights)
    rul = max(0, mean_eol - current_cycle)

    sorted_indices = np.argsort(eol_cycles)
    sorted_eols = eol_cycles[sorted_indices]
    sorted_weights = particle_filter.weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    
    lower_idx = np.searchsorted(cumulative_weights, 0.05)
    upper_idx = np.searchsorted(cumulative_weights, 0.95)
    
    lower_idx = min(lower_idx, N - 1)
    upper_idx = min(upper_idx, N - 1)
    
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
    
    # --- START FIX: Update filter creation ---
    pf = ParticleFilter(
        N=pf_config['N'],
        sigma_state_rho=pf_config['sigma_state_rho'],
        sigma_state_D=pf_config['sigma_state_D'],
        sigma_param_At=pf_config['sigma_param_At'] # Only pass 1 param noise
    )
    # --- END FIX ---
    
    pf.initialize(
        init_state=pf_config['init_state'],
        init_params_mean=pf_config['init_params_mean'],
        init_params_cov=pf_config['init_params_cov']
    )
    
    results = []
    last_cycle = 0
    
    if len(df['cycle']) > 1:
        rul_step_size = np.median(np.diff(df['cycle']))
    else:
        rul_step_size = 50000 
    
    print(f"      > Detected step size: {rul_step_size} cycles")

    for _, row in df.iterrows():
        current_cycle = row['cycle']
        measurement = [row['rho_smooth'], row['D_smooth']]
        
        delta_n = current_cycle - last_cycle
        
        if current_cycle > 0 and delta_n > 0:
            pf.predict(physics_state_transition, delta_n=delta_n)
        
        pf.update(measurement)
        
        # --- START FIX: Unpack 1 param ---
        # Estimate will return [rho, D, A_t]
        est = pf.estimate() 
        
        rul_dict = predict_probabilistic_RUL(pf, current_cycle, step_size_cycles=rul_step_size)
        
        rul_dict['cycle'] = current_cycle
        rul_dict['rho_filtered'] = est[0]
        rul_dict['D_filtered'] = est[1]
        rul_dict['A_t_filtered'] = est[2] # The 3rd element is A_t
        results.append(rul_dict)
        # --- END FIX ---
        
        last_cycle = current_cycle
        
    df_results = pd.DataFrame(results)
    return pd.merge(df.reset_index(drop=True), df_results, on='cycle')