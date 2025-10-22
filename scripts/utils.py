# # scripts/utils.py
# import numpy as np

# # --- Physical Constants ---
# Ex_S = 137.5e9   # Longitudinal Modulus [Pa] (Ex)
# Ex_90 = 8.4e9     # Transverse Modulus [Pa] (Ey)
# G = 6.2e9         # Shear Modulus [Pa]
# t = 0.132e-3      # Ply thickness [m]

# n_plies_0 = 4     # n=4
# n_plies_90 = 8    # m=8
# ts = t * n_plies_0  
# t90 = t * n_plies_90 
# d0 = 0.01e-3      
# l_bar = 0.5       

# # --- Pre-calculate constants ---
# XI = np.sqrt((G / d0) * (1 / (Ex_S * ts) + 1 / (Ex_90 * t90)))
# A_CONST = (Ex_S * t90) / (Ex_90 * ts)
# R_L_CONST = (2 / XI) * np.tanh(XI * l_bar)


# def g1_paris_law_model(rho, A_t, alpha_t):
#     """
#     Calculates the *next* rho after ONE CYCLE.
#     """
#     # --- START FIX: More stable physics model ---
#     # Using 10*rho instead of 100*rho to prevent explosion
#     delta_G_t = 1.0 + 10.0 * rho
#     # --- END FIX ---
    
#     delta_rho_per_cycle = A_t * (delta_G_t ** alpha_t)
    
#     return rho + delta_rho_per_cycle

# def g2_stiffness_model(rho):
#     """
#     Calculates D from rho.
#     """
#     return 1 / (1 + A_CONST * rho * R_L_CONST)

# def physics_state_transition(particles_state, particles_params, num_cycles=1):
#     """
#     The full hierarchical state transition function.
#     Simulates N particles forward by num_cycles.
#     """
#     rho_current = particles_state[:, 0]
#     A_t = particles_params[:, 0]
#     alpha_t = particles_params[:, 1]
    
#     num_cycles = int(num_cycles)
#     if num_cycles < 1:
#         num_cycles = 1
        
#     for _ in range(num_cycles):
#         rho_current = g1_paris_law_model(rho_current, A_t, alpha_t)
#         # Add a safety clip inside the loop
#         rho_current = np.clip(rho_current, 0, 1.0) 
    
#     D_final = g2_stiffness_model(rho_current)
#     D_final = np.clip(D_final, 0, 1.0)
    
#     # Handle any NaNs that might have slipped through
#     rho_current = np.nan_to_num(rho_current, nan=1.0)
#     D_final = np.nan_to_num(D_final, nan=0.0)
    
#     return np.stack([rho_current, D_final], axis=1)

# scripts/utils.py
import numpy as np

# --- Physical Constants (from your plan & Table 1) ---
Ex_S = 137.5e9   # Longitudinal Modulus [Pa] (Ex)
Ex_90 = 8.4e9     # Transverse Modulus [Pa] (Ey)
G = 6.2e9         # Shear Modulus [Pa]
t = 0.132e-3      # Ply thickness [m]

# Laminate stacking [0_n / 90_m / 0_n]
n_plies_0 = 4     # n=4
n_plies_90 = 8    # m=8
ts = t * n_plies_0  # Thickness of 0-degree sub-laminate
t90 = t * n_plies_90 # Thickness of 90-degree sub-laminate
d0 = 0.01e-3      # Assumed resin-rich thickness
l_bar = 0.5       # Assumed dimensionless half-spacing (l)

# --- Pre-calculate constants for stiffness model (Eq 2) ---
XI = np.sqrt((G / d0) * (1 / (Ex_S * ts) + 1 / (Ex_90 * t90)))
A_CONST = (Ex_S * t90) / (Ex_90 * ts)
R_L_CONST = (2 / XI) * np.tanh(XI * l_bar)


def g1_paris_law_model(rho, A_t, alpha_t, delta_n=1):
    """
    Calculates the *next* rho after 'delta_n' CYCLES in a single step.
    (Implements Section 3.1)
    """
    # STABLE PLACEHOLDER for Delta_G_t
    # Model Delta_G_t as increasing with crack density
    delta_G_t = 1.0 + 10.0 * rho
    
    # Calculate damage rate per-cycle
    delta_rho_per_cycle = A_t * (delta_G_t ** alpha_t)
    
    # --- START FIX ---
    # Calculate total damage for this step = rate * num_cycles
    total_delta_rho = delta_rho_per_cycle * delta_n
    # --- END FIX ---
    
    return rho + total_delta_rho

def g2_stiffness_model(rho):
    """
    Calculates D from rho.
    (Implements Section 3.2)
    """
    return 1 / (1 + A_CONST * rho * R_L_CONST)

def physics_state_transition(particles_state, particles_params, num_cycles=1):
    """
    The full hierarchical state transition function.
    Simulates N particles forward by num_cycles in a SINGLE STEP.
    """
    rho_current = particles_state[:, 0]
    A_t = particles_params[:, 0]
    alpha_t = particles_params[:, 1]
    
    num_cycles = int(num_cycles)
    if num_cycles < 1:
        num_cycles = 1
        
    # --- START FIX ---
    # REMOVED THE LOOP.
    # Calculate the next state in a single shot by passing num_cycles.
    rho_next = g1_paris_law_model(rho_current, A_t, alpha_t, delta_n=num_cycles)
    # --- END FIX ---
    
    # Apply safety clips
    rho_next = np.clip(rho_next, 0, 1.0)
    
    D_final = g2_stiffness_model(rho_next)
    D_final = np.clip(D_final, 0, 1.0)
    
    # Handle any NaNs that might have slipped through
    rho_next = np.nan_to_num(rho_next, nan=1.0)
    D_final = np.nan_to_num(D_final, nan=0.0)
    
    return np.stack([rho_next, D_final], axis=1)