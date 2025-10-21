# # scripts/utils.py
# import numpy as np

# # --- Physical Constants (from Table 1 & text in the paper) ---
# # [cite: 157, 76]
# Ex_S = 137.5e9   # Longitudinal Modulus [Pa]
# Ex_90 = 8.4e9     # Transverse Modulus [Pa]
# G = 6.2e9         # Shear Modulus [Pa]
# ts = 0.132e-3 * 4 # Thickness of 0-degree sub-laminate (assuming 4 plies)
# t90 = 0.132e-3 * 8 # Thickness of 90-degree sub-laminate (assuming 8 plies)
# d0 = 0.01e-3      # Resin-rich thickness (assumed typical value)
# l_bar = 0.5       # Dimensionless half-spacing between cracks (assumed constant for simplicity)

# # Paris' Law parameters (these are what the PF will estimate, we start with a guess)
# # Based on Figure 2d, initial guess might be around these values
# A_t = 8.5e-5
# alpha_t = 1.8

# # --- Physics-Based Models ---

# def calculate_xi():
#     """Calculates the xi term from Equation 2.""" # [cite: 57]
#     term1 = 1 / (Ex_S * ts)
#     term2 = 1 / (Ex_90 * t90)
#     return np.sqrt((G / d0) * (term1 + term2))

# def stiffness_reduction_model(rho, xi):
#     """
#     Calculates stiffness reduction 'D' based on crack density 'rho'.
#     This is based on Equation 6 from the paper.
#     """ # [cite: 72]
#     a = (Ex_S * t90) / (Ex_90 * ts) # [cite: 76]
#     R_l = (2 / xi) * np.tanh(xi * l_bar) # [cite: 74]
#     D = 1 / (1 + a * rho * R_l)
#     return D

# def paris_law_crack_growth(rho, S_max=0.8*550e6, R=0.14):
#     """
#     Calculates the next crack density rho_n from rho_n-1.
#     This is the discrete form from Equation 5.
#     Here, we simplify Delta_G_t to be a function of rho.
#     For this example, we'll use a simplified energy release rate
#     that increases with crack density. A full implementation would
#     recalculate G_t based on stress analysis (Eq. 1).
#     """ # [cite: 65]
#     # Simplified placeholder for the complex Energy Release Rate (G_t)
#     # A real implementation requires the variational stress analysis from the paper.
#     # We model it as a value that increases as cracks form.
#     delta_G_t = 100 + 200 * rho

#     # Equation 5: rho_n = rho_{n-1} + A_t * (Delta_G_t)^alpha_t
#     delta_rho = A_t * (delta_G_t ** alpha_t)
    
#     # We scale this per-cycle damage by the measurement interval
#     # Assuming measurements are taken every 5000 cycles
#     return rho + delta_rho * 5000

# def physics_state_transition(particles):
#     """
#     The full state transition function for the particle filter.
#     1. Propagate micro-cracks (rho) using Paris' Law.
#     2. Calculate the corresponding macro-stiffness (D).
#     """
#     rho_n_minus_1 = particles[:, 0]
    
#     # 1. Predict next rho
#     rho_n = paris_law_crack_growth(rho_n_minus_1)
    
#     # 2. Calculate corresponding D
#     xi = calculate_xi()
#     D_n = stiffness_reduction_model(rho_n, xi)
    
#     return np.stack([rho_n, D_n], axis=1)

# scripts/utils.py
import numpy as np

# --- Physical Constants (from Table 1 & text) ---
Ex_S = 137.5e9   # Longitudinal Modulus [Pa] [cite: 157]
Ex_90 = 8.4e9     # Transverse Modulus [Pa] [cite: 157]
G = 6.2e9         # Shear Modulus [Pa] [cite: 157]
t = 0.132e-3      # Ply thickness [m] [cite: 157]

# Laminate stacking assumptions [0_n / 90_m / 0_n]
n_plies_0 = 4
n_plies_90 = 8
ts = t * n_plies_0  # Thickness of 0-degree sub-laminate
t90 = t * n_plies_90 # Thickness of 90-degree sub-laminate
d0 = 0.01e-3      # Assumed resin-rich thickness
l_bar = 0.5       # Assumed dimensionless half-spacing

# --- Pre-calculate constants for speed ---
XI = np.sqrt((G / d0) * (1 / (Ex_S * ts) + 1 / (Ex_90 * t90)))
A_CONST = (Ex_S * t90) / (Ex_90 * ts)
R_L_CONST = (2 / XI) * np.tanh(XI * l_bar)


def g1_paris_law_model(rho_n_minus_1, A_t, alpha_t):
    """
    State transition g1 for micro-crack density (rho).
    Implements rho_n = rho_{n-1} + A_t * (Delta_G_t)^alpha_t 
    """
    
    # --- STABLE PLACEHOLDER for Delta_G_t ---
    # The real Eq (1) is very complex. We use a stable model
    # where G_t increases with crack density rho.
    # This must be non-zero to prevent math errors when alpha_t is high.
    delta_G_t = 1.0 + 100 * rho_n_minus_1
    
    # Calculate change in rho (per cycle)
    # NOTE: The paper's data is sampled every ~5000 cycles
    # We'll assume the model predicts the change over that *step*
    delta_rho = A_t * (delta_G_t ** alpha_t)
    
    rho_n = rho_n_minus_1 + delta_rho
    return rho_n

def g2_stiffness_model(rho_n):
    """
    State transition g2 for stiffness ratio (D).
    Implements D = 1 / (1 + a * rho * R(l)) 
    """
    D_n = 1 / (1 + A_CONST * rho_n * R_L_CONST)
    return D_n

def physics_state_transition(particles):
    """
    The full hierarchical state transition function for the particle filter.
    Takes N particles (rho, D, A_t, alpha_t) and predicts (rho_n, D_n)
    """
    # Unpack particle data
    rho_n_minus_1 = particles[:, 0]
    # D_n_minus_1 is not needed for prediction, as D_n depends on rho_n
    A_t = particles[:, 2]
    alpha_t = particles[:, 3]
    
    # 1. Propagate micro-cracks (rho) using Paris' Law
    rho_n = g1_paris_law_model(rho_n_minus_1, A_t, alpha_t)
    
    # 2. Calculate the corresponding macro-stiffness (D)
    D_n = g2_stiffness_model(rho_n)
    
    # Return the new state [rho_n, D_n]
    return np.stack([rho_n, D_n], axis=1)   