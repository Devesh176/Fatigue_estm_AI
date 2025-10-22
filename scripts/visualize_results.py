# scripts/5_visualize_results.py
import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(df, coupon_name, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Particle Filter Damage Prognosis: {coupon_name}', fontsize=16)
    
    # --- a) Microcrack density estimation ---
    axs[0, 0].plot(df['cycle'], df['rho_smooth'], 'ko', label='Smoothed Data', markersize=5, alpha=0.7)
    axs[0, 0].plot(df['cycle'], df['rho_filtered'], 'r-', label='Filtered Estimate', linewidth=2)
    axs[0, 0].set_title('a) Microcrack Density Estimation')
    axs[0, 0].set_xlabel('Cycle (n)')
    axs[0, 0].set_ylabel('Crack Density (ρ)')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(bottom=0)

    # --- b) Stiffness loss estimation ---
    axs[0, 1].plot(df['cycle'], df['D_smooth'], 'ko', label='Smoothed Data', markersize=5, alpha=0.7)
    axs[0, 1].plot(df['cycle'], df['D_filtered'], 'b-', label='Filtered Estimate', linewidth=2)
    axs[0, 1].set_title('b) Stiffness Loss Estimation')
    axs[0, 1].set_xlabel('Cycle (n)')
    axs[0, 1].set_ylabel('Stiffness Ratio (D)')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(top=1.0, bottom=0.0) # Set bottom to 0

    # --- c) RUL Prediction Cone ---
    true_eol = df['cycle'].iloc[-1]
    true_rul = true_eol - df['cycle']
    
    axs[1, 0].plot(df['cycle'], true_rul, 'k-', label='True RUL', linewidth=2)
    axs[1, 0].plot(df['cycle'], df['RUL'], 'g-', label='Median RUL Prediction', linewidth=2)
    axs[1, 0].fill_between(df['cycle'], df['RUL_lower'], df['RUL_upper'], color='gray', alpha=0.4, label='90% Confidence Interval')
    axs[1, 0].set_title('c) RUL Prediction')
    axs[1, 0].set_xlabel('Cycle (n)')
    axs[1, 0].set_ylabel('Remaining Useful Life (cycles)')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(bottom=0)
    
    # --- d) Parameter evolution (MODIFIED) ---
    axs[1, 1].plot(df['cycle'], df['A_t_filtered'], 'r-', label='A_t (Damage Rate) Estimate')
    axs[1, 1].set_ylabel('A_t (Damage Rate)', color='r')
    axs[1, 1].tick_params(axis='y', labelcolor='r')
    axs[1, 1].set_title('d) Parameter Evolution')
    axs[1, 1].set_xlabel('Cycle (n)')
    axs[1, 1].legend()
    # --- END MODIFICATION ---

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)