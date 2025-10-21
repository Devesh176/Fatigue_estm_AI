# # 5_visualize_results.py
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_all_results(df):
#     """
#     Generates plots similar to Figure 2 in the paper.
#     """
#     fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
#     # a) Microcrack density estimation
#     axs[0, 0].plot(df['cycle'], df['rho_smooth'], 'ko', label='Smoothed Data')
#     axs[0, 0].plot(df['cycle'], df['rho_filtered'], 'r-', label='Filtered Estimate')
#     axs[0, 0].set_title('a) Microcrack Density Estimation')
#     axs[0, 0].set_xlabel('Cycle (n)')
#     axs[0, 0].set_ylabel('Crack Density (ρ)')
#     axs[0, 0].legend()

#     # b) Stiffness loss estimation
#     axs[0, 1].plot(df['cycle'], df['D_smooth'], 'ko', label='Smoothed Data')
#     axs[0, 1].plot(df['cycle'], df['D_filtered'], 'b-', label='Filtered Estimate')
#     axs[0, 1].set_title('b) Stiffness Loss Estimation')
#     axs[0, 1].set_xlabel('Cycle (n)')
#     axs[0, 1].set_ylabel('Stiffness Ratio (D)')
#     axs[0, 1].legend()

#     # c) RUL Prediction Cone
#     true_eol = df['cycle'].iloc[-1]
#     true_rul = true_eol - df['cycle']
    
#     axs[1, 0].plot(df['cycle'], true_rul, 'k-', label='True RUL')
#     axs[1, 0].plot(df['cycle'], df['RUL'], 'go', ls='-', label='Median RUL Prediction')
#     axs[1, 0].fill_between(df['cycle'], df['RUL_lower'], df['RUL_upper'], color='gray', alpha=0.4, label='90% Confidence Interval')
#     axs[1, 0].set_title('c) RUL Prediction')
#     axs[1, 0].set_xlabel('Cycle (n)')
#     axs[1, 0].set_ylabel('Remaining Useful Life (cycles)')
#     axs[1, 0].legend()
    
#     # d) Parameter evolution (would require modifications to PF to track parameters)
#     axs[1, 1].text(0.5, 0.5, 'd) Parameter Evolution (requires PF modification)', ha='center')
#     axs[1, 1].set_title('d) Parameter Evolution')

#     plt.tight_layout()
#     plt.show()
# 5_visualize_results.py
import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(df):
    """
    Generates all 4 plots similar to Figure 2 in the paper.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Particle Filter Damage Prognosis Results', fontsize=16)
    
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
    axs[0, 1].set_ylim(top=1.0)

    # --- c) RUL Prediction Cone ---
    # Calculate true RUL based on the last cycle in the data
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
    
    # --- d) Parameter evolution ---
    ax_d2 = axs[1, 1].twinx() # Create a second y-axis
    
    # Plot A_t (theta_1)
    axs[1, 1].plot(df['cycle'], df['A_t_filtered'], 'r-', label='A_t (θ1) Estimate')
    axs[1, 1].set_ylabel('A_t (θ1)', color='r')
    axs[1, 1].tick_params(axis='y', labelcolor='r')
    
    # Plot alpha_t (theta_2)
    ax_d2.plot(df['cycle'], df['alpha_t_filtered'], 'b-', label='α_t (θ2) Estimate')
    ax_d2.set_ylabel('α_t (θ2)', color='b')
    ax_d2.tick_params(axis='y', labelcolor='b')

    axs[1, 1].set_title('d) Parameter Evolution')
    axs[1, 1].set_xlabel('Cycle (n)')
    
    # Add legends for twin axes
    lines1, labels1 = axs[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()