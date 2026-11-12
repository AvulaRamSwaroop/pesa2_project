"""
experiments/plot_convergence.py
Generates academic convergence curve plots for CEC 2022 results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'cec_results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'cec_plots')

def generate_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    dimensions = [10, 20]
    functions = range(1, 13)

    print("=" * 60)
    print("Generating Convergence Curves...")
    print("=" * 60)

    for D in dimensions:
        max_fes = 200000 if D == 10 else 1000000
        # Reconstruct the exact X-axis checkpoints used in the benchmarks
        fes_checkpoints = [int((D ** ((k / 5.0) - 3.0)) * max_fes) for k in range(16)]

        for func_no in functions:
            file_name = os.path.join(RESULTS_DIR, f"DE_{func_no}_{D}.txt")
            
            if not os.path.exists(file_name):
                continue
                
            # Load the matrix
            matrix = np.loadtxt(file_name, delimiter="\t")
            
            # Get the mean error at each of the 16 evaluation checkpoints
            mean_convergence = np.mean(matrix[:16, :], axis=1)

            # Plotting
            plt.figure(figsize=(8, 5))
            
            # Using a logarithmic scale for the Y-axis is standard for CEC graphs 
            # because errors drop from massive numbers down to decimals.
            plt.semilogy(fes_checkpoints, mean_convergence, marker='o', linestyle='-', color='b', linewidth=2, label="DE Algorithm")
            
            plt.title(f"Convergence Curve: CEC 2022 Function {func_no} (D={D})", fontsize=14, fontweight='bold')
            plt.xlabel("Function Evaluations (FEs)", fontsize=12)
            plt.ylabel("Mean Error (Log Scale)", fontsize=12)
            plt.grid(True, which="both", ls="--", alpha=0.6)
            plt.legend(fontsize=12)
            plt.tight_layout()

            # Save the image
            plot_file = os.path.join(PLOTS_DIR, f"Convergence_F{func_no}_D{D}.png")
            plt.savefig(plot_file, dpi=300)  # High resolution for Word/PDF reports
            plt.close()
            
            print(f"   [+] Saved Graph: {plot_file}")

    print(f"\nAll plots generated! You can find the images in: {PLOTS_DIR}")

if __name__ == "__main__":
    generate_plots()