"""
experiments/calc_statistics.py
Calculates Best, Worst, Median, Mean, and Std Dev for CEC 2022 matrices.
Exports the results to a CSV file for easy copy-pasting into reports.
"""
import os
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'cec_results')

def calculate_statistics():
    dimensions = [10, 20]
    functions = range(1, 13)
    
    # Store data for the final table
    table_data = []

    print("=" * 60)
    print("Calculating CEC 2022 Statistics...")
    print("=" * 60)

    for D in dimensions:
        for func_no in functions:
            file_name = os.path.join(RESULTS_DIR, f"DE_{func_no}_{D}.txt")
            
            if not os.path.exists(file_name):
                print(f"[!] Missing file: {file_name}")
                continue
                
            # Load the 17x30 matrix
            matrix = np.loadtxt(file_name, delimiter="\t")
            
            # The final function evaluations are in the 17th row (index 16)
            # Some functions might have stopped exactly at the 16th row (index 15) if precision hit 1e-8 early
            # We take the best (minimum) error achieved across all checkpoints for each run
            final_errors = np.min(matrix[:16, :], axis=0) 
            
            best = np.min(final_errors)
            worst = np.max(final_errors)
            median = np.median(final_errors)
            mean = np.mean(final_errors)
            std = np.std(final_errors)
            
            table_data.append({
                "Dimension": D,
                "Function": f"F{func_no}",
                "Best": f"{best:.4e}",
                "Worst": f"{worst:.4e}",
                "Median": f"{median:.4e}",
                "Mean": f"{mean:.4e}",
                "Std Dev": f"{std:.4e}"
            })
            
            print(f"Processed F{func_no} (D={D}) -> Mean: {mean:.4e}")

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(table_data)
    output_csv = os.path.join(PROJECT_ROOT, "cec_statistics_table.csv")
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"Success! Statistics table saved to:\n{output_csv}")
    print("You can open this CSV in Excel and copy it straight to your report.")

if __name__ == "__main__":
    calculate_statistics()