"""
experiments/run_cec2022.py
Executes the CEC 2022 Single Objective Benchmark Suite.
Generates the required 17x30 matrix txt files.
"""
import os
import sys
import argparse
import numpy as np

# Resolve absolute paths
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
CEC_DIR = os.path.join(PROJECT_ROOT, 'cec2022', 'Python')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'cec_results')

# Change directory BEFORE importing the CEC code
os.chdir(CEC_DIR)
sys.path.append(CEC_DIR)
from CEC2022 import cec22_test_func

# Add root back to path to import your DE algorithm
sys.path.append(PROJECT_ROOT)
from algorithms.de import DifferentialEvolution

def load_random_seeds():
    seed_file = os.path.join(CEC_DIR, "input_data", "Rand_Seeds.txt")
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Missing seed file at: {seed_file}")
    return np.loadtxt(seed_file, dtype=float).astype(int)

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=int, help="Specify function number to run (1-12)")
    args = parser.parse_args()

    dimensions = [10, 20]
    num_runs = 30
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rand_seeds = load_random_seeds()

    # Determine which functions to run based on terminal command
    functions_to_run = [args.func] if args.func else range(1, 13)

    print("=" * 60)
    print(f"Starting CEC 2022 Benchmark Suite (DE Algorithm)")
    print("=" * 60)

    for D in dimensions:
        max_fes = 200000 if D == 10 else 1000000
        bounds = [[-100, 100]] * D
        
        checkpoints = [int((D ** ((k / 5.0) - 3.0)) * max_fes) for k in range(16)]
        
        for func_no in functions_to_run:
            print(f"\n-> Optimizing Function F{func_no} | D = {D} | {num_runs} Runs")
            results_matrix = np.zeros((17, num_runs))
            
            for run_id in range(1, num_runs + 1):
                print(f"   - Executing Run {run_id}/{num_runs} ...", end="\r", flush=True)
                
                seed_ind = int((D / 10 * func_no * num_runs + run_id) - num_runs)
                seed_ind = (seed_ind % 1000)
                run_seed = rand_seeds[seed_ind]
                
                def objective(x):
                    val = cec22_test_func(x, D, 1, func_no)
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return float(val[0])
                
                optimizer = DifferentialEvolution(
                    objective_func=objective,
                    bounds=bounds,
                    max_evals=max_fes,
                    seed=run_seed
                )
                
                run_results = optimizer.run(checkpoints)
                results_matrix[:, run_id - 1] = run_results
            
            print(" " * 50, end="\r")
            
            file_name = os.path.join(RESULTS_DIR, f"DE_{func_no}_{D}.txt")
            np.savetxt(file_name, results_matrix, fmt="%.8e", delimiter="\t")
            print(f"   [+] Saved {file_name}")

if __name__ == "__main__":
    main()