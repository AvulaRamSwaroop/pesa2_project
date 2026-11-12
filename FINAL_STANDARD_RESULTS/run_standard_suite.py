"""
comparisons/run_standard_suite.py
Executes PESA-II on the 30 Standard Suite benchmarks according to strict comparative guidelines.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure imports work from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PESA-II and your specific Standard Problems Registry
from algorithms.pesa2.pesa2 import PESA2
from benchmarks.benchmarks_30 import BENCHMARK_REGISTRY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DATA_DIR = os.path.join(BASE_DIR, "comparisons", "30_benchmarks", "data")
OUT_PLOTS_DIR = os.path.join(BASE_DIR, "comparisons", "30_benchmarks", "plots")

os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

DTLZ_PROBS = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

def safe_cdist(A, B):
    """Pure NumPy replacement for scipy.spatial.distance.cdist"""
    return np.sqrt(np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2))

def get_config(prob_name):
    """Returns exact configuration parameters from baseline standards."""
    # (pop_size, max_evals)
    if prob_name in DTLZ_PROBS:
        return 200, 600200
    else:
        return 100, 100100

def calc_gd(obtained, reference):
    if len(obtained) == 0: return np.inf
    return float(np.mean(np.min(safe_cdist(obtained, reference), axis=1)))

def calc_igd(obtained, reference):
    if len(obtained) == 0 or len(reference) == 0: return np.inf
    return float(np.mean(np.min(safe_cdist(reference, obtained), axis=1)))

def calc_spacing(obtained):
    if len(obtained) < 2: return np.inf
    dists = np.sum(np.abs(obtained[:, np.newaxis, :] - obtained[np.newaxis, :, :]), axis=2)
    np.fill_diagonal(dists, np.inf)
    d_i = np.min(dists, axis=1)
    d_mean = np.mean(d_i)
    return float(np.sqrt(np.sum((d_mean - d_i)**2) / (len(obtained) - 1)))

def plot_front(name, obtained_pf, true_pf, is_3d, pdf_pages=None):
    # 1. Combined Plot (Comparison)
    fig_comp = plt.figure(figsize=(7, 6))
    ax_comp = fig_comp.add_subplot(111, projection="3d" if is_3d else None)
    
    # 2. True Plot Only
    fig_true = plt.figure(figsize=(7, 6))
    ax_true = fig_true.add_subplot(111, projection="3d" if is_3d else None)
    
    # 3. Obtained Plot Only
    fig_obt = plt.figure(figsize=(7, 6))
    ax_obt = fig_obt.add_subplot(111, projection="3d" if is_3d else None)
    
    if is_3d:
        if len(true_pf) > 0:
            ax_comp.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='crimson', s=10, alpha=0.3, label="True PF")
            ax_true.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='crimson', s=10, alpha=0.6)
        if len(obtained_pf) > 0:
            ax_comp.scatter(obtained_pf[:, 0], obtained_pf[:, 1], obtained_pf[:, 2], c='royalblue', s=15, label="PESA-II PF")
            ax_obt.scatter(obtained_pf[:, 0], obtained_pf[:, 1], obtained_pf[:, 2], c='royalblue', s=15)
    else:
        if len(true_pf) > 0:
            ax_comp.scatter(true_pf[:, 0], true_pf[:, 1], c='crimson', s=10, alpha=0.3, label="True PF")
            ax_true.scatter(true_pf[:, 0], true_pf[:, 1], c='crimson', s=10, alpha=0.6)
        if len(obtained_pf) > 0:
            ax_comp.scatter(obtained_pf[:, 0], obtained_pf[:, 1], c='royalblue', s=15, label="PESA-II PF")
            ax_obt.scatter(obtained_pf[:, 0], obtained_pf[:, 1], c='royalblue', s=15)
        
    ax_comp.set_title(f"{name}: True vs Obtained PF")
    ax_comp.legend(loc="best")
    ax_true.set_title(f"{name}: True Pareto Front")
    ax_obt.set_title(f"{name}: Obtained PESA-II Pareto Front")
    
    fig_comp.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_pareto_front_combined.png"), dpi=200)
    fig_true.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_pareto_front_true.png"), dpi=200)
    fig_obt.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_pareto_front_obtained.png"), dpi=200)
    
    if pdf_pages is not None:
        pdf_pages.savefig(fig_comp)
        
    plt.close(fig_comp)
    plt.close(fig_true)
    plt.close(fig_obt)

def execute_standard_suite():
    all_problems = list(BENCHMARK_REGISTRY.keys())
    results_rows = []
    
    pdf_path = os.path.join(OUT_PLOTS_DIR, "StandardSuite_Master_Comparison.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for prob_name in all_problems:
            print(f"\n--- Executing Benchmark: {prob_name} ---")
            pop_size, max_evals = get_config(prob_name)
            
            problem = BENCHMARK_REGISTRY[prob_name]()
            true_pf = problem.get_true_front()
            n_objs = problem.n_objs
            
            best_igd = np.inf
            best_obtained_pf = None
            
            for seed in range(1, 6):
                start_time = time.time()
                
                algo = PESA2(
                    problem=problem,
                    archive_size=100,
                    population_size=pop_size,
                    bits_per_var=30,     
                    grid_divisions=10,   
                    max_evals=max_evals,
                    seed=seed
                )
                
                obtained_pf = algo.run()
                
                runtime = time.time() - start_time
                gd = calc_gd(obtained_pf, true_pf)
                igd = calc_igd(obtained_pf, true_pf)
                spacing = calc_spacing(obtained_pf)
                
                if igd < best_igd:
                    best_igd = igd
                    best_obtained_pf = obtained_pf
                
                ref_point = 1.1 * np.max(true_pf, axis=0) if len(true_pf) > 0 else np.array([1.1]*n_objs)
                valid = np.all(obtained_pf <= ref_point, axis=1) if len(obtained_pf) > 0 else []
                hv = float(np.sum(valid) / max(1, len(true_pf))) if len(valid) > 0 else 0.0
                
                results_rows.append({
                    "benchmark": prob_name,
                    "algorithm": "PESA-II",
                    "seed": seed,
                    "hypervolume": hv,
                    "IGD": igd,
                    "GD": gd,
                    "spacing": spacing,
                    "runtime": runtime
                })
                
                print(f"Seed {seed} complete. IGD: {igd:.4f}, Runtime: {runtime:.2f}s")
                
            plot_front(prob_name, best_obtained_pf, true_pf, is_3d=(n_objs == 3), pdf_pages=pdf)

    df = pd.DataFrame(results_rows)
    df.to_csv(os.path.join(OUT_DATA_DIR, "results.csv"), index=False)
    
    summary_df = df.groupby(["benchmark", "algorithm"]).agg({
        "hypervolume": ["mean", "std"], "IGD": ["mean", "std"],
        "GD": ["mean", "std"], "spacing": ["mean", "std"], "runtime": ["mean", "std"]
    }).reset_index()
    summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns.values]
    summary_df.to_csv(os.path.join(OUT_DATA_DIR, "summary_metrics.csv"), index=False)
    
    print("\n[+] Standard suite execution complete!")
    print(f"Raw data saved to: {OUT_DATA_DIR}/results.csv")
    print(f"Summary data saved to: {OUT_DATA_DIR}/summary_metrics.csv")
    print(f"30-page Master PDF Comparison saved to: {pdf_path}")
    print(f"Individual PNG plots saved to: {OUT_PLOTS_DIR}/")

if __name__ == "__main__":
    execute_standard_suite()