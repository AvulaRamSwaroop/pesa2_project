"""
comparisons/run_mmo2020_suite.py
Executes PESA-II on the 24 CEC2020 MMO benchmarks.
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

# Import PESA-II and the CEC2020 MMO Registry
from algorithms.pesa2.pesa2 import PESA2
from benchmarks.mmo2020_problems import MMO_BENCHMARK_REGISTRY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DATA_DIR = os.path.join(BASE_DIR, "comparisons", "mmo2020_benchmarks", "data")
OUT_PLOTS_DIR = os.path.join(BASE_DIR, "comparisons", "mmo2020_benchmarks", "plots")

os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

def safe_cdist(A, B):
    return np.sqrt(np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2))

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

def plot_comparison(name, obtained_data, true_data, is_3d, plot_type, pdf_pages=None):
    fig_comp = plt.figure(figsize=(7, 6))
    ax_comp = fig_comp.add_subplot(111, projection="3d" if is_3d else None)
    
    fig_true = plt.figure(figsize=(7, 6))
    ax_true = fig_true.add_subplot(111, projection="3d" if is_3d else None)
    
    fig_obt = plt.figure(figsize=(7, 6))
    ax_obt = fig_obt.add_subplot(111, projection="3d" if is_3d else None)
    
    dim = 3 if is_3d else 2
    
    # Safely slice to 2D or 3D for visualization (decision space might have > 3 vars)
    t_data = true_data[:, :dim] if len(true_data) > 0 else np.array([])
    o_data = obtained_data[:, :dim] if len(obtained_data) > 0 else np.array([])
    
    if is_3d:
        if len(t_data) > 0:
            ax_comp.scatter(t_data[:, 0], t_data[:, 1], t_data[:, 2], c='crimson', s=10, alpha=0.3, label=f"True {plot_type.upper()}")
            ax_true.scatter(t_data[:, 0], t_data[:, 1], t_data[:, 2], c='crimson', s=10, alpha=0.6)
        if len(o_data) > 0:
            ax_comp.scatter(o_data[:, 0], o_data[:, 1], o_data[:, 2], c='royalblue', s=15, label=f"Obtained {plot_type.upper()}")
            ax_obt.scatter(o_data[:, 0], o_data[:, 1], o_data[:, 2], c='royalblue', s=15)
    else:
        if len(t_data) > 0:
            ax_comp.scatter(t_data[:, 0], t_data[:, 1], c='crimson', s=10, alpha=0.3, label=f"True {plot_type.upper()}")
            ax_true.scatter(t_data[:, 0], t_data[:, 1], c='crimson', s=10, alpha=0.6)
        if len(o_data) > 0:
            ax_comp.scatter(o_data[:, 0], o_data[:, 1], c='royalblue', s=15, label=f"Obtained {plot_type.upper()}")
            ax_obt.scatter(o_data[:, 0], o_data[:, 1], c='royalblue', s=15)
            
    title_suffix = "Objective Space (PF)" if plot_type == "pareto_front" else "Decision Space (PS)"
    ax_comp.set_title(f"{name}: True vs Obtained - {title_suffix}")
    ax_comp.legend(loc="best")
    ax_true.set_title(f"{name}: True {title_suffix}")
    ax_obt.set_title(f"{name}: Obtained {title_suffix}")
    
    fig_comp.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_{plot_type}_combined.png"), dpi=200)
    fig_true.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_{plot_type}_true.png"), dpi=200)
    fig_obt.savefig(os.path.join(OUT_PLOTS_DIR, f"{name}_{plot_type}_obtained.png"), dpi=200)
    
    if pdf_pages is not None:
        pdf_pages.savefig(fig_comp)
        
    plt.close(fig_comp)
    plt.close(fig_true)
    plt.close(fig_obt)

def execute_mmo2020_suite():
    all_problems = list(MMO_BENCHMARK_REGISTRY.keys())
    results_rows = []
    
    pdf_path = os.path.join(OUT_PLOTS_DIR, "MMO2020_Master_Comparison.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for prob_name in all_problems:
            print(f"\n--- Executing MMO Benchmark: {prob_name} ---")
            
            # CEC2020 Standard Parameters
            pop_size = 100
            max_evals = 20100
            
            problem = MMO_BENCHMARK_REGISTRY[prob_name]()
            # MMO problems typically return (true_ps, true_pf)
            true_ps, true_pf = problem.get_reference_data()
            n_objs = problem.n_objs
            n_vars = problem.n_vars
            
            best_igd = np.inf
            best_obtained_pf = None
            best_obtained_ps = None
            
            # CEC2020 requires 3 independent runs (Seeds 1-3)
            for seed in range(1, 4):
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
                
                # Extract Decision Space (Pareto Set) from the archive chromosomes
                obtained_ps = np.array([algo.decode(sol.chromosome) for sol in algo.archive])
                
                runtime = time.time() - start_time
                gd = calc_gd(obtained_pf, true_pf)
                igd = calc_igd(obtained_pf, true_pf)
                spacing = calc_spacing(obtained_pf)
                
                # Track the best run for plotting
                if igd < best_igd:
                    best_igd = igd
                    best_obtained_pf = obtained_pf
                    best_obtained_ps = obtained_ps
                
                # Approximate Hypervolume
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
                
            # Plot both Pareto Front (Objective Space) and Pareto Set (Decision Space) for the best run
            plot_comparison(prob_name, best_obtained_pf, true_pf, is_3d=(n_objs == 3), plot_type="pareto_front", pdf_pages=pdf)
            plot_comparison(prob_name, best_obtained_ps, true_ps, is_3d=(n_vars >= 3), plot_type="pareto_set", pdf_pages=pdf)

            # Progressive save inside the loop
            df = pd.DataFrame(results_rows)
            df.to_csv(os.path.join(OUT_DATA_DIR, "cec2020_mmo_results.csv"), index=False)
            
            summary_df = df.groupby(["benchmark", "algorithm"]).agg({
                "hypervolume": ["mean", "std"], "IGD": ["mean", "std"],
                "GD": ["mean", "std"], "spacing": ["mean", "std"], "runtime": ["mean", "std"]
            }).reset_index()
            summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns.values]
            summary_df.to_csv(os.path.join(OUT_DATA_DIR, "cec2020_mmo_summary.csv"), index=False)

    print("\n[+] MMO2020 suite execution complete!")
    print(f"Raw data saved to: {OUT_DATA_DIR}/cec2020_mmo_results.csv")
    print(f"Summary data saved to: {OUT_DATA_DIR}/cec2020_mmo_summary.csv")
    print(f"Master PDF Comparison saved to: {pdf_path}")
    print(f"Individual PNG plots saved to: {OUT_PLOTS_DIR}/")

if __name__ == "__main__":
    execute_mmo2020_suite()