"""
experiments/run.py
Main execution runner: Triggers the algorithm, generates local PNGs, logs metrics,
and compiles a single master PDF containing all benchmark comparisons.
"""
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from benchmarks.benchmarks_30 import BENCHMARK_REGISTRY
from algorithms.pesa2.pesa2 import PESA2
from metrics_initial.metrics import calc_igd, calc_gd, calc_spacing
from visualization.plots import create_and_save_plots
from experiments.report import save_metrics

def run_benchmark(name, seed):
    """Executes a single benchmark run."""
    problem = BENCHMARK_REGISTRY[name]()
    is_3d = problem.n_objs == 3
    
    # Grid/Evaluation scaling based on complexity
    arch_size = 250 if is_3d else 100
    max_evals = 25000 if is_3d else 15000
    grid_divisions = 8 if is_3d else 16

    pesa = PESA2(
        problem=problem,
        archive_size=arch_size,
        population_size=10,
        bits_per_var=30,
        grid_divisions=grid_divisions,
        max_evals=max_evals,
        seed=seed
    )

    obtained = pesa.run()
    true_pf = problem.get_true_front(n_points=600)
    
    metrics = {
        "IGD": calc_igd(obtained, true_pf),
        "GD": calc_gd(obtained, true_pf),
        "Spacing": calc_spacing(obtained)
    }
    
    return obtained, true_pf, is_3d, metrics

def main():
    # 1. Setup timestamp and output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = 42
    os.makedirs("results", exist_ok=True)
    
    pdf_filename = os.path.join("results", f"Benchmark_Comparison_{timestamp}.pdf")

    print("=" * 70)
    print(f"Starting PESA-II Benchmark Batch")
    print(f"Seed: {seed} | Timestamp: {timestamp}")
    print("=" * 70)

    # 2. Open the PDF context manager to combine plots into a single file
    with PdfPages(pdf_filename) as pdf:
        
        for idx, name in enumerate(BENCHMARK_REGISTRY.keys(), start=1):
            print(f"[{idx}/25] Evaluating {name}...")
            
            # Run the algorithm
            obtained, true_pf, is_3d, metrics = run_benchmark(name, seed=seed)
            
            # Save 3 PNGs to plots/name/pngs and return the side-by-side figure
            fig_pdf = create_and_save_plots(name, obtained, true_pf, is_3d, timestamp)
            
            # Save metrics to the central CSV file
            save_metrics(name, metrics, timestamp, seed)
            
            # Append the side-by-side figure as a new page in the PDF, then close it from memory
            pdf.savefig(fig_pdf)
            plt.close(fig_pdf)
            
            print(f"   -> IGD: {metrics['IGD']:.4f} | GD: {metrics['GD']:.4f} | SP: {metrics['Spacing']:.4f}")

    print("\n" + "=" * 70)
    print("Run Complete!")
    print("- Individual PNGs saved in: ./plots/<benchmark_name>/pngs/")
    print("- Metrics appended to:      ./results/master_metrics_report.csv")
    print(f"- Master PDF generated at:  {pdf_filename}")

if __name__ == "__main__":
    main()