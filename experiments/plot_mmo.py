"""
experiments/plot_mmo.py
Generates side-by-side Decision Space (PS) and Objective Space (PF) comparison plots
for all 24 CEC 2020 MMO benchmark functions.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.mmo2020_problems import MMO_BENCHMARK_REGISTRY


def plot_single_function(name, problem, ps_obt, pf_obt, ps_ref, pf_ref):
    fig = plt.figure(figsize=(13, 5.5))

    # --- Subplot 1: Decision Space (PS) ---
    if problem.n_vars == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        if len(ps_ref) > 0:
            ax1.scatter(ps_ref[:, 0], ps_ref[:, 1], ps_ref[:, 2], c="crimson", s=8, alpha=0.35, label="True PS")
        if len(ps_obt) > 0:
            ax1.scatter(ps_obt[:, 0], ps_obt[:, 1], ps_obt[:, 2], c="royalblue", s=18, marker="o", label="PESA-II PS")
        ax1.set_zlabel("$x_3$")
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        if len(ps_ref) > 0:
            ax1.scatter(ps_ref[:, 0], ps_ref[:, 1], c="crimson", s=10, alpha=0.4, label="True PS")
        if len(ps_obt) > 0:
            ax1.scatter(ps_obt[:, 0], ps_obt[:, 1], c="royalblue", s=18, marker="o", label="PESA-II PS")
        ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_title(f"{name}: Decision Space (Pareto Set)")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.legend(loc="best")

    # --- Subplot 2: Objective Space (PF) ---
    if problem.n_objs == 3:
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        if len(pf_ref) > 0:
            ax2.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], c="crimson", s=8, alpha=0.35, label="True PF")
        if len(pf_obt) > 0:
            ax2.scatter(pf_obt[:, 0], pf_obt[:, 1], pf_obt[:, 2], c="royalblue", s=18, marker="o", label="PESA-II PF")
        ax2.set_zlabel("$f_3$")
    else:
        ax2 = fig.add_subplot(1, 2, 2)
        if len(pf_ref) > 0:
            ax2.scatter(pf_ref[:, 0], pf_ref[:, 1], c="crimson", s=10, alpha=0.4, label="True PF")
        if len(pf_obt) > 0:
            ax2.scatter(pf_obt[:, 0], pf_obt[:, 1], c="royalblue", s=18, marker="o", label="PESA-II PF")
        ax2.grid(True, linestyle="--", alpha=0.4)

    ax2.set_title(f"{name}: Objective Space (Pareto Front)")
    ax2.set_xlabel("$f_1$")
    ax2.set_ylabel("$f_2$")
    ax2.legend(loc="best")

    plt.tight_layout()
    return fig


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "mmo_results")
    plots_dir = os.path.join(base_dir, "mmo_plots")
    os.makedirs(plots_dir, exist_ok=True)

    pdf_path = os.path.join(plots_dir, "MMO2020_Master_Comparison.pdf")
    print("Generating CEC 2020 MMO PS/PF Comparison Plots...")

    with PdfPages(pdf_path) as pdf:
        for name, prob_cls in MMO_BENCHMARK_REGISTRY.items():
            ps_csv = os.path.join(results_dir, f"{name}_PESA2_PS.csv")
            pf_csv = os.path.join(results_dir, f"{name}_PESA2_PF.csv")

            if not os.path.exists(ps_csv) or not os.path.exists(pf_csv):
                print(f"[!] Skipping {name} (Run experiments.run_mmo2020 first)")
                continue

            problem = prob_cls()
            ps_ref, pf_ref = problem.get_reference_data(base_dir)
            ps_obt = np.atleast_2d(np.loadtxt(ps_csv, delimiter=","))
            pf_obt = np.atleast_2d(np.loadtxt(pf_csv, delimiter=","))

            fig = plot_single_function(name, problem, ps_obt, pf_obt, ps_ref, pf_ref)

            png_path = os.path.join(plots_dir, f"MMO_{name}_PS_PF.png")
            fig.savefig(png_path, dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

            print(f"[+] Saved Graph: {png_path}")

    print(f"\nAll MMO plots generated! Images & Master PDF saved in:\n  {plots_dir}")


if __name__ == "__main__":
    main()