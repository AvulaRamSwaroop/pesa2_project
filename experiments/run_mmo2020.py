"""
experiments/run_mmo2020.py
Executes PESA-II on the 24 CEC 2020 MMO benchmark problems, saves PS/PF outputs,
and generates the statistical summary tables (Best, Worst, Mean, Median, Std).
"""
import os
import sys
import time
import argparse
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.pesa2.pesa2 import PESA2
from benchmarks.mmo2020_problems import (
    MMO_BENCHMARK_REGISTRY,
    calc_igd_metric,
    calc_cr_and_rpsp,
    calc_rhv,
)


def run_single_mmo(problem, seed, fast_mode=False):
    """Runs PESA2 on a single MMO problem instance and returns PS, PF, and 4 metrics."""
    problem.clear_history()
    is_3d = problem.n_objs == 3

    # CEC 2020 MMO Rules scaled for fast execution vs full competition mode
    if fast_mode:
        arch_size = 100 * problem.n_ops
        max_evals = 3000 * problem.n_ops
    else:
        arch_size = 200 * problem.n_ops
        max_evals = 10000 * problem.n_ops

    grid_divisions = 10 if is_3d else 15

    pesa = PESA2(
        problem=problem,
        archive_size=arch_size,
        population_size=20,
        bits_per_var=20 if fast_mode else 30,
        grid_divisions=grid_divisions,
        max_evals=max_evals,
        seed=seed,
    )

    obtained_pf = np.atleast_2d(np.asarray(pesa.run(), dtype=float))
    obtained_ps = problem.recover_ps_from_pf(obtained_pf)
    true_ps, true_pf = problem.get_reference_data()

    igdx = calc_igd_metric(obtained_ps, true_ps)
    igdf = calc_igd_metric(obtained_pf, true_pf)
    rpsp = calc_cr_and_rpsp(obtained_ps, true_ps, igdx)
    rhv = calc_rhv(obtained_pf, true_pf)

    metrics = {"rPSP": rpsp, "rHV": rhv, "IGDX": igdx, "IGDF": igdf}
    return obtained_ps, obtained_pf, metrics


def run_all_mmo2020(target_funcs=None, n_runs=5, fast_mode=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "mmo_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter registry if specific functions were requested
    if target_funcs:
        registry = {k: v for k, v in MMO_BENCHMARK_REGISTRY.items() if k in target_funcs}
    else:
        registry = MMO_BENCHMARK_REGISTRY

    indicators = ["rPSP", "rHV", "IGDX", "IGDF"]
    all_stats = {ind: {} for ind in indicators}
    csv_rows = []

    print("=" * 75)
    print(f"Starting CEC 2020 MMO Benchmark Suite ({len(registry)} Functions | {n_runs} Runs Each)")
    print(f"Target Functions: {list(registry.keys())}")
    print(f"Mode: {'FAST' if fast_mode else 'FULL CEC COMPETITION'} | Output: {results_dir}")
    print("=" * 75)

    for idx, (name, prob_cls) in enumerate(registry.items(), start=1):
        t0 = time.time()
        problem = prob_cls()
        run_metrics = {ind: [] for ind in indicators}

        best_igdx = np.inf
        best_ps, best_pf = None, None

        for r in range(n_runs):
            seed = 42 + r
            ps, pf, m = run_single_mmo(problem, seed=seed, fast_mode=fast_mode)
            for ind in indicators:
                run_metrics[ind].append(m[ind])

            if m["IGDX"] < best_igdx:
                best_igdx = m["IGDX"]
                best_ps, best_pf = ps, pf

        # Save best run PS and PF for plotting
        np.savetxt(os.path.join(results_dir, f"{name}_PESA2_PS.csv"), best_ps, delimiter=",")
        np.savetxt(os.path.join(results_dir, f"{name}_PESA2_PF.csv"), best_pf, delimiter=",")

        # Compute Best, Worst, Mean, Median, Std for each indicator
        row_summary = {"Function": name}
        for ind in indicators:
            vals = np.array(run_metrics[ind], dtype=float)
            stats = [
                float(np.min(vals)),
                float(np.max(vals)),
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.std(vals)),
            ]
            all_stats[ind][name] = list(vals) + stats
            row_summary[f"{ind}_Best"] = f"{stats[0]:.4f}"
            row_summary[f"{ind}_Mean"] = f"{stats[2]:.4f}"
            row_summary[f"{ind}_Std"] = f"{stats[4]:.4f}"

        csv_rows.append(row_summary)
        elapsed = time.time() - t0
        print(
            f"[{idx:02d}/{len(registry)}] {name:<10} ({elapsed:4.1f}s) | "
            f"rPSP: {row_summary['rPSP_Mean']} | IGDX: {row_summary['IGDX_Mean']} | "
            f"IGDF: {row_summary['IGDF_Mean']} | rHV: {row_summary['rHV_Mean']}"
        )

    # Note: If running a subset, we only append/update the stats to avoid overwriting previous results.
    # 1. Update master CSV table
    csv_path = os.path.join(base_dir, "mmo_statistics_table.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a" if target_funcs else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        if not file_exists or not target_funcs:
            writer.writeheader()
        writer.writerows(csv_rows)

    # 2. Append to official CEC 2020 MMO .txt matrices
    for ind in indicators:
        txt_path = os.path.join(results_dir, f"PESA2_{ind}.txt")
        header = ["Function"] + [f"Run{i+1}" for i in range(n_runs)] + ["Best", "Worst", "Mean", "Median", "Std"]
        
        with open(txt_path, "a" if target_funcs else "w") as f:
            if not os.path.isfile(txt_path) or not target_funcs:
                f.write("\t".join(header) + "\n")
            for name, vals in all_stats[ind].items():
                val_str = "\t".join(f"{v:.6e}" for v in vals)
                f.write(f"{name}\t{val_str}\n")

    print("\n" + "=" * 75)
    print(f"Success! Master MMO Statistics table updated at:\n  {csv_path}")
    print(f"Official CEC2020 .txt indicator tables & PS/PF CSVs updated in:\n  {results_dir}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--funcs", nargs="+", help="Specific MMO functions to run (e.g., MMF16_l1 MMF16_l2)")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs (use 21 for full CEC rule)")
    parser.add_argument("--full", action="store_true", help="Run with full 10000*N_ops evaluations")
    args = parser.parse_args()

    run_all_mmo2020(target_funcs=args.funcs, n_runs=args.runs, fast_mode=not args.full)