"""
experiments/report.py
Handles continuous metric tracking, appending run data cleanly to a central report.
"""
import os
import csv

def save_metrics(name, metrics, timestamp, seed, output_dir="results"):
    """Appends performance metrics to a master CSV tracker."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "master_metrics_report.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        fieldnames = ['Timestamp', 'Seed', 'Benchmark', 'IGD', 'GD', 'Spacing']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write headers if the file was just created
        if not file_exists:
            writer.writeheader()
            
        row = {
            'Timestamp': timestamp,
            'Seed': seed,
            'Benchmark': name,
            'IGD': f"{metrics['IGD']:.6f}",
            'GD': f"{metrics['GD']:.6f}",
            'Spacing': f"{metrics['Spacing']:.6f}"
        }
        writer.writerow(row)