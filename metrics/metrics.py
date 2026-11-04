"""
metrics/metrics.py
Calculates MOEA performance metrics: IGD, GD, and Spacing.
"""
import numpy as np
from scipy.spatial.distance import cdist

def calc_igd(obtained, true_pf):
    """Inverted Generational Distance: Average distance from True PF to nearest Obtained."""
    if len(obtained) == 0: return float('inf')
    distances = cdist(true_pf, obtained)
    return np.mean(np.min(distances, axis=1))

def calc_gd(obtained, true_pf):
    """Generational Distance: Average distance from Obtained to nearest True PF."""
    if len(obtained) == 0: return float('inf')
    distances = cdist(obtained, true_pf)
    return np.mean(np.min(distances, axis=1))

def calc_spacing(obtained):
    """Spacing: Measures the uniformity of the obtained non-dominated front."""
    if len(obtained) < 2: return float('inf')
    # Use L1 distance (Manhattan) as is standard for Spacing metric
    distances = cdist(obtained, obtained, metric='cityblock')
    np.fill_diagonal(distances, np.inf)
    d_i = np.min(distances, axis=1)
    d_mean = np.mean(d_i)
    return np.sqrt(np.sum((d_i - d_mean)**2) / (len(obtained) - 1))