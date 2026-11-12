"""
benchmarks/mmo2020_problems.py
Complete CEC 2020 Multimodal Multiobjective Optimization (MMO) Test Suite (24 Functions)
and Official Performance Indicators (IGDX, IGDF, rPSP, rHV).
"""
import os
import numpy as np
import scipy.io

def numpy_cdist(A, B):
    """Pure NumPy replacement for scipy.spatial.distance.cdist (Euclidean distance)"""
    # A is (M, D), B is (N, D)
    # Returns (M, N) distance matrix
    return np.sqrt(np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2))


class MMOProblem:
    """Base class compatible with algorithms/pesa2/pesa2.py"""
    def __init__(self, name, n_vars, n_objs, xl, xu, n_ops):
        self.name = name
        self.n_vars = n_vars
        self.n_var = n_vars
        self.dim = n_vars
        self.n_objs = n_objs
        self.n_obj = n_objs
        self.n_ops = n_ops
        self.xl = np.array(xl, dtype=float)
        self.xu = np.array(xu, dtype=float)
        self.lower_bounds = self.xl
        self.upper_bounds = self.xu
        self.bounds = [(float(l), float(u)) for l, u in zip(xl, xu)]

        # Tracks evaluated (X, F) pairs so we can recover Decision Space (PS) from PESA2
        self._history_x = []
        self._history_f = []

    def clear_history(self):
        self._history_x.clear()
        self._history_f.clear()

    def evaluate(self, x):
        x_arr = np.asarray(x, dtype=float)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        f_arr = self._eval_batch(x_arr)

        # Cache evaluations for Decision Space (PS) recovery
        self._history_x.append(x_arr.copy())
        self._history_f.append(f_arr.copy())

        return f_arr[0] if is_1d else f_arr

    def __call__(self, x):
        return self.evaluate(x)

    def recover_ps_from_pf(self, obtained_pf):
        """Matches the final Pareto Front (PF) back to its Decision Space (PS) coordinates."""
        obtained_pf = np.atleast_2d(np.asarray(obtained_pf, dtype=float))
        if not self._history_x or len(obtained_pf) == 0:
            return np.empty((0, self.n_vars))

        all_x = np.vstack(self._history_x)
        all_f = np.vstack(self._history_f)

        dists = numpy_cdist(obtained_pf, all_f)
        closest_idx = np.argmin(dists, axis=1)
        return all_x[closest_idx]

    def get_reference_data(self, base_dir=None):
        """Loads true PS and PF from .mat files, or generates them analytically if missing."""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        possible_paths = [
            os.path.join(base_dir, "MM_testfunctions", "Reference_PSPF_data", f"{self.name}_Reference_PSPF_data.mat"),
            os.path.join(base_dir, "benchmarks", "MM_testfunctions", "Reference_PSPF_data", f"{self.name}_Reference_PSPF_data.mat"),
            os.path.join(base_dir, "Reference_PSPF_data", f"{self.name}_Reference_PSPF_data.mat"),
        ]

        for mat_path in possible_paths:
            if os.path.exists(mat_path):
                try:
                    data = scipy.io.loadmat(mat_path)
                    ps = data.get("PS", data.get("truePS", None))
                    pf = data.get("PF", data.get("truePF", None))
                    if ps is not None and pf is not None:
                        return np.asarray(ps, dtype=float), np.asarray(pf, dtype=float)
                except Exception:
                    pass

        # Fallback: sample analytical pareto set/front so execution never crashes
        return self._analytical_ps_pf(n_points=500)

    def get_true_front(self, n_points=500):
        _, pf = self.get_reference_data()
        return pf

    def _analytical_ps_pf(self, n_points=500):
        rng = np.random.default_rng(42)
        samples = rng.uniform(self.xl, self.xu, size=(n_points * 20, self.n_vars))
        objs = self._eval_batch(samples)
        # Filter non-dominated
        is_nd = np.ones(objs.shape[0], dtype=bool)
        for i, f in enumerate(objs):
            if is_nd[i]:
                is_nd[is_nd] = ~np.all(objs[is_nd] <= f, axis=1) | np.all(objs[is_nd] == f, axis=1)
                is_nd[i] = True
        return samples[is_nd][:n_points], objs[is_nd][:n_points]

    def _eval_batch(self, x):
        raise NotImplementedError


# ==================== 24 CEC 2020 MMO TEST FUNCTIONS ====================

class MMF1(MMOProblem):
    def __init__(self):
        super().__init__("MMF1", 2, 2, [1.0, -1.0], [3.0, 1.0], n_ops=2)
    def _eval_batch(self, x):
        f1 = np.abs(x[:, 0] - 2.0)
        f2 = 1.0 - np.sqrt(f1) + 2.0 * (x[:, 1] - np.sin(6.0 * np.pi * f1 + np.pi)) ** 2
        return np.column_stack([f1, f2])

class MMF1_e(MMOProblem):
    def __init__(self):
        a = np.e
        super().__init__("MMF1_e", 2, 2, [1.0, -(a**3)], [3.0, a**3], n_ops=2)
    def _eval_batch(self, x):
        a = np.e
        f1 = np.abs(x[:, 0] - 2.0)
        mask = x[:, 0] < 2.0
        f2 = np.empty_like(f1)
        f2[mask] = 1.0 - np.sqrt(f1[mask]) + 2.0 * (x[mask, 1] - np.sin(6.0 * np.pi * f1[mask] + np.pi)) ** 2
        f2[~mask] = 1.0 - np.sqrt(f1[~mask]) + 2.0 * (x[~mask, 1] - (a ** x[~mask, 0]) * np.sin(6.0 * np.pi * f1[~mask] + np.pi)) ** 2
        return np.column_stack([f1, f2])

class MMF2(MMOProblem):
    def __init__(self):
        super().__init__("MMF2", 2, 2, [0.0, 0.0], [1.0, 2.0], n_ops=2)
    def _eval_batch(self, x):
        f1 = x[:, 0]
        y2 = np.where(x[:, 1] <= 1.0, x[:, 1] - np.sqrt(f1), x[:, 1] - 1.0 - np.sqrt(f1))
        f2 = 1.0 - np.sqrt(f1) + 2.0 * (4.0 * (y2 ** 2) - 2.0 * np.cos(20.0 * y2 * np.pi / np.sqrt(2.0)) + 2.0)
        return np.column_stack([f1, f2])

class MMF4(MMOProblem):
    def __init__(self):
        super().__init__("MMF4", 2, 2, [-1.0, 0.0], [1.0, 2.0], n_ops=2)
    def _eval_batch(self, x):
        f1 = np.abs(x[:, 0])
        y2 = np.where(x[:, 1] < 1.0, x[:, 1] - np.sin(np.pi * f1), x[:, 1] - 1.0 - np.sin(np.pi * f1))
        f2 = 1.0 - (x[:, 0] ** 2) + 2.0 * (y2 ** 2)
        return np.column_stack([f1, f2])

class MMF5(MMOProblem):
    def __init__(self):
        super().__init__("MMF5", 2, 2, [1.0, -1.0], [3.0, 3.0], n_ops=2)
    def _eval_batch(self, x):
        f1 = np.abs(x[:, 0] - 2.0)
        s = np.sin(6.0 * np.pi * f1 + np.pi)
        y2 = np.where(x[:, 1] <= 1.0, x[:, 1] - s, x[:, 1] - 2.0 - s)
        f2 = 1.0 - np.sqrt(f1) + 2.0 * (y2 ** 2)
        return np.column_stack([f1, f2])

class MMF7(MMOProblem):
    def __init__(self):
        super().__init__("MMF7", 2, 2, [1.0, -1.0], [3.0, 1.0], n_ops=2)
    def _eval_batch(self, x):
        f1 = np.abs(x[:, 0] - 2.0)
        term = (0.3 * (f1 ** 2) * np.cos(24.0 * np.pi * f1 + 4.0 * np.pi) + 0.6 * f1) * np.sin(6.0 * np.pi * f1 + np.pi)
        f2 = 1.0 - np.sqrt(f1) + (x[:, 1] - term) ** 2
        return np.column_stack([f1, f2])

class MMF8(MMOProblem):
    def __init__(self):
        super().__init__("MMF8", 2, 2, [-np.pi, 0.0], [np.pi, 9.0], n_ops=2)
    def _eval_batch(self, x):
        abs_x1 = np.abs(x[:, 0])
        f1 = np.sin(abs_x1)
        y2 = np.where(x[:, 1] <= 4.0, x[:, 1] - f1 - abs_x1, x[:, 1] - 4.0 - f1 - abs_x1)
        f2 = np.sqrt(np.maximum(0.0, 1.0 - f1 ** 2)) + 2.0 * (y2 ** 2)
        return np.column_stack([f1, f2])

class _MMF10Base(MMOProblem):
    def _eval_batch(self, x):
        f1 = x[:, 0]
        g = 2.0 - np.exp(-((x[:, 1] - 0.2) / 0.004) ** 2) - 0.8 * np.exp(-((x[:, 1] - 0.6) / 0.4) ** 2)
        return np.column_stack([f1, g / f1])

class MMF10(_MMF10Base):
    def __init__(self): super().__init__("MMF10", 2, 2, [0.1, 0.1], [1.1, 1.1], n_ops=1)

class MMF10_l(_MMF10Base):
    def __init__(self): super().__init__("MMF10_l", 2, 2, [0.1, 0.1], [1.1, 1.1], n_ops=2)

class _MMF11Base(MMOProblem):
    def _eval_batch(self, x):
        f1 = x[:, 0]
        g = 2.0 - np.exp(-2.0 * np.log(2.0) * ((x[:, 1] - 0.1) / 0.8) ** 2) * (np.sin(2.0 * np.pi * x[:, 1]) ** 6)
        return np.column_stack([f1, g / f1])

class MMF11(_MMF11Base):
    def __init__(self): super().__init__("MMF11", 2, 2, [0.1, 0.1], [1.1, 1.1], n_ops=1)

class MMF11_l(_MMF11Base):
    def __init__(self): super().__init__("MMF11_l", 2, 2, [0.1, 0.1], [1.1, 1.1], n_ops=2)

class _MMF12Base(MMOProblem):
    def _eval_batch(self, x):
        f1 = x[:, 0]
        g = 2.0 - np.exp(-2.0 * np.log(2.0) * ((x[:, 1] - 0.1) / 0.8) ** 2) * (np.sin(2.0 * np.pi * x[:, 1]) ** 6)
        h = 1.0 - (f1 / g) ** 2 - (f1 / g) * np.sin(2.0 * np.pi * 4.0 * f1)
        return np.column_stack([f1, g * h])

class MMF12(_MMF12Base):
    def __init__(self): super().__init__("MMF12", 2, 2, [0.0, 0.0], [1.0, 1.0], n_ops=1)

class MMF12_l(_MMF12Base):
    def __init__(self): super().__init__("MMF12_l", 2, 2, [0.0, 0.0], [1.0, 1.0], n_ops=2)

class _MMF13Base(MMOProblem):
    def _eval_batch(self, x):
        f1 = x[:, 0]
        t = x[:, 1] + np.sqrt(x[:, 2])
        g = 2.0 - np.exp(-2.0 * np.log(2.0) * ((t - 0.1) / 0.8) ** 2) * (np.sin(2.0 * np.pi * t) ** 6)
        return np.column_stack([f1, g / f1])

class MMF13(_MMF13Base):
    def __init__(self): super().__init__("MMF13", 3, 2, [0.1, 0.1, 0.1], [1.1, 1.1, 1.1], n_ops=1)

class MMF13_l(_MMF13Base):
    def __init__(self): super().__init__("MMF13_l", 3, 2, [0.1, 0.1, 0.1], [1.1, 1.1, 1.1], n_ops=2)

def _sphere_3d(x, g):
    f1 = np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1]) * (1.0 + g)
    f2 = np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1]) * (1.0 + g)
    f3 = np.sin(0.5 * np.pi * x[:, 0]) * (1.0 + g)
    return np.column_stack([f1, f2, f3])

class MMF14(MMOProblem):
    def __init__(self): super().__init__("MMF14", 3, 3, [0.0]*3, [1.0]*3, n_ops=2)
    def _eval_batch(self, x):
        g = 2.0 - (np.sin(2.0 * np.pi * x[:, 2]) ** 2)
        return _sphere_3d(x, g)

class MMF14_a(MMOProblem):
    def __init__(self): super().__init__("MMF14_a", 3, 3, [0.0]*3, [1.0]*3, n_ops=2)
    def _eval_batch(self, x):
        t = x[:, 2] - 0.5 * np.sin(np.pi * x[:, 1]) + 0.25
        g = 2.0 - (np.sin(2.0 * np.pi * t) ** 2)
        return _sphere_3d(x, g)

class _MMF15Base(MMOProblem):
    def _eval_batch(self, x):
        g = 2.0 - np.exp(-2.0 * np.log(2.0) * ((x[:, 2] - 0.1) / 0.8) ** 2) * (np.sin(2.0 * np.pi * x[:, 2]) ** 2)
        return _sphere_3d(x, g)

class MMF15(_MMF15Base):
    def __init__(self): super().__init__("MMF15", 3, 3, [0.0]*3, [1.0]*3, n_ops=1)

class MMF15_l(_MMF15Base):
    def __init__(self): super().__init__("MMF15_l", 3, 3, [0.0]*3, [1.0]*3, n_ops=2)

class _MMF15aBase(MMOProblem):
    def _eval_batch(self, x):
        t = x[:, 2] - 0.5 * np.sin(np.pi * x[:, 1]) + 0.25
        g = 2.0 - np.exp(-2.0 * np.log(2.0) * ((t - 0.1) / 0.8) ** 2) * (np.sin(2.0 * np.pi * t) ** 2)
        return _sphere_3d(x, g)

class MMF15_a(_MMF15aBase):
    def __init__(self): super().__init__("MMF15_a", 3, 3, [0.0]*3, [1.0]*3, n_ops=1)

class MMF15_a_l(_MMF15aBase):
    def __init__(self): super().__init__("MMF15_a_l", 3, 3, [0.0]*3, [1.0]*3, n_ops=2)

class _MMF16Base(MMOProblem):
    def __init__(self, name, n_pg, n_pl):
        super().__init__(name, 3, 3, [0.0]*3, [1.0]*3, n_ops=n_pg + n_pl)
        self.n_pg = n_pg
        self.n_pl = n_pl
    def _eval_batch(self, x):
        mask = x[:, 2] < 0.5
        g = np.empty(x.shape[0], dtype=float)
        g[mask] = 2.0 - (np.sin(2.0 * self.n_pg * np.pi * x[mask, 2]) ** 2)
        g[~mask] = 2.0 - np.exp(-2.0 * np.log(2.0) * ((x[~mask, 2] - 0.1) / 0.8) ** 2) * (np.sin(2.0 * self.n_pl * np.pi * x[~mask, 2]) ** 2)
        return _sphere_3d(x, g)

class MMF16_l1(_MMF16Base):
    def __init__(self): super().__init__("MMF16_l1", n_pg=2, n_pl=1)

class MMF16_l2(_MMF16Base):
    def __init__(self): super().__init__("MMF16_l2", n_pg=1, n_pl=2)

class MMF16_l3(_MMF16Base):
    def __init__(self): super().__init__("MMF16_l3", n_pg=2, n_pl=2)


# Official order from Table I in the CEC 2020 MMO specification
MMO_BENCHMARK_REGISTRY = {
    "MMF1": MMF1, "MMF2": MMF2, "MMF4": MMF4, "MMF5": MMF5,
    "MMF7": MMF7, "MMF8": MMF8, "MMF10": MMF10, "MMF11": MMF11,
    "MMF12": MMF12, "MMF13": MMF13, "MMF14": MMF14, "MMF15": MMF15,
    "MMF1_e": MMF1_e, "MMF14_a": MMF14_a, "MMF15_a": MMF15_a,
    "MMF10_l": MMF10_l, "MMF11_l": MMF11_l, "MMF12_l": MMF12_l,
    "MMF13_l": MMF13_l, "MMF15_l": MMF15_l, "MMF15_a_l": MMF15_a_l,
    "MMF16_l1": MMF16_l1, "MMF16_l2": MMF16_l2, "MMF16_l3": MMF16_l3,
}


# ==================== OFFICIAL CEC 2020 MMO METRICS ====================

def calc_igd_metric(obtained, reference):
    """Inverted Generational Distance (used for both IGDX and IGDF)."""
    obtained = np.atleast_2d(np.asarray(obtained, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    if len(obtained) == 0 or len(reference) == 0:
        return np.inf
    min_dists = np.min(numpy_cdist(reference, obtained), axis=1)
    return float(np.mean(min_dists))

def calc_cr_and_rpsp(obtained_ps, true_ps, igdx):
    """Calculates Cover Rate (CR) and reciprocal Pareto Set Proximity (1/PSP = rPSP)."""
    obtained_ps = np.atleast_2d(np.asarray(obtained_ps, dtype=float))
    true_ps = np.atleast_2d(np.asarray(true_ps, dtype=float))
    if len(obtained_ps) == 0 or len(true_ps) == 0 or igdx == 0:
        return 1e4

    n_var = true_ps.shape[1]
    v_max = np.max(true_ps, axis=0)
    v_min = np.min(true_ps, axis=0)
    l_max = np.max(obtained_ps, axis=0)
    l_min = np.min(obtained_ps, axis=0)

    cr_prod = 1.0
    for d in range(n_var):
        denom = v_max[d] - v_min[d]
        if denom <= 1e-12:
            continue
        if l_min[d] >= v_max[d] or l_max[d] <= v_min[d]:
            return 1e4
        inter = min(l_max[d], v_max[d]) - max(l_min[d], v_min[d])
        cr_prod *= (inter / denom) ** 2

    cr = cr_prod ** (1.0 / (2.0 * n_var))
    if cr <= 1e-10:
        return 1e4
    psp = cr / igdx
    return float(1.0 / psp)

def calc_rhv(obtained_pf, true_pf):
    """Calculates reciprocal Hypervolume (1/HV = rHV)."""
    obtained_pf = np.atleast_2d(np.asarray(obtained_pf, dtype=float))
    true_pf = np.atleast_2d(np.asarray(true_pf, dtype=float))
    if len(obtained_pf) == 0:
        return 1e4

    ref_point = 1.1 * np.max(true_pf, axis=0)
    valid = np.all(obtained_pf <= ref_point, axis=1)
    pts = obtained_pf[valid]
    if len(pts) == 0:
        return 1e4

    if pts.shape[1] == 2:
        # Exact 2D Hypervolume
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        prev_f2 = ref_point[1]
        for p in pts:
            if p[1] < prev_f2:
                hv += (ref_point[0] - p[0]) * (prev_f2 - p[1])
                prev_f2 = p[1]
    else:
        # Fast Monte Carlo 3D Hypervolume approximation
        mins = np.min(pts, axis=0)
        vol_box = float(np.prod(ref_point - mins))
        if vol_box <= 0:
            return 1e4
        rng = np.random.default_rng(42)
        samples = rng.uniform(mins, ref_point, size=(10000, pts.shape[1]))
        dominated = np.zeros(len(samples), dtype=bool)
        for p in pts:
            dominated |= np.all(samples >= p, axis=1)
        hv = vol_box * (np.sum(dominated) / len(samples))

    return float(1.0 / max(hv, 1e-10))