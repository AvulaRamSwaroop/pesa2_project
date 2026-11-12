"""
benchmarks/problems.py
Fully verified benchmark suite for PESA-II Multi-Objective Optimization.
Implements all 25 benchmark problems featured in the research comparison.
Includes performance optimizations for 3D/surface True Pareto Front generation.
"""

import numpy as np

class BenchmarkProblem:
    def __init__(self, name, n_vars, n_objs, bounds):
        self.name = name
        self.n_vars = n_vars
        self.n_objs = n_objs
        self.bounds = np.array(bounds, dtype=float)

    def evaluate(self, x):
        raise NotImplementedError

    def get_true_front(self, n_points=500):
        raise NotImplementedError

def filter_non_dominated(pts):
    pts = np.asarray(pts)
    n = len(pts)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        dominates_i = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        if np.any(dominates_i):
            is_dominated[i] = True
    front = pts[~is_dominated]
    if front.shape[1] == 2:
        return front[np.argsort(front[:, 0])]
    return front

# --- ZDT SUITE ---
class ZDT1(BenchmarkProblem):
    def __init__(self, n_vars=30):
        super().__init__("ZDT1", n_vars, 2, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_vars - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - np.sqrt(f1)))

class ZDT2(BenchmarkProblem):
    def __init__(self, n_vars=30):
        super().__init__("ZDT2", n_vars, 2, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_vars - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - f1 ** 2))

class ZDT3(BenchmarkProblem):
    def __init__(self, n_vars=30):
        super().__init__("ZDT3", n_vars, 2, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_vars - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        return np.array([f1, f2])
    def get_true_front(self, n_points=1000):
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        return filter_non_dominated(np.column_stack((f1, f2)))

class ZDT4(BenchmarkProblem):
    def __init__(self, n_vars=10):
        bounds = [[0.0, 1.0]] + [[-5.0, 5.0]] * (n_vars - 1)
        super().__init__("ZDT4", n_vars, 2, bounds)
    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_vars - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - np.sqrt(f1)))

class ZDT6(BenchmarkProblem):
    def __init__(self, n_vars=10):
        super().__init__("ZDT6", n_vars, 2, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0]) ** 6)
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_vars - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        f1 = np.linspace(0.280775, 1.0, n_points)
        return np.column_stack((f1, 1.0 - f1 ** 2))

# --- BINH SUITE ---
class Binh1(BenchmarkProblem):
    def __init__(self):
        super().__init__("Binh1", 2, 2, [[-5.0, 10.0], [-5.0, 10.0]])
    def evaluate(self, x):
        return np.array([x[0]**2 + x[1]**2, (x[0] - 5.0)**2 + (x[1] - 5.0)**2])
    def get_true_front(self, n_points=500):
        t = np.linspace(0.0, 5.0, n_points)
        return np.column_stack((2.0 * (t ** 2), 2.0 * ((t - 5.0) ** 2)))

class Binh3(BenchmarkProblem):
    def __init__(self):
        super().__init__("Binh3", 2, 3, [[-1.05, 1.05], [-1.05, 1.05]])
    def evaluate(self, x):
        return np.array([x[0] - 1e-6, x[1] - 1e-6, (x[0] - 1.0)**2 + (x[1] - 1.0)**2])
    def get_true_front(self, n_points=500):
        grid_size = 80  # Capped to 80x80 (6,400 points) to prevent O(N^2) CPU explosion
        x1, x2 = np.meshgrid(np.linspace(-1.05, 1.05, grid_size), np.linspace(-1.05, 1.05, grid_size))
        pts = np.column_stack((x1.flatten() - 1e-6, x2.flatten() - 1e-6, (x1.flatten() - 1.0)**2 + (x2.flatten() - 1.0)**2))
        return filter_non_dominated(pts)

# --- DEB SUITE ---
class Deb(BenchmarkProblem):
    def __init__(self):
        super().__init__("Deb", 2, 2, [[0.0, 1.0], [0.0, 1.0]])
    def evaluate(self, x):
        g = 1.0 + 9.0 * x[1]
        return np.array([x[0], g * (1.0 - (x[0] / g) ** 2)])
    def get_true_front(self, n_points=500):
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((f1, 1.0 - f1 ** 2))

class Deb2(BenchmarkProblem):
    def __init__(self):
        super().__init__("Deb2", 2, 2, [[0.0, 1.0], [0.0, 1.0]])
    def evaluate(self, x):
        g = 1.0 + 9.0 * x[1]
        f2 = g * (1.0 - (x[0] / g) ** 0.25 - (x[0] / g) * np.sin(10.0 * np.pi * x[0]))
        return np.array([x[0], f2])
    def get_true_front(self, n_points=1000):
        f1 = np.linspace(0.0, 1.0, n_points)
        return filter_non_dominated(np.column_stack((f1, 1.0 - f1 ** 0.25 - f1 * np.sin(10.0 * np.pi * f1))))

# --- FONSECA & KURSAWE ---
class Fonseca(BenchmarkProblem):
    def __init__(self, n_vars=3):
        super().__init__("Fonseca", n_vars, 2, [[-4.0, 4.0]] * n_vars)
    def evaluate(self, x):
        inv = 1.0 / np.sqrt(self.n_vars)
        f1 = 1.0 - np.exp(-np.sum((x - inv) ** 2))
        f2 = 1.0 - np.exp(-np.sum((x + inv) ** 2))
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        inv = 1.0 / np.sqrt(self.n_vars)
        t = np.linspace(-inv, inv, n_points)
        return np.column_stack((1.0 - np.exp(-self.n_vars * (t - inv) ** 2), 1.0 - np.exp(-self.n_vars * (t + inv) ** 2)))

class Fonseca2(BenchmarkProblem):
    def __init__(self):
        super().__init__("Fonseca2", 2, 2, [[-4.0, 4.0], [-4.0, 4.0]])
    def evaluate(self, x):
        f1 = 1.0 - np.exp(-((x[0] - 1.0) ** 2 + (x[1] + 1.0) ** 2))
        f2 = 1.0 - np.exp(-((x[0] + 1.0) ** 2 + (x[1] - 1.0) ** 2))
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        t = np.linspace(-1.0, 1.0, n_points)
        return np.column_stack((1.0 - np.exp(-2.0 * (t - 1.0) ** 2), 1.0 - np.exp(-2.0 * (t + 1.0) ** 2)))

class Kursawe(BenchmarkProblem):
    def __init__(self, n_vars=3):
        super().__init__("Kursawe", n_vars, 2, [[-5.0, 5.0]] * n_vars)
    def evaluate(self, x):
        f1 = np.sum([-10.0 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2)) for i in range(len(x) - 1)])
        f2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x ** 3))
        return np.array([f1, f2])
    def get_true_front(self, n_points=1000):
        samples = np.random.uniform(-5.0, 5.0, (25000, self.n_vars))
        return filter_non_dominated(np.array([self.evaluate(s) for s in samples]))

# --- LAUMANNS, MURATA, QUAGLIARELLA ---
class Laumanns(BenchmarkProblem):
    def __init__(self):
        super().__init__("Laumanns", 2, 2, [[-5.0, 5.0], [-5.0, 5.0]])
    def evaluate(self, x):
        return np.array([x[0]**2 + x[1]**2, (x[0] + 2.0)**2 + x[1]**2])
    def get_true_front(self, n_points=500):
        t = np.linspace(-2.0, 0.0, n_points)
        return np.column_stack((t ** 2, (t + 2.0) ** 2))

class Murata(BenchmarkProblem):
    def __init__(self):
        super().__init__("Murata", 2, 2, [[1.0, 4.0], [1.0, 4.0]])
    def evaluate(self, x):
        return np.array([2.0 * np.sqrt(x[0]), x[0] * (1.0 - x[1]) + 5.0])
    def get_true_front(self, n_points=500):
        t = np.linspace(1.0, 4.0, n_points)
        return np.column_stack((2.0 * np.sqrt(t), -3.0 * t + 5.0))

class Quagliarella(BenchmarkProblem):
    def __init__(self):
        super().__init__("Quagliarella", 2, 2, [[-5.12, 5.12], [-5.12, 5.12]])
    def evaluate(self, x):
        f1 = (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2
        f2 = (x[0] + 2.0)**2 + (x[1] - 1.0)**2
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        grid_size = 80  # Capped to prevent O(N^2) CPU explosion
        X1, X2 = np.meshgrid(np.linspace(-1.0, 4.0, grid_size), np.linspace(-1.0, 4.0, grid_size))
        return filter_non_dominated(np.array([self.evaluate([a, b]) for a, b in zip(X1.flatten(), X2.flatten())]))

# --- RENDON & SCHAFFER ---
class Rendon(BenchmarkProblem):
    def __init__(self):
        super().__init__("Rendon", 2, 2, [[-3.0, 3.0], [-3.0, 3.0]])
    def evaluate(self, x):
        return np.array([1.0 / (x[0]**2 + x[1]**2 + 1.0), x[0]**2 + 3.0 * (x[1]**2) + 1.0])
    def get_true_front(self, n_points=500):
        t = np.linspace(0.0, 3.0, n_points)
        return np.column_stack((1.0 / (t**2 + 1.0), t**2 + 1.0))

class Rendon2(BenchmarkProblem):
    def __init__(self):
        super().__init__("Rendon2", 2, 2, [[-5.0, 5.0], [-5.0, 5.0]])
    def evaluate(self, x):
        return np.array([x[0] + x[1], x[0]**2 - 2.0 * x[0] + x[1]**2 - 2.0 * x[1]])
    def get_true_front(self, n_points=500):
        t = np.linspace(-5.0, -2.0, n_points)
        return np.column_stack((t, 0.5 * (t ** 2) - 2.0 * t))

class Schaffer(BenchmarkProblem):
    def __init__(self):
        super().__init__("Schaffer", 1, 2, [[-10.0, 10.0]])
    def evaluate(self, x):
        return np.array([x[0]**2, (x[0] - 2.0)**2])
    def get_true_front(self, n_points=500):
        t = np.linspace(0.0, 2.0, n_points)
        return np.column_stack((t ** 2, (t - 2.0) ** 2))

class Schaffer2(BenchmarkProblem):
    def __init__(self):
        super().__init__("Schaffer2", 1, 2, [[-5.0, 10.0]])
    def evaluate(self, x):
        x1 = x[0]
        f1 = -x1 if x1 <= 1.0 else (-2.0 + x1 if x1 <= 3.0 else (4.0 - x1 if x1 <= 4.0 else -4.0 + x1))
        return np.array([f1, (x1 - 5.0)**2])
    def get_true_front(self, n_points=500):
        x = np.concatenate([np.linspace(1.0, 2.0, n_points // 2), np.linspace(4.0, 5.0, n_points // 2)])
        return filter_non_dominated(np.array([self.evaluate([xi]) for xi in x]))

# --- SRINIVAS, TANAKA, VIENNET ---
class Srinivas(BenchmarkProblem):
    def __init__(self):
        super().__init__("Srinivas", 2, 2, [[-20.0, 20.0], [-20.0, 20.0]])
    def evaluate(self, x):
        f1 = 2.0 + (x[0] - 2.0)**2 + (x[1] - 1.0)**2
        f2 = 9.0 * x[0] - (x[1] - 1.0)**2
        c1 = x[0]**2 + x[1]**2 - 225.0
        c2 = x[0] - 3.0 * x[1] + 10.0
        if c1 > 0 or c2 > 0:
            penalty = max(0.0, c1) + max(0.0, c2)
            return np.array([1e5 + penalty, 1e5 + penalty])
        return np.array([f1, f2])
    def get_true_front(self, n_points=500):
        theta = np.linspace(0.0, np.pi / 2.0, n_points)
        x1, x2 = 15.0 * np.cos(theta), 15.0 * np.sin(theta)
        return np.column_stack((2.0 + (x1 - 2.0)**2 + (x2 - 1.0)**2, 9.0 * x1 - (x2 - 1.0)**2))

class Tanaka(BenchmarkProblem):
    def __init__(self):
        super().__init__("Tanaka", 2, 2, [[1e-5, np.pi], [1e-5, np.pi]])
    def evaluate(self, x):
        c1 = -(x[0]**2 + x[1]**2 - 1.0 - 0.1 * np.cos(16.0 * np.arctan2(x[0], x[1])))
        c2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.5
        if c1 > 0 or c2 > 0:
            penalty = max(0.0, c1) + max(0.0, c2)
            return np.array([1e5 + penalty, 1e5 + penalty])
        return np.array([x[0], x[1]])
    def get_true_front(self, n_points=1000):
        t = np.linspace(1e-5, np.pi / 2.0, n_points)
        r = np.sqrt(1.0 + 0.1 * np.cos(16.0 * t))
        x1, x2 = r * np.sin(t), r * np.cos(t)
        valid = ((x1 - 0.5)**2 + (x2 - 0.5)**2) <= 0.5
        return np.column_stack((x1[valid], x2[valid]))

class Viennet(BenchmarkProblem):
    def __init__(self):
        super().__init__("Viennet", 2, 3, [[-3.0, 3.0], [-3.0, 3.0]])
    def evaluate(self, x):
        r2 = x[0]**2 + x[1]**2
        f1 = 0.5 * r2 + np.sin(r2)
        f2 = ((3.0 * x[0] - 2.0 * x[1] + 4.0)**2) / 8.0 + ((x[0] - x[1] + 1.0)**2) / 27.0 + 15.0
        f3 = 1.0 / (r2 + 1.0) - 1.1 * np.exp(-r2)
        return np.array([f1, f2, f3])
    def get_true_front(self, n_points=500):
        grid_size = 80  # Capped to prevent O(N^2) CPU explosion
        X1, X2 = np.meshgrid(np.linspace(-3.0, 3.0, grid_size), np.linspace(-3.0, 3.0, grid_size))
        return filter_non_dominated(np.array([self.evaluate([a, b]) for a, b in zip(X1.flatten(), X2.flatten())]))

class Viennet2(BenchmarkProblem):
    def __init__(self):
        super().__init__("Viennet2", 2, 3, [[-4.0, 4.0], [-4.0, 4.0]])
    def evaluate(self, x):
        f1 = ((x[0] - 2.0)**2) / 2.0 + ((x[1] + 1.0)**2) / 13.0 + 3.0
        f2 = ((x[0] + x[1] - 3.0)**2) / 36.0 + ((-x[0] + x[1] + 2.0)**2) / 8.0 - 17.0
        f3 = ((x[0] + 2.0 * x[1] - 1.0)**2) / 175.0 + ((2.0 * x[1] - x[0])**2) / 17.0 - 13.0
        return np.array([f1, f2, f3])
    def get_true_front(self, n_points=500):
        grid_size = 80  # Capped to prevent O(N^2) CPU explosion
        X1, X2 = np.meshgrid(np.linspace(-4.0, 4.0, grid_size), np.linspace(-4.0, 4.0, grid_size))
        return filter_non_dominated(np.array([self.evaluate([a, b]) for a, b in zip(X1.flatten(), X2.flatten())]))

# --- DTLZ SUITE ---
class DTLZ1(BenchmarkProblem):
    def __init__(self, n_vars=7, n_objs=3):
        super().__init__("DTLZ1", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        k = self.n_vars - self.n_objs + 1
        xm = x[-k:]
        g = 100.0 * (k + np.sum((xm - 0.5)**2 - np.cos(20.0 * np.pi * (xm - 0.5))))
        f = []
        for i in range(self.n_objs):
            val = 0.5 * (1.0 + g)
            for j in range(self.n_objs - 1 - i): val *= x[j]
            if i > 0: val *= (1.0 - x[self.n_objs - 1 - i])
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        return 0.5 * np.random.dirichlet(np.ones(self.n_objs), n_points)

class DTLZ2(BenchmarkProblem):
    def __init__(self, n_vars=12, n_objs=3):
        super().__init__("DTLZ2", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        k = self.n_vars - self.n_objs + 1
        g = np.sum((x[-k:] - 0.5)**2)
        f = []
        for i in range(self.n_objs):
            val = 1.0 + g
            for j in range(self.n_objs - 1 - i): val *= np.cos(0.5 * np.pi * x[j])
            if i > 0: val *= np.sin(0.5 * np.pi * x[self.n_objs - 1 - i])
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        pts = np.abs(np.random.normal(0, 1, (n_points, self.n_objs)))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

class DTLZ3(BenchmarkProblem):
    def __init__(self, n_vars=12, n_objs=3):
        super().__init__("DTLZ3", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        k = self.n_vars - self.n_objs + 1
        xm = x[-k:]
        g = 100.0 * (k + np.sum((xm - 0.5)**2 - np.cos(20.0 * np.pi * (xm - 0.5))))
        f = []
        for i in range(self.n_objs):
            val = 1.0 + g
            for j in range(self.n_objs - 1 - i): val *= np.cos(0.5 * np.pi * x[j])
            if i > 0: val *= np.sin(0.5 * np.pi * x[self.n_objs - 1 - i])
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        pts = np.abs(np.random.normal(0, 1, (n_points, self.n_objs)))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

class DTLZ4(BenchmarkProblem):
    def __init__(self, n_vars=12, n_objs=3, alpha=100.0):
        super().__init__("DTLZ4", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
        self.alpha = alpha
    def evaluate(self, x):
        k = self.n_vars - self.n_objs + 1
        g = np.sum((x[-k:] - 0.5)**2)
        f = []
        for i in range(self.n_objs):
            val = 1.0 + g
            for j in range(self.n_objs - 1 - i): val *= np.cos(0.5 * np.pi * (x[j]**self.alpha))
            if i > 0: val *= np.sin(0.5 * np.pi * (x[self.n_objs - 1 - i]**self.alpha))
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        pts = np.abs(np.random.normal(0, 1, (n_points, self.n_objs)))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

class DTLZ5(BenchmarkProblem):
    def __init__(self, n_vars=12, n_objs=3):
        super().__init__("DTLZ5", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        g = np.sum((x[-(self.n_vars - self.n_objs + 1):] - 0.5)**2)
        theta = [0.5 * np.pi * x[0]] + [(np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[i]) for i in range(1, self.n_objs - 1)]
        f = []
        for i in range(self.n_objs):
            val = 1.0 + g
            for j in range(self.n_objs - 1 - i): val *= np.cos(theta[j])
            if i > 0: val *= np.sin(theta[self.n_objs - 1 - i])
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        t = np.linspace(0.0, 0.5 * np.pi, n_points)
        return np.column_stack(((1.0 / np.sqrt(2.0)) * np.cos(t), (1.0 / np.sqrt(2.0)) * np.cos(t), np.sin(t)))

class DTLZ6(BenchmarkProblem):
    def __init__(self, n_vars=12, n_objs=3):
        super().__init__("DTLZ6", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        g = np.sum(x[-(self.n_vars - self.n_objs + 1):] ** 0.1)
        theta = [0.5 * np.pi * x[0]] + [(np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[i]) for i in range(1, self.n_objs - 1)]
        f = []
        for i in range(self.n_objs):
            val = 1.0 + g
            for j in range(self.n_objs - 1 - i): val *= np.cos(theta[j])
            if i > 0: val *= np.sin(theta[self.n_objs - 1 - i])
            f.append(val)
        return np.array(f)
    def get_true_front(self, n_points=500):
        t = np.linspace(0.0, 0.5 * np.pi, n_points)
        return np.column_stack(((1.0 / np.sqrt(2.0)) * np.cos(t), (1.0 / np.sqrt(2.0)) * np.cos(t), np.sin(t)))

class DTLZ7(BenchmarkProblem):
    def __init__(self, n_vars=22, n_objs=3):
        super().__init__("DTLZ7", n_vars, n_objs, [[0.0, 1.0]] * n_vars)
    def evaluate(self, x):
        k = self.n_vars - self.n_objs + 1
        g = 1.0 + (9.0 / k) * np.sum(x[-k:])
        f = list(x[:self.n_objs - 1])
        h = self.n_objs - np.sum([f[i] / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * f[i])) for i in range(self.n_objs - 1)])
        f.append((1.0 + g) * h)
        return np.array(f)
    def get_true_front(self, n_points=500):
        F1, F2 = np.meshgrid(np.linspace(0.0, 1.0, int(np.sqrt(n_points))), np.linspace(0.0, 1.0, int(np.sqrt(n_points))))
        f1, f2 = F1.flatten(), F2.flatten()
        h = 3.0 - (f1 * (1.0 + np.sin(3.0 * np.pi * f1)) + f2 * (1.0 + np.sin(3.0 * np.pi * f2)))
        return filter_non_dominated(np.column_stack((f1, f2, h)))

# GLOBAL REGISTRY
BENCHMARK_REGISTRY = {
    "ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3, "ZDT4": ZDT4, "ZDT6": ZDT6,
    "Binh(1)": Binh1, "Binh(3)": Binh3, "Deb": Deb, "Deb(2)": Deb2,
    "Fonseca": Fonseca, "Fonseca(2)": Fonseca2, "Kursawe": Kursawe,
    "Laumanns": Laumanns, "Murata": Murata, "Quagliarella": Quagliarella,
    "Rendon": Rendon, "Rendon(2)": Rendon2, "Schaffer": Schaffer, "Schaffer(2)": Schaffer2,
    "Srinivas": Srinivas, "Tanaka": Tanaka, "Viennet": Viennet, "Viennet(2)": Viennet2,
    "DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "DTLZ3": DTLZ3, "DTLZ4": DTLZ4, "DTLZ5": DTLZ5, "DTLZ6": DTLZ6, "DTLZ7": DTLZ7
}