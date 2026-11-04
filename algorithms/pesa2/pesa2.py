"""
algorithms/pesa2/pesa2.py
Exact implementation of PESA-II with reproducible RandomState support.
"""
import numpy as np

class Solution:
    def __init__(self, chromosome, objectives=None):
        self.chromosome = np.array(chromosome, dtype=np.uint8)
        self.objectives = objectives

class HyperboxGrid:
    def __init__(self, n_objs, grid_divisions, rng):
        self.n_objs = n_objs
        self.grid_divisions = grid_divisions
        self.rng = rng
        self.lower_bounds = None
        self.upper_bounds = None
        self.grid_boxes = {}

    def update_grid(self, archive):
        self.grid_boxes.clear()
        if not archive: return
        objs = np.array([sol.objectives for sol in archive])
        min_vals = np.min(objs, axis=0)
        max_vals = np.max(objs, axis=0)
        span = max_vals - min_vals
        span[span == 0] = 1e-6
        self.lower_bounds = min_vals - 0.01 * span
        self.upper_bounds = max_vals + 0.01 * span
        delta = (self.upper_bounds - self.lower_bounds) / self.grid_divisions

        for sol in archive:
            coords = np.floor((sol.objectives - self.lower_bounds) / delta).astype(int)
            box_key = tuple(np.clip(coords, 0, self.grid_divisions - 1))
            if box_key not in self.grid_boxes:
                self.grid_boxes[box_key] = []
            self.grid_boxes[box_key].append(sol)

    def select_parent(self):
        keys = list(self.grid_boxes.keys())
        if not keys: return None
        if len(keys) == 1:
            winning_box = keys[0]
        else:
            idx1, idx2 = self.rng.choice(len(keys), size=2, replace=False)
            b1, b2 = keys[idx1], keys[idx2]
            winning_box = b1 if len(self.grid_boxes[b1]) <= len(self.grid_boxes[b2]) else b2
        candidates = self.grid_boxes[winning_box]
        return candidates[self.rng.randint(len(candidates))]

    def get_most_crowded_box(self):
        return max(self.grid_boxes.keys(), key=lambda b: len(self.grid_boxes[b]))


class PESA2:
    def __init__(self, problem, archive_size=100, population_size=10, bits_per_var=30, grid_divisions=10, max_evals=20000, seed=42):
        self.problem = problem
        self.archive_size = archive_size
        self.population_size = population_size
        self.bits_per_var = bits_per_var
        self.chromosome_len = problem.n_vars * bits_per_var
        self.grid_divisions = grid_divisions
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        
        self.grid = HyperboxGrid(problem.n_objs, grid_divisions, self.rng)
        self.archive = []
        self.evals = 0

    def decode(self, chromosome):
        decoded = np.zeros(self.problem.n_vars)
        for i in range(self.problem.n_vars):
            bits = chromosome[i * self.bits_per_var:(i + 1) * self.bits_per_var]
            val = 0
            for b in bits: val = (val << 1) | int(b)
            norm = val / ((1 << self.bits_per_var) - 1)
            low, high = self.problem.bounds[i]
            decoded[i] = low + norm * (high - low)
        return decoded

    def evaluate(self, sol):
        sol.objectives = self.problem.evaluate(self.decode(sol.chromosome))
        self.evals += 1

    def dominates(self, o1, o2):
        return np.all(o1 <= o2) and np.any(o1 < o2)

    def add_to_archive(self, sol):
        to_remove = []
        for member in self.archive:
            if self.dominates(member.objectives, sol.objectives): return False
            if self.dominates(sol.objectives, member.objectives): to_remove.append(member)
        for m in to_remove: self.archive.remove(m)
        
        self.archive.append(sol)
        self.grid.update_grid(self.archive)

        if len(self.archive) > self.archive_size:
            crowded_box = self.grid.get_most_crowded_box()
            members = self.grid.grid_boxes[crowded_box]
            self.archive.remove(members[self.rng.randint(len(members))])
            self.grid.update_grid(self.archive)
        return True

    def run(self):
        for _ in range(self.archive_size):
            sol = Solution(self.rng.randint(0, 2, self.chromosome_len, dtype=np.uint8))
            self.evaluate(sol)
            self.add_to_archive(sol)

        while self.evals < self.max_evals:
            ip = []
            for _ in range(self.population_size // 2):
                p1, p2 = self.grid.select_parent(), self.grid.select_parent()
                mask = self.rng.randint(0, 2, self.chromosome_len, dtype=np.uint8)
                c1_c = np.where(mask, p1.chromosome, p2.chromosome)
                c2_c = np.where(mask, p2.chromosome, p1.chromosome)
                
                pm = 1.0 / self.chromosome_len
                c1_c ^= (self.rng.rand(self.chromosome_len) < pm).astype(np.uint8)
                c2_c ^= (self.rng.rand(self.chromosome_len) < pm).astype(np.uint8)
                ip.extend([Solution(c1_c), Solution(c2_c)])

            for child in ip:
                if self.evals >= self.max_evals: break
                self.evaluate(child)
                self.add_to_archive(child)

        return np.array([sol.objectives for sol in self.archive])