"""
algorithms/de.py
Standard Differential Evolution (DE/rand/1/bin) for Single Objective CEC 2022.
Includes strict type-casting and bulletproof checkpoint tracking.
"""
import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_func, bounds, pop_size=100, mutation_factor=0.5, crossover_prob=0.9, max_evals=200000, seed=42):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.F = mutation_factor
        self.CR = crossover_prob
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        
        self.dim = len(bounds)
        self.evals = 0
        
    def evaluate(self, x):
        """Evaluates the solution and strictly forces it to a scalar float."""
        self.evals += 1
        val = self.objective_func(x)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(val[0])

    def run(self, checkpoints):
        """Runs DE and perfectly records fitness at the 16 exact evaluation checkpoints."""
        # Initialize Population
        pop = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([self.evaluate(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        
        results = []
        checkpoint_idx = 0
        
        # 1. Catch any checkpoints that were crossed during initial population evaluation
        while checkpoint_idx < len(checkpoints) and self.evals >= checkpoints[checkpoint_idx]:
            results.append(best_val)
            checkpoint_idx += 1
            
        # 2. Main DE Evolutionary Loop
        while self.evals < self.max_evals and best_val > 1e-8:
            for i in range(self.pop_size):
                
                # Prevent overshooting MaxFES
                if self.evals >= self.max_evals:
                    break
                    
                # Selection
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = self.rng.choice(candidates, 3, replace=False)
                
                # Mutation
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
                
                # Crossover
                cross_points = self.rng.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluation
                trial_fitness = self.evaluate(trial)
                
                # Update Population
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_val:
                        best_val = trial_fitness
                
                # 3. Safely record checkpoint if crossed (using while loop prevents skipping)
                while checkpoint_idx < len(checkpoints) and self.evals >= checkpoints[checkpoint_idx]:
                    results.append(best_val)
                    checkpoint_idx += 1
                    
                # Early termination condition (Error <= 1e-8)
                if best_val <= 1e-8:
                    break
                    
        # 4. If we reached the target early (or loop ended naturally), pad remaining checkpoints
        while len(results) < len(checkpoints):
            results.append(1e-8)
            
        # 5. Append the exact number of evaluations taken (FE_term) as the mandatory 17th entry
        results.append(self.evals) 
        
        return results