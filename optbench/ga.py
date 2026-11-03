"""Genetic algorithm: elitism, tournament selection, uniform / multi-parent crossover, step mutation."""
from dataclasses import dataclass

import numpy as np

from .problem import Problem, Result, Solver, init_population


@dataclass
class GA(Solver):
    pop_size: int = 10
    iters: int = 20
    elite_size: int = 2
    tournament_k: int = 3
    n_parents: int = 3                   # 2: uniform crossover, 3: three-parent uniform crossover
    crossover_rate: float = 1.0
    mutation_rate: float = 0.05          # per gene
    mutation_step: float = 5.0           # max step of a mutation (variable units)
    mutation_schedule: str = "increasing"  # "constant" | "increasing" | "decreasing" step amplitude over the run
    seed: int = 0
    label: str = "GA"

    def _tournament(self, fit, exclude, rng):
        pool = np.array([i for i in range(len(fit)) if i not in exclude])
        cand = rng.choice(pool, size=min(self.tournament_k, len(pool)), replace=False)
        return int(cand[np.argmax(fit[cand])])

    def _crossover(self, parents, rng):
        P = np.stack(parents)                                    # (n_parents, dim)
        dim = P.shape[1]
        pick = lambda: P[rng.integers(0, len(P), dim), np.arange(dim)]   # each gene from a random parent
        return pick(), pick()

    def _amplitude(self, t) -> float:
        p = t / max(self.iters - 1, 1)
        scale = {"constant": 1.0, "increasing": p, "decreasing": 1.0 - p}[self.mutation_schedule]
        return self.mutation_step * scale

    def run(self, problem: Problem, x0, elites=None) -> Result:
        rng = np.random.default_rng(self.seed)
        X = init_population(problem, self.pop_size, x0, elites, rng)
        best_x, best_fit, hist = None, -np.inf, []

        for t in range(self.iters):
            fit = problem.evaluate(X)
            order = np.argsort(-fit)
            if fit[order[0]] > best_fit:
                best_fit, best_x = float(fit[order[0]]), X[order[0]].copy()
            hist.append(best_fit)

            new = [X[i].copy() for i in order[: self.elite_size]]            # elitism (kept unmutated)
            children = []
            while len(new) + len(children) < self.pop_size:
                chosen = []
                for _ in range(self.n_parents):
                    chosen.append(self._tournament(fit, chosen, rng))
                parents = [X[i] for i in chosen]
                if rng.random() < self.crossover_rate:
                    children += list(self._crossover(parents, rng))
                else:
                    children += [parents[0].copy(), parents[1].copy()]
            children = np.array(children[: self.pop_size - len(new)])

            amp = self._amplitude(t)
            if amp > 0:
                mask = rng.random(children.shape) < self.mutation_rate
                children = children + mask * rng.uniform(-amp, amp, size=children.shape)

            X = problem.repair(np.vstack([new, children]), (t + 1) / self.iters)

        return Result(best_x, best_fit, hist, problem.n_evals)
