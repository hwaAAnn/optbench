"""Simulated annealing with batched neighborhood evaluation.

Every iteration draws `n_neighbors` candidates, evaluates them as one batch, proposes the best one and
accepts it with the Metropolis rule. Variants:
  schedule : "geometric" | "reheat" (periodic re-heating) | "adaptive" (cools slower when acceptance is low)
  neighbor : "multi"     (1-3 genes, creep or reset, probabilities/steps decay over the run)
             "adaptive"  (1 gene, reset probability jumps when the search stagnates)
  tabu_tenure > 0 : SA-Tabu hybrid, recently accepted solutions are rejected.
Temperatures are in fitness units: scale `t0` / `t_end` to the fitness range of your problem.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .problem import Problem, Result, Solver, best_start


@dataclass
class SimulatedAnnealing(Solver):
    n_neighbors: int = 10
    iters: int = 20
    t0: float = 1.0
    t_end: float = 1e-3
    schedule: str = "geometric"
    reheat_interval: int = 10
    reheat_factor: float = 1.5
    target_accept: float = 0.3
    alpha: float = 0.9
    neighbor: str = "multi"
    p_reset0: float = 0.2
    p_reset1: float = 0.05
    step0: float = 5.0
    step1: float = 1.0
    stagnation_th: int = 5
    tabu_tenure: int = 0
    use_elites: bool = False             # start from the best of {current} + warm-start elites
    seed: int = 0
    label: str = "SA"

    def _temperature(self, it: int, accept_rate: float) -> float:
        T = self.t0 * (self.t_end / self.t0) ** (it / self.iters)
        if self.schedule == "reheat" and it > 0 and it % self.reheat_interval == 0:
            T = min(self.t0, T * self.reheat_factor)
        elif self.schedule == "adaptive":
            T *= (1.0 / self.alpha) if accept_rate < self.target_accept else self.alpha
        return max(T, self.t_end)

    def _neighbors(self, x, it, stagnation, problem: Problem, rng) -> np.ndarray:
        progress = it / self.iters
        lb, ub = problem.lb, problem.ub
        step_max = max(self.step1, self.step0 + (self.step1 - self.step0) * progress)
        out = []
        for _ in range(self.n_neighbors):
            y = x.copy()
            if self.neighbor == "adaptive":
                p_reset = 0.7 if stagnation >= self.stagnation_th else 0.2
                idx = np.array([rng.integers(problem.dim)])
            else:
                p_reset = self.p_reset0 + (self.p_reset1 - self.p_reset0) * progress
                idx = rng.choice(problem.dim, size=int(rng.integers(1, 4)), replace=False)
            if rng.random() < p_reset:
                y[idx] = rng.uniform(lb[idx], ub[idx])                           # reset: exploration
            else:
                y[idx] += rng.uniform(-step_max, step_max, size=len(idx))        # creep: exploitation
            out.append(y)
        return problem.repair(np.array(out), progress)

    def run(self, problem: Problem, x0, elites=None) -> Result:
        rng = np.random.default_rng(self.seed)
        cur_x, cur_fit = best_start(problem, x0, elites, self.use_elites)
        best_x, best_fit, hist = cur_x.copy(), cur_fit, []
        tabu = deque(maxlen=self.tabu_tenure) if self.tabu_tenure > 0 else None
        key = lambda v: tuple(np.round(v, 6))            # hashable solution identity for the tabu memory
        if tabu is not None:
            tabu.append(key(cur_x))
        accepts, stagnation = [], 0

        for it in range(self.iters):
            accept_rate = float(np.mean(accepts[-5:])) if accepts else 1.0
            T = self._temperature(it, accept_rate)

            cand = self._neighbors(cur_x, it, stagnation, problem, rng)
            fit = problem.evaluate(cand)
            j = int(np.argmax(fit))
            prop_x, prop_fit = cand[j], float(fit[j])

            blocked = tabu is not None and key(prop_x) in tabu
            ok = (not blocked) and (prop_fit >= cur_fit or (T > 1e-12 and rng.random() < np.exp((prop_fit - cur_fit) / T)))
            accepts.append(int(ok))
            if ok:
                cur_x, cur_fit = prop_x.copy(), prop_fit
                if tabu is not None:
                    tabu.append(key(cur_x))

            if cur_fit > best_fit:
                best_x, best_fit, stagnation = cur_x.copy(), cur_fit, 0
            else:
                stagnation += 1
            hist.append(best_fit)

        return Result(best_x, best_fit, hist, problem.n_evals)
