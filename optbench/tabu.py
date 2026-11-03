"""Tabu search over single-gene moves (fixed-size steps, so moves can recur and be recognized as tabu).

A move is (gene index, new value). The best admissible neighbor becomes the next solution even if it is
worse, and its move is stored in the tabu list. A tabu move is still allowed if it satisfies the
aspiration criterion (better than the global best, or within `eps` of it for the "epsilon" variant).
  move       : "unit" (+/-1) | "multi" (+/-1 with 70%, +/-2 with 30%) | "adaptive" (step decays max_step -> min_step)
  adaptive_tenure : lengthen the tabu list when every neighbor is blocked, shorten it after progress.
"""
from dataclasses import dataclass

import numpy as np

from .problem import Problem, Result, Solver, best_start


@dataclass
class TabuSearch(Solver):
    n_neighbors: int = 10
    iters: int = 20
    tenure: int = 15
    move: str = "adaptive"
    max_step: float = 5.0
    min_step: float = 1.0
    aspiration: str = "basic"            # "basic" | "epsilon"
    eps: float = 0.01
    adaptive_tenure: bool = False
    tenure_min: int = 3
    tenure_max: int = 15
    use_elites: bool = False
    seed: int = 0
    label: str = "TabuSearch"

    def _step(self, it, rng) -> float:
        if self.move == "unit":
            return 1.0
        if self.move == "multi":
            return 1.0 if rng.random() < 0.7 else 2.0
        return max(self.min_step, self.max_step - (self.max_step - self.min_step) * it / self.iters)

    def _neighbors(self, x, it, problem: Problem, rng):
        cand, moves = [], []
        for _ in range(self.n_neighbors):
            y = x.copy()
            j = int(rng.integers(problem.dim))
            y[j] += int(rng.choice([-1, 1])) * self._step(it, rng)
            y = problem.repair(y, it / self.iters)
            cand.append(y)
            moves.append((j, round(float(y[j]), 6)))
        return np.array(cand), moves

    def _aspires(self, fit, best_fit) -> bool:
        if self.aspiration == "epsilon":
            return fit >= best_fit - self.eps * abs(best_fit)
        return fit > best_fit

    def run(self, problem: Problem, x0, elites=None) -> Result:
        rng = np.random.default_rng(self.seed)
        cur_x, cur_fit = best_start(problem, x0, elites, self.use_elites)
        best_x, best_fit, hist = cur_x.copy(), cur_fit, []
        tabu, tenure, stuck = [], self.tenure, 0

        for it in range(self.iters):
            cand, moves = self._neighbors(cur_x, it, problem, rng)
            fit = problem.evaluate(cand)

            pick = None
            for i in np.argsort(-fit):                      # best-first, first admissible wins
                if moves[i] in tabu and not self._aspires(fit[i], best_fit):
                    continue
                pick = int(i)
                break

            if pick is None:                                # every neighbor is tabu: stay
                stuck += 1
                if self.adaptive_tenure and stuck >= 3:
                    tenure, stuck = min(tenure + 1, self.tenure_max), 0
            else:
                cur_x, cur_fit = cand[pick], float(fit[pick])
                tabu.append(moves[pick])
                stuck = 0
                if self.adaptive_tenure:
                    tenure = max(tenure - 1, self.tenure_min)
            tabu = tabu[-tenure:]

            if cur_fit > best_fit:
                best_x, best_fit = cur_x.copy(), cur_fit
            hist.append(best_fit)

        return Result(best_x, best_fit, hist, problem.n_evals)
