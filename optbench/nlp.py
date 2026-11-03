"""Gradient-based solver on the reduced quadratic surrogates (multi-start SLSQP).

    maximize   indicator_obj(x)
    subject to sum_k a_k * indicator_k(x) <= rhs   for every IndicatorConstraint,   lb <= x <= ub

The surrogates are quadratic, so objective, constraints and their gradients are analytic. The problem is
non-convex in general, hence several starts (current solution, warm-start elites, random points).
Replaces the original integer MILP formulation, which is not needed once variables are treated as continuous.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from .problem import Result
from .surrogate import IndicatorConstraint, ReducedQuadratic, penalized_fitness


@dataclass
class SurrogateNLP:
    n_random_starts: int = 5
    max_iter: int = 200
    seed: int = 0
    penalty: float = 1e4                 # only used to rank the multi-start solutions
    label: str = "Surrogate-NLP"

    def run(self, reduced: Dict[str, ReducedQuadratic], objective: str, constraints: List[IndicatorConstraint],
            lb, ub, x0, elites: Optional[np.ndarray] = None) -> Result:
        rng = np.random.default_rng(self.seed)
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        starts = [np.asarray(x0, float)]
        if elites is not None and len(elites):
            starts += list(np.atleast_2d(elites))
        starts += list(rng.uniform(lb, ub, size=(self.n_random_starts, len(lb))))

        cons = []
        for c in constraints:
            cons.append({
                "type": "ineq",
                "fun": lambda x, c=c: float(c.rhs - sum(a * reduced[k].value(x)[0] for k, a in c.coefs.items())),
                "jac": lambda x, c=c: -sum(a * reduced[k].grad(x) for k, a in c.coefs.items()),
            })

        best_x, best_fit, hist, nfev = None, -np.inf, [], 0
        for s in starts:
            res = minimize(lambda x: -reduced[objective].value(x)[0], np.clip(s, lb, ub),
                           jac=lambda x: -reduced[objective].grad(x), method="SLSQP",
                           bounds=list(zip(lb, ub)), constraints=cons, options={"maxiter": self.max_iter})
            nfev += res.nfev
            x = np.clip(res.x, lb, ub)
            fit = float(penalized_fitness({k: r.value(x) for k, r in reduced.items()}, objective, constraints,
                                          self.penalty)[0])
            if fit > best_fit:
                best_x, best_fit = x, fit
            hist.append(best_fit)
        return Result(best_x, best_fit, hist, nfev)
