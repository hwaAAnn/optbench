"""Common problem / solver interface shared by every algorithm.

A problem is a box-constrained continuous vector with a black-box *batch* fitness (higher is better):

    fitness = objective - penalty * constraint_violation

`evaluate(X)` receives an (n, dim) array, so an expensive simulator can parallelize inside it.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class SoftBound:
    """Upper bound of one variable that is tightened linearly over the run (progress 0 -> 1)."""
    index: int
    start_ub: float
    end_ub: float

    def ub_at(self, progress: float) -> float:
        return self.start_ub - (self.start_ub - self.end_ub) * progress


class Problem:
    def __init__(self, lb, ub, evaluate: Callable[[np.ndarray], np.ndarray], soft_bound: Optional[SoftBound] = None):
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self._evaluate = evaluate
        self.soft_bound = soft_bound
        self.n_evals = 0

    @property
    def dim(self) -> int:
        return len(self.lb)

    def evaluate(self, X) -> np.ndarray:
        X = np.atleast_2d(X)
        self.n_evals += len(X)
        return np.asarray(self._evaluate(X), dtype=float)

    def repair(self, X, progress: float = 0.0) -> np.ndarray:
        """Clip to bounds and apply the (time-varying) soft bound. Works for 1-D and 2-D input."""
        X = np.clip(np.asarray(X, dtype=float), self.lb, self.ub)
        sb = self.soft_bound
        if sb is not None:
            j = sb.index
            X[..., j] = np.clip(X[..., j], self.lb[j], sb.ub_at(progress))
        return X

    def random_vectors(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lb, self.ub, size=(n, self.dim))


@dataclass
class Result:
    best_x: np.ndarray
    best_fit: float
    history: List[float] = field(default_factory=list)   # best-so-far fitness per iteration
    n_evals: int = 0


class Solver:
    label = "solver"

    def run(self, problem: Problem, x0: np.ndarray, elites: Optional[np.ndarray] = None) -> Result:
        raise NotImplementedError


def init_population(problem: Problem, size: int, x0, elites, rng: np.random.Generator) -> np.ndarray:
    """[warm-start elites..., current solution, random vectors...] truncated/padded to `size`."""
    rows = []
    if elites is not None and len(elites):
        rows += list(np.atleast_2d(elites)[: size - 1])
    rows.append(np.asarray(x0))
    if len(rows) < size:
        rows += list(problem.random_vectors(size - len(rows), rng))
    return problem.repair(np.array(rows), 0.0)


def random_reset(X: np.ndarray, rate: float, problem: Problem, rng: np.random.Generator) -> np.ndarray:
    """Each gene is re-sampled uniformly inside the bounds with probability `rate`."""
    if rate <= 0:
        return X
    mask = rng.random(X.shape) < rate
    return np.where(mask, rng.uniform(problem.lb, problem.ub, X.shape), X)


def best_start(problem: Problem, x0, elites, use_elites: bool):
    """Starting point of a trajectory method: the current solution, or the best of {current} + elites."""
    cands = [np.asarray(x0)]
    if use_elites and elites is not None and len(elites):
        cands += list(np.atleast_2d(elites))
    cands = problem.repair(np.array(cands), 0.0)
    fit = problem.evaluate(cands)
    b = int(np.argmax(fit))
    return cands[b], float(fit[b])
