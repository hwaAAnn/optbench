"""Particle Swarm Optimization variants for continuous set-points.

PSO            : inertia + cognitive + social, velocity clamp, boundary reflection, random-reset mutation.
PSOMultiSwarm  : swarm split into sub-swarms; each particle is pulled by its personal best, its sub-swarm's
                 best and the global best. Optionally (`prob_move`) sub-unit velocities become probabilistic
                 +/-1 moves (a velocity of 0.3 moves the gene by one unit with probability 0.3), a stochastic
                 step that works well when one unit is already the actuator resolution.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .problem import Problem, Result, Solver, init_population, random_reset


def bounce(X, V, lb, ub, decay):
    """Reflect positions that left the box and damp/reverse the velocity of those genes."""
    low = X < lb
    X = np.where(low, 2 * lb - X, X)
    V = np.where(low, -decay * V, V)
    up = X > ub
    X = np.where(up, 2 * ub - X, X)
    V = np.where(up, -decay * V, V)
    return X, V


@dataclass
class PSO(Solver):
    swarm_size: int = 10
    iters: int = 20
    w: float = 0.7                       # inertia
    c1: float = 1.5                      # cognitive
    c2: float = 1.5                      # social
    vmax_ratio: Optional[float] = 0.7    # velocity clamp as a fraction of the range
    mutation_rate: float = 0.05
    bounce_decay: float = 0.5
    seed: int = 0
    label: str = "PSO"

    def run(self, problem: Problem, x0, elites=None) -> Result:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        X = init_population(problem, self.swarm_size, x0, elites, rng).astype(float)
        V = np.zeros_like(X)
        vmax = None if self.vmax_ratio is None else self.vmax_ratio * (ub - lb)
        pbest, pbest_fit = X.copy(), np.full(len(X), -np.inf)
        gbest, gbest_fit, hist = None, -np.inf, []

        for t in range(self.iters):
            fit = problem.evaluate(X)
            better = fit > pbest_fit
            pbest[better], pbest_fit[better] = X[better], fit[better]
            b = int(np.argmax(fit))
            if fit[b] > gbest_fit:
                gbest_fit, gbest = float(fit[b]), X[b].copy()
            hist.append(gbest_fit)

            r1, r2 = rng.random(X.shape), rng.random(X.shape)
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            if vmax is not None:
                V = np.clip(V, -vmax, vmax)
            X, V = bounce(X + V, V, lb, ub, self.bounce_decay)
            progress = (t + 1) / self.iters
            X = problem.repair(X, progress).astype(float)
            X = problem.repair(random_reset(X, self.mutation_rate, problem, rng), progress).astype(float)

        return Result(gbest, gbest_fit, hist, problem.n_evals)


@dataclass
class PSOMultiSwarm(Solver):
    swarm_size: int = 10
    iters: int = 20
    n_subswarms: int = 2
    w: float = 0.7
    c1: float = 1.5                      # personal best
    c2: float = 1.5                      # sub-swarm best
    c3: float = 0.7                      # global best
    prob_move: bool = True               # probabilistic +/-1 step for |velocity| < 1
    mutation_rate: float = 0.05
    bounce_decay: float = 0.5
    seed: int = 0
    label: str = "PSO-MultiSwarm"

    def run(self, problem: Problem, x0, elites=None) -> Result:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        X = init_population(problem, self.swarm_size, x0, elites, rng).astype(float)
        V = np.zeros_like(X)
        n = len(X)
        sub = np.arange(n) * self.n_subswarms // n                 # sub-swarm id of each particle
        pbest, pbest_fit = X.copy(), np.full(n, -np.inf)
        sub_best = np.zeros((self.n_subswarms, X.shape[1]))
        sub_fit = np.full(self.n_subswarms, -np.inf)
        gbest, gbest_fit, hist = None, -np.inf, []

        for t in range(self.iters):
            fit = problem.evaluate(X)
            better = fit > pbest_fit
            pbest[better], pbest_fit[better] = X[better], fit[better]
            for s in range(self.n_subswarms):
                members = np.where(sub == s)[0]
                b = members[int(np.argmax(fit[members]))]
                if fit[b] > sub_fit[s]:
                    sub_fit[s], sub_best[s] = fit[b], X[b].copy()
            b = int(np.argmax(fit))
            if fit[b] > gbest_fit:
                gbest_fit, gbest = float(fit[b]), X[b].copy()
            hist.append(gbest_fit)

            r1, r2, r3 = rng.random(X.shape), rng.random(X.shape), rng.random(X.shape)
            V = (self.w * V + self.c1 * r1 * (pbest - X)
                 + self.c2 * r2 * (sub_best[sub] - X) + self.c3 * r3 * (gbest - X))

            move = V
            if self.prob_move:                           # probability-based move for sub-unit velocities
                prob_step = np.sign(V) * (rng.random(V.shape) < np.abs(V))
                move = np.where(np.abs(V) < 1.0, prob_step, V)
            X, V = bounce(X + move, V, lb, ub, self.bounce_decay)
            progress = (t + 1) / self.iters
            X = problem.repair(X, progress).astype(float)
            X = problem.repair(random_reset(X, self.mutation_rate, problem, rng), progress).astype(float)

        return Result(gbest, gbest_fit, hist, problem.n_evals)
