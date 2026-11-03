"""Synthetic stand-in for a plant simulator. Nothing here is derived from real operations.

The best set-point drifts with the process context (so past states are informative for warm-starting),
the landscape has mild interactions and ripples (a few local optima), and one
soft constraint couples a group of variables:  fitness = objective - penalty * relative_violation.
"""
from dataclasses import dataclass

import numpy as np

from .problem import Problem, SoftBound


@dataclass
class ToyBundle:
    problem: Problem
    x_now: np.ndarray
    context_now: np.ndarray
    hist_context: np.ndarray
    hist_decision: np.ndarray
    hist_score: np.ndarray


def _ideal(ctx, A, mid, span):
    return mid + 0.3 * span * np.tanh(ctx @ A)


def _fitness(X, ideal, lb, span, penalty=2000.0):
    z = (X - ideal) / span
    objective = 1000 - 800 * np.mean(z ** 2, axis=1) - 100 * np.mean(z[:, :-1] * z[:, 1:], axis=1) \
        + 40 * np.mean(np.cos(6 * np.pi * z), axis=1)
    group = np.mean((X[:, :6] - lb[:6]) / span[:6], axis=1)          # constrained group of variables
    violation = np.maximum((group - 0.55) / 0.55, 0.0)
    return objective - penalty * violation


def make_toy(dim: int = 24, n_context: int = 8, n_hist: int = 3000, seed: int = 0) -> ToyBundle:
    rng = np.random.default_rng(seed)
    lb, ub = np.zeros(dim), np.full(dim, 100.0)
    span, mid = ub - lb, (lb + ub) / 2
    A = rng.normal(0, 1, (n_context, dim)) / np.sqrt(n_context)

    walk = np.cumsum(rng.normal(0, 0.1, (n_hist + 1, n_context)), axis=0)
    walk = (walk - walk.mean(0)) / walk.std(0)
    hist_ctx, ctx_now = walk[:-1], walk[-1]

    ideal_hist = _ideal(hist_ctx, A, mid, span)
    hist_dec = np.clip(ideal_hist + rng.normal(0, 0.12, ideal_hist.shape) * span, lb, ub)
    hist_score = _fitness(hist_dec, ideal_hist, lb, span)

    ideal_now = _ideal(ctx_now[None, :], A, mid, span)
    x_now = np.clip(ideal_now[0] + rng.normal(0, 0.25, dim) * span, lb, ub)

    problem = Problem(lb, ub, lambda X: _fitness(np.asarray(X, float), ideal_now, lb, span),
                      soft_bound=SoftBound(index=0, start_ub=95, end_ub=75))
    return ToyBundle(problem, x_now, ctx_now, hist_ctx, hist_dec, hist_score)
