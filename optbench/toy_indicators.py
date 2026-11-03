"""Synthetic black-box indicator models for the surrogate demo. Nothing here is derived from real operations.

Three indicators depend on decision variables and a drifting process context:
  indicator_1 : primary indicator (maximized), concave in the decisions with a context-dependent optimum
  indicator_2 / indicator_3 : enter two ratio constraints, calibrated so that roughly half of the box is feasible
The context effect goes through tanh and a sine ripple, so the quadratic surrogate is a good but imperfect fit.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .surrogate import IndicatorConstraint


@dataclass
class ToyIndicators:
    blackbox: object
    lb: np.ndarray
    ub: np.ndarray
    x_now: np.ndarray
    context_now: np.ndarray
    hist_context: np.ndarray
    hist_decision: np.ndarray
    hist_score: np.ndarray
    objective: str
    constraints: List[IndicatorConstraint]
    penalty: float


def make_toy_indicators(dim: int = 12, n_context: int = 6, n_hist: int = 2000, seed: int = 0) -> ToyIndicators:
    rng = np.random.default_rng(seed)
    lb, ub = np.zeros(dim), np.full(dim, 300.0)                    # set-points on a scale of a few hundred units
    span, mid = ub - lb, (ub + lb) / 2
    A = rng.normal(0, 1, (n_context, dim)) / np.sqrt(n_context)
    w2, w3 = rng.normal(0, 1, dim), rng.normal(0, 1, dim)
    u2, u3 = rng.normal(0, 1, n_context), rng.normal(0, 1, n_context)

    def blackbox(X, C) -> Dict[str, np.ndarray]:
        X, C = np.atleast_2d(X), np.atleast_2d(C)
        z = (X - mid) / span                                        # in [-0.5, 0.5]
        ideal = 0.3 * np.tanh(C @ A)
        return {
            "indicator_1": 500 - 600 * np.mean((z - ideal) ** 2, axis=1) + 8 * np.sin(4 * z[:, 0]),
            "indicator_2": 200 + 80 * (z @ w2) / np.sqrt(dim) + 15 * (C @ u2) + 6 * np.sin(3 * z[:, 1]),
            "indicator_3": 300 + 60 * (z @ w3) / np.sqrt(dim) + 10 * (C @ u3) + 5 * np.cos(3 * z[:, 2]),
        }

    walk = np.cumsum(rng.normal(0, 0.1, (n_hist + 1, n_context)), axis=0)
    walk = (walk - walk.mean(0)) / walk.std(0)
    hist_ctx, ctx_now = walk[:-1], walk[-1]

    # constraint ratios calibrated at the current context: about half of the box satisfies each of them
    Z = rng.uniform(lb, ub, (4000, dim))
    ref = blackbox(Z, np.tile(ctx_now, (len(Z), 1)))
    r23 = float(np.median(ref["indicator_2"] / ref["indicator_3"]))
    r31 = float(np.quantile(ref["indicator_3"] / ref["indicator_1"], 0.6))
    constraints = [
        IndicatorConstraint({"indicator_2": 1.0, "indicator_3": -r23}),        # i2 - r23 * i3 <= 0
        IndicatorConstraint({"indicator_3": 1.0, "indicator_1": -r31}),        # i3 - r31 * i1 <= 0
    ]
    penalty = 1e3

    # history: operators set decisions near a context-dependent target, with noise
    ideal_h = mid + span * 0.3 * np.tanh(hist_ctx @ A)
    hist_dec = np.clip(ideal_h + rng.normal(0, 0.12, ideal_h.shape) * span, lb, ub)
    from .surrogate import penalized_fitness
    hist_score = penalized_fitness(blackbox(hist_dec, hist_ctx), "indicator_1", constraints, penalty)

    ideal_n = mid + span * 0.3 * np.tanh(ctx_now @ A)
    x_now = np.clip(ideal_n + rng.normal(0, 0.25, dim) * span, lb, ub)
    return ToyIndicators(blackbox, lb, ub, x_now, ctx_now, hist_ctx, hist_dec, hist_score,
                         "indicator_1", constraints, penalty)
