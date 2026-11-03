"""Quadratic surrogates of black-box indicator models, reduced to the decision variables.

Idea
    The indicator models used by the simulator are expensive black boxes. A degree-2 polynomial fitted
    on (decision + context) variables approximates them. Once the *current* context is known, every term
    that contains a context variable collapses into a constant, a linear or a quadratic coefficient of the
    decision variables, so each indicator becomes
            f(x) = c0 + g.x + x'Hx           (x = decision variables only)
    which is cheap to evaluate and has analytic gradients, so a gradient-based NLP solver can be used.

Training data
    `augment_samples` queries the black box on space-filling decision samples combined with historical
    contexts (model distillation), which gives far more data than the real labels alone.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .problem import Problem

BlackBox = Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]]    # (X_decision, X_context) -> {indicator: (n,)}


@dataclass
class IndicatorConstraint:
    """sum_k coefs[k] * indicator_k <= rhs   (e.g. a ratio bound  a - r*b <= 0)."""
    coefs: Dict[str, float]
    rhs: float = 0.0

    def slack(self, ind: Dict[str, np.ndarray]) -> np.ndarray:
        """>= 0 when satisfied."""
        return self.rhs - sum(c * ind[k] for k, c in self.coefs.items())


def penalized_fitness(ind: Dict[str, np.ndarray], objective: str, constraints: List[IndicatorConstraint],
                      penalty: float) -> np.ndarray:
    violation = sum(np.maximum(-c.slack(ind), 0.0) for c in constraints)
    return ind[objective] - penalty * violation


def augment_samples(blackbox: BlackBox, lb, ub, hist_context: np.ndarray, n: int, seed: int = 0):
    """Latin-hypercube decision samples x random historical contexts, labeled by the black box."""
    rng = np.random.default_rng(seed)
    X = qmc.scale(qmc.LatinHypercube(d=len(lb), seed=seed).random(n), lb, ub)
    C = hist_context[rng.integers(0, len(hist_context), n)]
    return X, C, blackbox(X, C)


@dataclass
class ReducedQuadratic:
    """f(x) = c0 + g.x + x'Hx with H symmetric."""
    c0: float
    g: np.ndarray
    H: np.ndarray

    def value(self, X) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.c0 + X @ self.g + np.einsum("ni,ij,nj->n", X, self.H, X)

    def grad(self, x) -> np.ndarray:
        return self.g + 2 * self.H @ x


class QuadraticSurrogate:
    def __init__(self, n_decision: int):
        self.n_dec = n_decision

    def fit(self, X: np.ndarray, C: np.ndarray, y: np.ndarray) -> "QuadraticSurrogate":
        F = np.hstack([X, C])
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-12
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.reg = LinearRegression().fit(self.poly.fit_transform((F - self.mu) / self.sd), y)
        return self

    def predict(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        F = (np.hstack([X, C]) - self.mu) / self.sd
        return self.reg.predict(self.poly.transform(F))

    def reduce(self, context: np.ndarray) -> ReducedQuadratic:
        """Fix the context at its current value and return the quadratic form in the decision variables."""
        d = self.n_dec
        ctx = (np.asarray(context, float) - self.mu[d:]) / self.sd[d:]          # standardized context
        c0 = float(self.reg.intercept_)
        g, H = np.zeros(d), np.zeros((d, d))                                    # in standardized decision space
        for w, e in zip(self.reg.coef_, self.poly.powers_):
            factor = w * np.prod(ctx ** e[d:])
            ed = e[:d]
            n = int(ed.sum())
            if n == 0:
                c0 += factor
            elif n == 1:
                g[int(np.argmax(ed))] += factor
            else:
                idx = np.flatnonzero(ed)
                i, j = (idx[0], idx[0]) if len(idx) == 1 else (idx[0], idx[1])
                if i == j:
                    H[i, i] += factor
                else:
                    H[i, j] += factor / 2
                    H[j, i] += factor / 2

        # back to raw decision units:  u = D x - m  with D = diag(1/sd), m = mu/sd
        D, m = 1.0 / self.sd[:d], self.mu[:d] / self.sd[:d]
        Hm = H @ m
        return ReducedQuadratic(c0 - g @ m + m @ Hm, D * g - 2 * D * Hm, H * np.outer(D, D))


def make_surrogate_problem(reduced: Dict[str, ReducedQuadratic], objective: str,
                           constraints: List[IndicatorConstraint], penalty: float, lb, ub, soft_bound=None) -> Problem:
    """Wrap the reduced surrogates as an ordinary `Problem`, so any metaheuristic can run on them."""
    def evaluate(X):
        return penalized_fitness({k: r.value(X) for k, r in reduced.items()}, objective, constraints, penalty)
    return Problem(lb, ub, evaluate, soft_bound)
