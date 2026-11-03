"""Surrogate-assisted optimization on a synthetic black box.

    python run_surrogate_demo.py

1. distill the black-box indicator models into quadratic surrogates (space-filling samples x historical contexts)
2. fix the current context -> quadratic forms in the decision variables only
3. solve them with a gradient-based NLP (multi-start SLSQP) and, for comparison, with PSO running on the surrogate
4. every solution is re-evaluated with the *black box*; the surrogate itself is never trusted for the final score
"""
import numpy as np
import pandas as pd

from optbench import PSO, SurrogateNLP, knn_elite_init
from optbench.surrogate import QuadraticSurrogate, augment_samples, make_surrogate_problem, penalized_fitness
from optbench.toy_indicators import make_toy_indicators

N_AUGMENT = 3000
toy = make_toy_indicators()
names = ["indicator_1", "indicator_2", "indicator_3"]

# 1) distillation
X, C, Y = augment_samples(toy.blackbox, toy.lb, toy.ub, toy.hist_context, N_AUGMENT)
surrogates = {k: QuadraticSurrogate(len(toy.lb)).fit(X, C, Y[k]) for k in names}

# 2) condition on the current context
reduced = {k: s.reduce(toy.context_now) for k, s in surrogates.items()}

# warm start from similar past states
elites = knn_elite_init(toy.context_now, toy.hist_context, toy.hist_decision, n_elites=9, hist_score=toy.hist_score)


def true_fitness(x):
    ind = toy.blackbox(x[None, :], toy.context_now[None, :])
    return float(penalized_fitness(ind, toy.objective, toy.constraints, toy.penalty)[0]), {k: float(v[0]) for k, v in ind.items()}


# 3) solvers on the surrogate
nlp = SurrogateNLP(penalty=toy.penalty).run(reduced, toy.objective, toy.constraints, toy.lb, toy.ub, toy.x_now, elites)
sur_problem = make_surrogate_problem(reduced, toy.objective, toy.constraints, toy.penalty, toy.lb, toy.ub)
pso = PSO(swarm_size=30, iters=100).run(sur_problem, toy.x_now, elites)

# 4) black-box re-evaluation
rows = []
for label, x in [("current", toy.x_now), ("best KNN elite", elites[np.argmax([true_fitness(e)[0] for e in elites])]),
                 ("Surrogate-NLP", nlp.best_x), ("PSO on surrogate", pso.best_x)]:
    fit, ind = true_fitness(x)
    sur = float(penalized_fitness({k: r.value(x) for k, r in reduced.items()}, toy.objective, toy.constraints,
                                  toy.penalty)[0])
    rows.append({"solution": label, "true fitness": fit, "surrogate fitness": sur,
                 "constraints ok": all(c.slack({k: np.array([v]) for k, v in ind.items()})[0] >= -1e-6
                                       for c in toy.constraints)})

print(f"\nblack-box queries used to build the surrogates: {N_AUGMENT}")
print(pd.DataFrame(rows).round(1).to_string(index=False))
