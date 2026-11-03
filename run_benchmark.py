"""Run every solver on the synthetic problem with the same evaluation budget.

    python run_benchmark.py                 # 200 evaluations per run, 5 seeds
    python run_benchmark.py --budget 400 --seeds 10 --no-warmstart
"""
import argparse
import copy

import numpy as np
import pandas as pd

from optbench import GA, PSO, PSOMultiSwarm, SimulatedAnnealing, TabuSearch, knn_elite_init
from optbench.toy import make_toy

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=200, help="fitness evaluations per run")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--no-warmstart", action="store_true")
args = ap.parse_args()

PER_ITER = 10                                   # particles / individuals / neighbors per iteration
iters = args.budget // PER_ITER

toy = make_toy()
elites = None
if not args.no_warmstart:
    elites = knn_elite_init(toy.context_now, toy.hist_context, toy.hist_decision, n_elites=PER_ITER - 1,
                            hist_score=toy.hist_score)

solvers = [
    PSO(swarm_size=PER_ITER, iters=iters),
    PSOMultiSwarm(swarm_size=PER_ITER, iters=iters),
    GA(pop_size=PER_ITER, iters=iters),
    SimulatedAnnealing(n_neighbors=PER_ITER, iters=iters, t0=20.0, t_end=0.2, use_elites=True),
    SimulatedAnnealing(n_neighbors=PER_ITER, iters=iters, t0=20.0, t_end=0.2, neighbor="adaptive",
                       tabu_tenure=20, use_elites=True, label="SA+Tabu"),
    TabuSearch(n_neighbors=PER_ITER, iters=iters, use_elites=True),
]

rows = []
for proto in solvers:
    best = []
    for seed in range(args.seeds):
        s = copy.deepcopy(proto)
        s.seed = seed
        toy.problem.n_evals = 0
        best.append(s.run(toy.problem, toy.x_now, elites).best_fit)
    rows.append({"solver": proto.label, "mean": np.mean(best), "std": np.std(best), "max": np.max(best)})

base = float(toy.problem.evaluate(toy.x_now[None, :])[0])
print(f"\nbudget={args.budget} evals, seeds={args.seeds}, warm start={'off' if args.no_warmstart else 'on'}")
print(f"fitness of the current solution: {base:.1f}\n")
print(pd.DataFrame(rows).round(1).to_string(index=False))
