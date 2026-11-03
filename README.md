# optbench: Metaheuristics and Surrogate-Assisted Solvers for Set-point Optimization

Clean, self-contained implementations of **PSO, multi-swarm PSO, GA, Simulated Annealing, Tabu Search** and a **quadratic-surrogate NLP solver** behind one interface, extracted from an industry–academia project on operating-condition optimization of a continuous process (see the overview repository for the full pipeline).

> **Disclosure** — No project data, models, variable names, or KPI formulas are included. The solvers only see a black-box `evaluate(X) -> fitness`; the included toy problems are synthetic. No benchmark results are published here.

## Problem setting

- A solution is a **continuous vector** of set-points inside box bounds. The real actuators have a resolution of one unit on a scale of a few hundred, so variables are treated as continuous.
- Fitness is a black box (a model-based simulator in the original project) and is evaluated in **batches**, so an expensive simulator can parallelize inside `evaluate`.
- Constraints are handled as a **relative-violation penalty**: `fitness = objective - penalty * violation`.
- A **soft bound** can tighten the upper bound of one variable over the run (`SoftBound`).
- Search can be **warm-started** from set-points that performed best in past process states similar to the current one (`knn_elite_init`).

```python
from optbench import Problem, PSO

problem = Problem(lb, ub, evaluate)               # evaluate: (n, dim) -> (n,)
result = PSO(swarm_size=10, iters=20).run(problem, x_now, elites)
result.best_x, result.best_fit, result.history
```

## Metaheuristics (black-box fitness)

| Solver | File | Main ingredients | Variants explored |
|---|---|---|---|
| `PSO` | `optbench/pso.py` | inertia + personal/global pull, velocity clamp, boundary reflection with damping, random-reset mutation | — |
| `PSOMultiSwarm` | `optbench/pso.py` | sub-swarm best + global best, optional probabilistic ±1 moves for sub-unit velocities | number of sub-swarms |
| `GA` | `optbench/ga.py` | elitism, tournament selection, uniform / three-parent crossover, step mutation | mutation schedule (constant / increasing / decreasing), parent count |
| `SimulatedAnnealing` | `optbench/sa.py` | batched neighborhood, best-of-batch proposal, Metropolis acceptance | cooling (geometric / re-heating / acceptance-adaptive), neighbor generator (multi-gene / stagnation-adaptive), SA-Tabu hybrid |
| `TabuSearch` | `optbench/tabu.py` | single-gene moves, tabu list of moves, aspiration | step (unit / multi / decaying), aspiration (basic / epsilon), adaptive tenure |

## Surrogate-assisted optimization

When the indicator models are too expensive to query thousands of times, replace them by a cheap surrogate and optimize that instead.

1. **Distill** each black-box indicator model into a degree-2 polynomial over (decision + context) variables, using space-filling decision samples combined with historical contexts (`augment_samples`, `QuadraticSurrogate.fit`).
2. **Condition on the current context.** Every term containing a context variable collapses into a constant, linear or quadratic coefficient of the decision variables, giving `f(x) = c0 + g·x + xᵀHx` per indicator (`QuadraticSurrogate.reduce`, exact up to floating-point error).
3. **Solve** objective + ratio constraints (`IndicatorConstraint`) with analytic gradients using multi-start SLSQP (`SurrogateNLP`), starting from the current solution, warm-start elites and random points. The original work used a MILP formulation for integer set-points; with continuous variables a smooth NLP is sufficient.
4. Any metaheuristic can also run on the reduced surrogate via `make_surrogate_problem`.
5. **Always re-evaluate the final solution with the black box**; the surrogate is only used to search.

## Warm start

`knn_elite_init` standardizes the monitored (context) variables, finds the `n_candidates` most similar past states, and keeps the `n_elites` whose historical score was highest. Population methods start from `[elites, current solution, random...]`; trajectory methods (SA / Tabu) can start from the best of the current solution and the elites (`use_elites=True`).

## Run

```bash
pip install -r requirements.txt
python run_benchmark.py                          # all metaheuristics, same evaluation budget, synthetic problem
python run_benchmark.py --budget 400 --seeds 10 --no-warmstart
python run_surrogate_demo.py                     # distillation -> reduced quadratic -> NLP / PSO, re-checked on the black box
```

## Structure

```
optbench/
  problem.py          Problem, SoftBound, Result, shared helpers (repair, init, mutation)
  warmstart.py        KNN warm start
  pso.py  ga.py  sa.py  tabu.py
  surrogate.py        quadratic surrogate, context reduction, augmentation, penalized fitness
  nlp.py              multi-start SLSQP on the reduced surrogates
  toy.py              synthetic black-box problem + history (metaheuristics)
  toy_indicators.py   synthetic indicator models + constraints (surrogate demo)
run_benchmark.py  run_surrogate_demo.py
```
