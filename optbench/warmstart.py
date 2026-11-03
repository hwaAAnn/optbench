"""History-based warm start shared by all solvers."""
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def knn_elite_init(context_now, hist_context, hist_decision, n_elites: int,
                   hist_score: Optional[np.ndarray] = None, n_candidates: Optional[int] = None) -> np.ndarray:
    """Decision vectors that performed best in the past process states most similar to the current one.

    1. standardize the context variables and find `n_candidates` nearest past states,
    2. keep the `n_elites` of them with the highest historical score.
    Without `hist_score` the nearest states are used as they are.
    """
    hist_context = np.asarray(hist_context, dtype=float)
    hist_decision = np.asarray(hist_decision)
    med = np.nanmedian(hist_context, axis=0)
    hist_context = np.where(np.isnan(hist_context), med, hist_context)
    scaler = StandardScaler().fit(hist_context)

    k = (n_candidates or 5 * n_elites) if hist_score is not None else n_elites
    knn = NearestNeighbors(n_neighbors=min(k, len(hist_context))).fit(scaler.transform(hist_context))
    now = np.where(np.isnan(context_now), med, context_now)
    _, idx = knn.kneighbors(scaler.transform(np.asarray(now, dtype=float)[None, :]))
    idx = idx[0]
    if hist_score is not None:
        idx = idx[np.argsort(-np.asarray(hist_score)[idx])[:n_elites]]
    return hist_decision[idx]
