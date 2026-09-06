"""Item-based KNN collaborative filtering with shrinkage.

Prediction: r_ui = b_ui + sum_{j in S^k(i;u)} s_ij * (r_uj - b_uj) / sum|s_ij|

where b_ui = mu + b_u + b_i is the baseline estimate,
S^k(i;u) are the k most similar items to i that user u has rated,
and s_ij is the shrunk cosine similarity between items i and j.
"""

import numpy as np
import pandas as pd
from numba import njit

from cil.predictors.base import Predictor


@njit(cache=True)
def _predict_knn(sids, pids, R, mask, sim_matrix, mu, bu, bi, k_neighbors):
    n = len(sids)
    preds = np.empty(n)

    for idx in range(n):
        u = sids[idx]
        i = pids[idx]
        baseline_ui = mu + bu[u] + bi[i]

        user_rated = np.where(mask[u] > 0)[0]
        if len(user_rated) == 0:
            preds[idx] = baseline_ui
            continue

        sims = np.empty(len(user_rated))
        for j_idx in range(len(user_rated)):
            sims[j_idx] = sim_matrix[i, user_rated[j_idx]]

        if len(user_rated) > k_neighbors:
            top_k_idx = np.argsort(-np.abs(sims))[:k_neighbors]
        else:
            top_k_idx = np.arange(len(user_rated))

        num = 0.0
        denom = 0.0
        for j_idx in top_k_idx:
            j = user_rated[j_idx]
            s = sims[j_idx]
            num += s * (R[u, j] - (mu + bu[u] + bi[j]))
            denom += np.abs(s)

        preds[idx] = baseline_ui + num / denom if denom > 0 else baseline_ui

    return preds


class ItemKNNPredictor(Predictor):
    """Item-based KNN with baseline subtraction and shrunk cosine similarity."""

    def __init__(self, k: int = 50, shrinkage: float = 100.0) -> None:
        self._k = k
        self._shrinkage = shrinkage
        self._mu = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        n_users, n_items = num_scientists, num_papers
        sids = train_df["sid"].values
        pids = train_df["pid"].values
        ratings = train_df["rating"].values.astype(np.float64)

        # Compute user/item biases via alternating estimation
        self._mu = float(ratings.mean())
        self._bu = np.zeros(n_users)
        self._bi = np.zeros(n_items)
        for _ in range(10):
            for u in np.unique(sids):
                m = sids == u
                self._bu[u] = (ratings[m] - self._mu - self._bi[pids[m]]).mean()
            for i in np.unique(pids):
                m = pids == i
                self._bi[i] = (ratings[m] - self._mu - self._bu[sids[m]]).mean()

        # Build rating matrix and mask
        R = np.zeros((n_users, n_items), dtype=np.float64)
        mask = np.zeros((n_users, n_items), dtype=np.float64)
        R[sids, pids] = ratings
        mask[sids, pids] = 1.0

        # Centered ratings (subtract baselines); zero at unobserved entries.
        R_centered = np.zeros_like(R)
        for idx in range(len(sids)):
            u, i = sids[idx], pids[idx]
            R_centered[u, i] = ratings[idx] - self._mu - self._bu[u] - self._bi[i]

        # Shrunk cosine similarity
        norms = np.sqrt((R_centered**2).sum(axis=0))
        dot = R_centered.T @ R_centered
        denom = np.outer(norms, norms) + self._shrinkage
        self._sim = dot / np.maximum(denom, 1e-8)
        np.fill_diagonal(self._sim, 0.0)

        self._R = R
        self._mask = mask

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return _predict_knn(
            sids.astype(np.int64),
            pids.astype(np.int64),
            self._R,
            self._mask,
            self._sim,
            self._mu,
            self._bu,
            self._bi,
            self._k,
        )
