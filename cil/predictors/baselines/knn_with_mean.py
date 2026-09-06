"""KNNWithMean collaborative filtering.

Adjusts the KNNWithMean model from:
    https://www.nature.com/articles/s41598-025-15096-4#Sec39

Implementation based on Surprise library.    

Prediction: r̂_ui = μ_i + *
            ( ∑_{j ∈ N_k(i;u)} sim(i, j) * (r_uj - μ_j) )  /
            ( ∑_{j ∈ N_k(i;u)} sim(i, j) )

where s_ij / s_uv is the cosine similarity (raw ratings).
"""

import numpy as np
import pandas as pd
from numba import njit

from cil.predictors.base import Predictor
from cil.predictors.baselines.similarities import compute_cosine_sim

from numba import njit

@njit(cache=True)
def _compute_cosine_sim(
    n_users: int
):
    sim = np.zeros((n_users, n_users), dtype=np.float64)
    prods = np.zeros((n_users, n_users), dtype=np.float64)
    sqi = np.zeros((n_users, n_users), dtype=np.float64)
    sqj = np.zeros((n_users, n_users), dtype=np.float64)
    freq = np.zeros((n_users, n_users), dtype=np.float64)


@njit(cache=True)
def _predict_item_with_mean(
    sids: np.ndarray, 
    pids: np.ndarray, 
    R: np.ndarray, 
    mask: np.ndarray, 
    sim: np.ndarray, 
    k: int,
    item_means: np.ndarray,
    )-> np.ndarray:

    n = len(sids)
    preds = np.empty(n)
    for idx in range(n):
        u = sids[idx]
        i = pids[idx]
        
        user_rated = np.where(mask[u] > 0)[0]
        if len(user_rated) == 0:
            preds[idx] = item_means[i]
            continue

        sims = np.empty(len(user_rated))
        for j_idx in range(len(user_rated)):
            sims[j_idx] = sim[i, user_rated[j_idx]]

        if len(user_rated) > k:
            top_k_idx = np.argsort(-sims)[:k]
        else:
            top_k_idx = np.arange(len(user_rated))

        # compute formula
        num, denom = 0.0, 0.0
        for j_idx in top_k_idx:
            j = user_rated[j_idx]
            s = sims[j_idx]
            if s > 0:
                num   += s * (R[u, j] - item_means[j]) 
                denom += s
        preds[idx] = item_means[i] + (num / denom  if denom > 1e-8 else 0.0)
    return preds


class KNNWithMeansPredictor(Predictor):

    def __init__(self, k: int = 50) -> None:
        self._k = k
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

        self._mu = float(ratings.mean())

        # construct the user-item rating matrix R and the mask matrix
        R    = np.zeros((n_users, n_items), dtype=np.float64)
        mask = np.zeros((n_users, n_items), dtype=np.float64)

        R[sids, pids] = ratings
        mask[sids, pids] = 1.0

        # compute mean rating for each item
        item_counts = mask.sum(axis=0).clip(1)
        item_means  = (R * mask).sum(axis=0) / item_counts

        # compute the item-item similarity matrix
        ur = {}
        for sid, group in train_df.groupby("sid"):
            ur[sid] = list(zip(group["pid"], group["rating"]))
        min_support = 1 
        sim = compute_cosine_sim(n_items, ur, min_support)
        np.fill_diagonal(sim, 0.0)
 
        self._R = R
        self._mask = mask
        self._item_means = item_means
        self._sim = sim

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return _predict_item_with_mean(
            sids=sids.astype(np.int64),
            pids=pids.astype(np.int64),
            R=self._R,
            mask=self._mask,
            sim=self._sim,
            k=self._k,
            item_means=self._item_means,
        )