"""KNNBasic collaborative filtering.
Adjusts the KNNBasic model from:
    https://www.nature.com/articles/s41598-025-15096-4#Sec39

Item-based: r_ui = sum_{j in S^k(i;u)} s_ij * r_uj / sum s_ij 
User-based: r_ui = sum_{v in S^k(u;i)} s_uv * r_vi / sum s_uv

where s_ij / s_uv is the cosine similarity (raw ratings, no centering).
"""

import numpy as np
import pandas as pd
from numba import njit

from cil.predictors.base import Predictor
from cil.predictors.baselines.similarities import compute_cosine_sim

@njit(cache=True)
def _predict_item_based(
    sids: np.ndarray, 
    pids: np.ndarray, 
    R: np.ndarray, 
    mask: np.ndarray, 
    sim: np.ndarray, 
    mu: float, 
    k: int)-> np.ndarray:
    n = len(sids)
    preds = np.empty(n)
    for idx in range(n):
        u = sids[idx]
        i = pids[idx]

        # item IDs that user u has rated
        user_rated = np.where(mask[u] > 0)[0]
        if len(user_rated) == 0:
            preds[idx] = mu
            continue
        
        # similarities between item i and items rated by user u
        sims = np.empty(len(user_rated))
        for j_idx in range(len(user_rated)):
            sims[j_idx] = sim[i, user_rated[j_idx]]
        
        # extract top-k similar items
        if len(user_rated) > k:
            top_k_idx = np.argsort(-sims)[:k]
        else:
            top_k_idx = np.arange(len(user_rated))
        
        # compute weighted average rating for user u on item i
        num = 0.0
        denom = 0.0
        for j_idx in top_k_idx:
            j = user_rated[j_idx]
            s = sims[j_idx]
            if s > 0:
                num   += s * R[u, j]
                denom += s

        preds[idx] = num / denom if denom > 1e-8 else mu
    return preds


class KNNBasicPredictor(Predictor):

    def __init__(self, k: int = 50, user_based: bool = False) -> None:
        self._k          = k
        self._user_based = user_based
        self._mu         = 0.0

        if user_based:
            raise NotImplementedError("User-based KNN is not implemented yet.")

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        n_users, n_items = num_scientists, num_papers
        sids    = train_df["sid"].values
        pids    = train_df["pid"].values
        ratings = train_df["rating"].values.astype(np.float64)

        # construct the user-item rating matrix R and the mask matrix
        R = np.zeros((n_users, n_items), dtype=np.float64)
        print(R.shape)
        mask = np.zeros((n_users, n_items), dtype=np.float64)

        R[sids, pids]    = ratings
        mask[sids, pids] = 1.0

        # 2. Compute the item-item similarity matrix
        ur = {}
        for sid, group in train_df.groupby("sid"):
            ur[sid] = list(zip(group["pid"], group["rating"]))
        min_support = 1 
        sim = compute_cosine_sim(n_items, ur, min_support)
        np.fill_diagonal(sim, 0.0)

        self._R    = R
        self._mask = mask
        self._sim  = sim
        self._mu = ratings.mean()

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        fn = _predict_item_based
        return fn(
            sids.astype(np.int64),
            pids.astype(np.int64),
            self._R,
            self._mask,
            self._sim,
            self._mu,
            self._k,
        )