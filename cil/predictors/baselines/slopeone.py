"""SlopeOne Predictor for collaborative filtering.

Implements the SlopeOne model from:
    Lamire D., "Slope One Predictors for Online Rating-Based Collaborative Filtering"
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Optional
from cil.predictors.base import Predictor

from numba import njit

@njit(cache=True)
def _predict_slope_one(
    sids: np.ndarray,
    pids: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    freq: np.ndarray,
    dev: np.ndarray,
    mu: float,
) -> np.ndarray:
    n = len(sids)
    preds = np.empty(n)
    for idx in range(n):
        u = sids[idx]
        i = pids[idx]

        rated = np.where(mask[u] > 0)[0]
        if len(rated) == 0:
            preds[idx] = mu
            continue

        num, denom = 0.0, 0.0
        for j in rated:
            w = freq[i, j]
            if w > 0:
                num   += w * (R[u, j] + dev[i, j])
                denom += w

        preds[idx] = num / denom if denom > 1e-8 else mu
    return preds

class SlopeOnePredictor(Predictor):

    def __init__(self, weighted: bool = True):
        """
        Parameters
        ----------
        weighted : bool
            If True, weight by cardinality card(i,j) (Weighted SlopeOne).
            If False, use plain SlopeOne (all weights = 1).
        """
        self.weighted = weighted

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: Optional[pd.DataFrame] = None,
    ) -> None:
        self.num_scientists = num_scientists
        self.num_papers = num_papers

        n_users = num_scientists
        n_items = num_papers 
        sids = train_df["sid"]
        pids = train_df ["pid"]
        ratings = train_df["rating"]

        R    = np.zeros((n_users, n_items), dtype=np.float64)
        mask = np.zeros((n_users, n_items), dtype=np.float64)

        R[sids, pids] = ratings
        mask[sids, pids] = 1.0

        # freq[i,j] = number users who rated both items i and j
        freq = mask.T @ mask 
        # dev[i,j] = avg difference between item i and item j, across all users who rated both
        num = R.T @ mask - mask.T @ R
        dev = num / np.maximum(freq, 1)
        
        self._dev = dev 
        self._R = R
        self._mask = mask
        self._freq = freq 
        self._dev = dev 
        self._mu = ratings.mean()

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return _predict_slope_one(
            sids.astype(np.int64),
            pids.astype(np.int64),
            R=self._R,
            mask=self._mask,
            freq=self._freq,
            dev=self._dev,
            mu=self._mu
        )