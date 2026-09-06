"""Iterative SVD predictor for collaborative filtering.

Iteratively imputes missing values using truncated SVD reconstruction.
Starting from an initial fill (global mean of observed ratings), each
iteration computes a rank-k SVD approximation and uses it to re-impute
the missing entries while keeping observed entries fixed.
"""

import numpy as np
import pandas as pd

from cil.data_loader import DataLoader
from cil.predictors.base import Predictor


class IterativeSVDPredictor(Predictor):
    """Iterative SVD baseline.

    Algorithm:
        1. Fill missing entries with the global mean of observed ratings.
        2. Compute rank-k SVD approximation.
        3. Replace missing entries with the low-rank reconstruction.
        4. Repeat steps 2-3 for ``n_iters`` iterations.

    The final low-rank reconstruction is used for all predictions.

    Args:
        k: Number of singular values/vectors to retain.
        n_iters: Number of impute-reconstruct iterations.
    """

    def __init__(self, k: int = 10, n_iters: int = 20) -> None:
        self._k = k
        self._n_iters = n_iters

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        mat = DataLoader.to_matrix(train_df, num_scientists, num_papers)
        observed = ~np.isnan(mat)
        global_mean = np.nanmean(mat)

        # Initial imputation: fill missing entries with global mean
        mat = np.where(observed, mat, global_mean)

        for _ in range(self._n_iters):
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            recon = (U[:, : self._k] * S[: self._k]) @ Vh[: self._k]
            # Keep observed entries, replace missing with reconstruction
            mat[~observed] = recon[~observed]

        self._recon = recon

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return self._recon[sids, pids]
