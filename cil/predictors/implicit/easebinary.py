"""EASE on the binary (rating ∪ wishlist) matrix.

Builds X[u,i] = 1 if (u,i) is rated (above threshold) OR wishlisted, else 0. Solves the closed-form
EASE problem on this binary signal. Predicts a continuous score per (u,i) and maps
it to the rating scale via a least-squares affine fit on the training ratings."""

import numpy as np
from cil.predictors.base import Predictor


class EASEBinaryPredictor(Predictor):
    def __init__(self, lam=200.0, use_wishlist=True):
        self.lam = lam
        self.use_wishlist = use_wishlist
        self.rating_threshold = 10.0

    def fit(self, train_df, n_users, n_items, wishlist=None):
        # Filter by rating threshold
        sids = train_df["sid"].to_numpy()
        pids = train_df["pid"].to_numpy()

        X = np.zeros((n_users, n_items), dtype=np.float64)
        X[sids, pids] = (train_df["rating"].to_numpy() >= self.rating_threshold).astype(np.float64)
        
        if self.use_wishlist and wishlist is not None and len(wishlist) > 0:
            X[wishlist["sid"].to_numpy(), wishlist["pid"].to_numpy()] = 1.0

        G = X.T @ X
        d = np.diag_indices_from(G)
        G[d] += self.lam
        Pi = np.linalg.inv(G)
        B = Pi / (-np.diag(Pi))
        B[d] = 0.0
        S = X @ B  # n_users x n_items scores

        scores_train = S[sids, pids]
        r = train_df["rating"].to_numpy(dtype=np.float64)
        A = np.stack([scores_train, np.ones_like(scores_train)], axis=1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        self.S = S
        self.a = float(coef[0])
        self.b = float(coef[1])
        return self

    def predict(self, sids, pids):
        s = self.S[sids, pids]
        return self.a * s + self.b
