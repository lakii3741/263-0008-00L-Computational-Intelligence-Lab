import numpy as np
from cf.base import Predictor, Centered


class GlobalMean(Predictor):
    def fit(self, train_df, n_users, n_items, wishlist=None):
        self.mu = float(train_df["rating"].mean())
        return self

    def predict(self, sids, pids):
        return np.full(len(sids), self.mu)


class BiasModel(Predictor):
    """mu + bu + bi with shrinkage."""

    def __init__(self, shrink=25.0):
        self.shrink = shrink

    def fit(self, train_df, n_users, n_items, wishlist=None):
        self.c = Centered(user_bias=True, item_bias=True, shrink=self.shrink)
        self.c.fit(train_df, n_users, n_items)
        self.n_users = n_users
        self.n_items = n_items
        return self

    def predict(self, sids, pids):
        return self.c.baseline(sids, pids)


class SVDImpute(Predictor):
    """Impute missing with bias model, then take rank-k SVD."""

    def __init__(self, k=10, shrink=25.0):
        self.k = k
        self.shrink = shrink

    def fit(self, train_df, n_users, n_items, wishlist=None):
        self.c = Centered(user_bias=True, item_bias=True, shrink=self.shrink)
        self.c.fit(train_df, n_users, n_items)
        sids = train_df["sid"].to_numpy()
        pids = train_df["pid"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float64)
        base_train = self.c.baseline(sids, pids)
        resid = r - base_train

        M = np.zeros((n_users, n_items), dtype=np.float64)
        M[sids, pids] = resid

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        k = min(self.k, len(S))
        self.U = U[:, :k] * S[:k]
        self.Vt = Vt[:k]
        return self

    def predict(self, sids, pids):
        base = self.c.baseline(sids, pids)
        delta = np.einsum("ij,ji->i", self.U[sids], self.Vt[:, pids])
        return base + delta


class IterSVD(Predictor):
    """Iteratively re-SVD: impute missing with current low-rank reconstruction."""

    def __init__(self, k=10, n_iters=15, shrink=25.0):
        self.k = k
        self.n_iters = n_iters
        self.shrink = shrink

    def fit(self, train_df, n_users, n_items, wishlist=None):
        self.c = Centered(user_bias=True, item_bias=True, shrink=self.shrink)
        self.c.fit(train_df, n_users, n_items)
        sids = train_df["sid"].to_numpy()
        pids = train_df["pid"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float64)
        base_train = self.c.baseline(sids, pids)
        resid = r - base_train

        M = np.zeros((n_users, n_items), dtype=np.float64)
        M[sids, pids] = resid
        mask = np.zeros((n_users, n_items), dtype=bool)
        mask[sids, pids] = True

        cur = M.copy()
        for _ in range(self.n_iters):
            U, S, Vt = np.linalg.svd(cur, full_matrices=False)
            k = min(self.k, len(S))
            recon = (U[:, :k] * S[:k]) @ Vt[:k]
            cur = np.where(mask, M, recon)

        U, S, Vt = np.linalg.svd(cur, full_matrices=False)
        k = min(self.k, len(S))
        self.U = U[:, :k] * S[:k]
        self.Vt = Vt[:k]
        return self

    def predict(self, sids, pids):
        base = self.c.baseline(sids, pids)
        delta = np.einsum("ij,ji->i", self.U[sids], self.Vt[:, pids])
        return base + delta
