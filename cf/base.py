import numpy as np


class Predictor:
    def fit(self, train_df, n_users, n_items, wishlist=None):
        raise NotImplementedError

    def predict(self, sids, pids):
        raise NotImplementedError


class Centered:
    """Mean-centering helper. Subtract the global mean before fit, add it back at predict.
    Optional per-user and per-item bias terms estimated by Welford-style shrinkage."""

    def __init__(self, user_bias=False, item_bias=False, shrink=25.0):
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.shrink = shrink
        self.mu = 0.0
        self.bu = None
        self.bi = None

    def fit(self, df, n_users, n_items):
        r = df["rating"].to_numpy(dtype=np.float64)
        self.mu = float(r.mean())
        if self.item_bias:
            sums = np.zeros(n_items)
            cnts = np.zeros(n_items)
            np.add.at(sums, df["pid"].to_numpy(), r - self.mu)
            np.add.at(cnts, df["pid"].to_numpy(), 1.0)
            self.bi = sums / (cnts + self.shrink)
        if self.user_bias:
            sums = np.zeros(n_users)
            cnts = np.zeros(n_users)
            adj = r - self.mu
            if self.bi is not None:
                adj = adj - self.bi[df["pid"].to_numpy()]
            np.add.at(sums, df["sid"].to_numpy(), adj)
            np.add.at(cnts, df["sid"].to_numpy(), 1.0)
            self.bu = sums / (cnts + self.shrink)
        return self

    def baseline(self, sids, pids):
        out = np.full(len(sids), self.mu)
        if self.bi is not None:
            out = out + self.bi[pids]
        if self.bu is not None:
            out = out + self.bu[sids]
        return out

    def transform_train(self, df):
        r = df["rating"].to_numpy(dtype=np.float64)
        base = np.full(len(df), self.mu)
        if self.bi is not None:
            base = base + self.bi[df["pid"].to_numpy()]
        if self.bu is not None:
            base = base + self.bu[df["sid"].to_numpy()]
        out = df.copy()
        out["rating"] = r - base
        return out
