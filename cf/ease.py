import numpy as np
from cf.base import Predictor, Centered


class EASE(Predictor):
    """Linear item-item autoencoder. Minimises ||X - XB||^2 + lam*||B||^2 s.t. diag(B)=0.
    Closed form: B = (G + lam I)^-1 G with diag re-zeroed via Lagrangian trick."""

    def __init__(self, lam=300.0, use_bias=True, shrink=25.0):
        self.lam = lam
        self.use_bias = use_bias
        self.shrink = shrink

    def fit(self, train_df, n_users, n_items, wishlist=None):
        sids = train_df["sid"].to_numpy()
        pids = train_df["pid"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float64)

        if self.use_bias:
            self.c = Centered(user_bias=True, item_bias=True, shrink=self.shrink)
            self.c.fit(train_df, n_users, n_items)
            r = r - self.c.baseline(sids, pids)
        else:
            self.c = None

        X = np.zeros((n_users, n_items), dtype=np.float64)
        X[sids, pids] = r

        G = X.T @ X
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.lam
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag_idx] = 0.0

        self.S = X @ B  # n_users x n_items predicted residuals
        return self

    def predict(self, sids, pids):
        out = self.S[sids, pids]
        if self.c is not None:
            out = out + self.c.baseline(sids, pids)
        return out
