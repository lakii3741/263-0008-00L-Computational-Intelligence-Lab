import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _predict(sids, pids, R, mask, sim, mu, bu, bi, k_n):
    n = len(sids)
    out = np.empty(n)
    for ix in range(n):
        u = sids[ix]
        i = pids[ix]
        base_ui = mu + bu[u] + bi[i]
        rated = np.where(mask[u] > 0)[0]
        if len(rated) == 0:
            out[ix] = base_ui
            continue
        sims = np.empty(len(rated))
        for j in range(len(rated)):
            sims[j] = sim[i, rated[j]]
        if len(rated) > k_n:
            top = np.argsort(-np.abs(sims))[:k_n]
        else:
            top = np.arange(len(rated))
        num = 0.0
        den = 0.0
        for ji in top:
            j = rated[ji]
            s = sims[ji]
            num += s * (R[u, j] - (mu + bu[u] + bi[j]))
            den += np.abs(s)
        out[ix] = base_ui + (num / den if den > 0 else 0.0)
    return out


class ItemKNN:
    """Item-based KNN with shrunk cosine on baseline-centered ratings.
    Bias estimation: 10 alternating passes of unregularized per-user/item mean."""

    def __init__(self, k=50, shrinkage=100.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, train_df, n_users, n_items, wishlist=None):
        sids = train_df["sid"].to_numpy()
        pids = train_df["pid"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float64)

        self.mu = float(r.mean())
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        # alternating bias estimation
        u_grp = {u: np.where(sids == u)[0] for u in np.unique(sids)}
        i_grp = {i: np.where(pids == i)[0] for i in np.unique(pids)}
        for _ in range(10):
            for u, ix in u_grp.items():
                bu[u] = np.mean(r[ix] - self.mu - bi[pids[ix]])
            for i, ix in i_grp.items():
                bi[i] = np.mean(r[ix] - self.mu - bu[sids[ix]])

        R = np.zeros((n_users, n_items), dtype=np.float64)
        mask = np.zeros((n_users, n_items), dtype=np.float64)
        R[sids, pids] = r
        mask[sids, pids] = 1.0

        Rc = np.zeros_like(R)
        Rc[sids, pids] = r - self.mu - bu[sids] - bi[pids]
        Rc *= mask
        norms = np.sqrt((Rc ** 2).sum(axis=0))
        dot = Rc.T @ Rc
        denom = np.outer(norms, norms) + self.shrinkage
        sim = dot / np.maximum(denom, 1e-9)
        np.fill_diagonal(sim, 0.0)

        self.R = R
        self.mask = mask
        self.sim = sim
        self.bu = bu
        self.bi = bi
        return self

    def predict(self, sids, pids):
        return _predict(sids.astype(np.int64), pids.astype(np.int64),
                        self.R, self.mask, self.sim, self.mu, self.bu, self.bi, self.k)
