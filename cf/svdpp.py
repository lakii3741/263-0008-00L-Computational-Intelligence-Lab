import numpy as np
from numba import njit
from cf.base import Predictor


@njit(cache=True, fastmath=True)
def _train_epoch(u_order, u_starts, u_lens,
                 impl_starts, impl_lens, impl_inv_sqrt,
                 flat_items, flat_ratings, flat_impl,
                 mu, bu, bi, P, Q, Y, lr, reg, k):
    n_active = len(u_order)
    for ui in range(n_active):
        idx = u_order[ui]
        u = idx  # u_order indexes into per-user arrays where user-id == idx
        start = u_starts[ui]
        length = u_lens[ui]
        impl_s = impl_starts[ui]
        impl_l = impl_lens[ui]
        sq = impl_inv_sqrt[ui]

        sum_y = np.zeros(k)
        for j in range(impl_l):
            it = flat_impl[impl_s + j]
            for f in range(k):
                sum_y[f] += Y[it, f]

        y_grad = np.zeros(k)
        for ri in range(length):
            i = flat_items[start + ri]
            r = flat_ratings[start + ri]

            pred = mu + bu[u] + bi[i]
            for f in range(k):
                pred += Q[i, f] * (P[u, f] + sq * sum_y[f])
            err = r - pred

            qi = Q[i].copy()
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            for f in range(k):
                p_hat_f = P[u, f] + sq * sum_y[f]
                Q[i, f] += lr * (err * p_hat_f - reg * Q[i, f])
                P[u, f] += lr * (err * qi[f] - reg * P[u, f])
                y_grad[f] += err * qi[f]

        for j in range(impl_l):
            it = flat_impl[impl_s + j]
            for f in range(k):
                Y[it, f] += lr * (sq * y_grad[f] - reg * Y[it, f])


class SVDpp(Predictor):
    def __init__(self, k=20, n_epochs=20, lr=0.007, reg=0.02, seed=42,
                 include_wishlist=True, init_scale=0.1, lr_decay=1.0):
        self.k = k
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.include_wishlist = include_wishlist
        self.init_scale = init_scale
        self.lr_decay = lr_decay

    def fit(self, train_df, n_users, n_items, wishlist=None):
        rng = np.random.default_rng(self.seed)
        sids = train_df["sid"].to_numpy(np.int64)
        pids = train_df["pid"].to_numpy(np.int64)
        ratings = train_df["rating"].to_numpy(np.float64)
        self.mu = float(ratings.mean())

        # group ratings by user
        order = np.argsort(sids, kind="stable")
        sids_s = sids[order]
        pids_s = pids[order]
        ratings_s = ratings[order]
        # unique users present in training
        active_users, first_idx = np.unique(sids_s, return_index=True)
        u_starts_full = np.append(first_idx, len(sids_s)).astype(np.int64)
        u_starts = u_starts_full[:-1]
        u_lens = np.diff(u_starts_full).astype(np.int64)
        flat_items = pids_s
        flat_ratings = ratings_s

        # implicit set per active user: rated ∪ (wishlist if include)
        if self.include_wishlist and wishlist is not None and len(wishlist) > 0:
            w = wishlist.copy()
            w_by_u = {int(u): grp["pid"].to_numpy(np.int64) for u, grp in w.groupby("sid")}
        else:
            w_by_u = {}

        impl_lists = []
        impl_lens = np.zeros(len(active_users), dtype=np.int64)
        impl_inv_sqrt = np.zeros(len(active_users), dtype=np.float64)
        for ui, u in enumerate(active_users):
            rated = flat_items[u_starts[ui]:u_starts[ui] + u_lens[ui]]
            wish = w_by_u.get(int(u), np.empty(0, dtype=np.int64))
            impl = np.unique(np.concatenate([rated, wish])) if len(wish) else rated
            impl_lists.append(impl)
            impl_lens[ui] = len(impl)
            impl_inv_sqrt[ui] = 1.0 / np.sqrt(len(impl))

        impl_starts = np.zeros(len(active_users), dtype=np.int64)
        offset = 0
        for ui in range(len(active_users)):
            impl_starts[ui] = offset
            offset += impl_lens[ui]
        flat_impl = np.concatenate(impl_lists).astype(np.int64)

        # init parameters
        bu = np.zeros(n_users, dtype=np.float64)
        bi = np.zeros(n_items, dtype=np.float64)
        P = rng.normal(0, self.init_scale, (n_users, self.k)).astype(np.float64)
        Q = rng.normal(0, self.init_scale, (n_items, self.k)).astype(np.float64)
        Y = rng.normal(0, self.init_scale, (n_items, self.k)).astype(np.float64)

        cur_lr = self.lr
        for epoch in range(self.n_epochs):
            order_ep = rng.permutation(len(active_users)).astype(np.int64)
            # shuffle within each active user's ratings
            for ui in range(len(active_users)):
                s = u_starts[ui]
                l = u_lens[ui]
                if l > 1:
                    p = rng.permutation(l)
                    flat_items[s:s + l] = flat_items[s:s + l][p]
                    flat_ratings[s:s + l] = flat_ratings[s:s + l][p]
            # we pass u_order in the order of active_users[order_ep]; the kernel reads
            # u directly from active_users[order_ep[ui]], so we need that mapping
            u_order_users = active_users[order_ep].astype(np.int64)
            _train_epoch(u_order_users,
                         u_starts[order_ep], u_lens[order_ep],
                         impl_starts[order_ep], impl_lens[order_ep], impl_inv_sqrt[order_ep],
                         flat_items, flat_ratings, flat_impl,
                         self.mu, bu, bi, P, Q, Y, cur_lr, self.reg, self.k)
            cur_lr *= self.lr_decay

        # cache user vectors
        P_hat = P.copy()
        for ui, u in enumerate(active_users):
            impl = flat_impl[impl_starts[ui]:impl_starts[ui] + impl_lens[ui]]
            P_hat[u] += impl_inv_sqrt[ui] * Y[impl].sum(axis=0)

        # cold users: P_hat from wishlist Y-sum if available
        if w_by_u:
            active_set = set(active_users.tolist())
            for u, wish in w_by_u.items():
                if u in active_set or not (0 <= u < n_users) or len(wish) == 0:
                    continue
                P_hat[u] = Y[wish].sum(axis=0) / np.sqrt(len(wish))

        self.user_vec = P_hat
        self.Q = Q
        self.bu = bu
        self.bi = bi
        return self

    def predict(self, sids, pids):
        return (self.mu + self.bu[sids] + self.bi[pids]
                + np.einsum("ij,ij->i", self.user_vec[sids], self.Q[pids]))
