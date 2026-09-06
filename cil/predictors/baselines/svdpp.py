"""Numba-accelerated SVD++ predictor for collaborative filtering.

Implements the SVD++ model from:
    Koren, Y. "Factorization Meets the Neighborhood: a Multifaceted
    Collaborative Filtering Model." KDD 2008.

Prediction: r_ui = mu + b_u + b_i + q_i^T * (p_u + |I_u|^{-1/2} * sum_{j in I_u} y_j)
where I_u is the set of items user u has rated.
"""

import numpy as np
import pandas as pd
from numba import njit

from cil.predictors.base import Predictor


@njit(cache=True)
def _train_epoch(
    user_starts,
    user_lengths,
    flat_items,
    flat_ratings,
    mu,
    bu,
    bi,
    P,
    Q,
    Y,
    user_ids,
    impl_starts,
    impl_lengths,
    flat_impl_items,
    impl_inv_sqrt,
    lr,
    reg,
    k,
):
    n_users = len(user_starts)

    for ui in range(n_users):
        u = user_ids[ui]
        start = user_starts[ui]
        length = user_lengths[ui]
        impl_start = impl_starts[ui]
        impl_length = impl_lengths[ui]
        sq = impl_inv_sqrt[ui]

        sum_y = np.zeros(k)
        for j in range(impl_length):
            item_j = flat_impl_items[impl_start + j]
            for f in range(k):
                sum_y[f] += Y[item_j, f]

        y_grad = np.zeros(k)

        for ri in range(length):
            idx = start + ri
            i = flat_items[idx]
            r = flat_ratings[idx]

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

        for j in range(impl_length):
            item_j = flat_impl_items[impl_start + j]
            for f in range(k):
                Y[item_j, f] += lr * (sq * y_grad[f] - reg * Y[item_j, f])


class SVDppPredictor(Predictor):
    """SVD++ with implicit feedback, trained via SGD.

    Args:
        k: Latent factor dimensionality.
        lr: Learning rate for SGD.
        reg: L2 regularization strength.
        n_epochs: Number of SGD passes over the training data.
        seed: Random seed for initialization and shuffling.
        lr_decay: Multiplicative learning rate decay per epoch.
    """

    def __init__(
        self,
        k: int = 20,
        lr: float = 0.007,
        reg: float = 0.02,
        n_epochs: int = 20,
        seed: int = 42,
        lr_decay: float = 1.0,
    ) -> None:
        self._k = k
        self._lr = lr
        self._reg = reg
        self._n_epochs = n_epochs
        self._seed = seed
        self._lr_decay = lr_decay
        self._mu: float = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        rng = np.random.default_rng(self._seed)
        n_users, n_items = num_scientists, num_papers
        k = self._k

        self._mu = float(train_df["rating"].mean())
        self._bu = np.zeros(n_users, dtype=np.float64)
        self._bi = np.zeros(n_items, dtype=np.float64)
        P = rng.normal(0, 0.1, (n_users, k)).astype(np.float64)
        Q = rng.normal(0, 0.1, (n_items, k)).astype(np.float64)
        Y = rng.normal(0, 0.1, (n_items, k)).astype(np.float64)

        # Group ratings by user
        user_items = {}
        user_ratings = {}
        for u, grp in train_df.groupby("sid"):
            user_items[u] = grp["pid"].values.astype(np.int64)
            user_ratings[u] = grp["rating"].values.astype(np.float64)

        user_implicit = {u: user_items[u].copy() for u in user_items}

        # Flatten for numba
        active_users = list(user_items.keys())
        n_active = len(active_users)
        user_ids = np.array(active_users, dtype=np.int64)

        flat_items_list, flat_ratings_list = [], []
        user_starts = np.zeros(n_active, dtype=np.int64)
        user_lengths = np.zeros(n_active, dtype=np.int64)
        offset = 0
        for idx, u in enumerate(active_users):
            user_starts[idx] = offset
            user_lengths[idx] = len(user_items[u])
            flat_items_list.append(user_items[u])
            flat_ratings_list.append(user_ratings[u])
            offset += len(user_items[u])
        flat_items = np.concatenate(flat_items_list).astype(np.int64)
        flat_ratings = np.concatenate(flat_ratings_list).astype(np.float64)

        flat_impl_list = []
        impl_starts = np.zeros(n_active, dtype=np.int64)
        impl_lengths = np.zeros(n_active, dtype=np.int64)
        impl_inv_sqrt = np.zeros(n_active, dtype=np.float64)
        offset = 0
        for idx, u in enumerate(active_users):
            impl = user_implicit[u]
            impl_starts[idx] = offset
            impl_lengths[idx] = len(impl)
            impl_inv_sqrt[idx] = 1.0 / np.sqrt(len(impl))
            flat_impl_list.append(impl)
            offset += len(impl)
        flat_impl_items = np.concatenate(flat_impl_list).astype(np.int64)

        # Train
        current_lr = self._lr
        for epoch in range(self._n_epochs):
            perm = rng.permutation(n_active)

            # Shuffle each user's ratings in-place (matches old per-user permutation)
            for idx in range(n_active):
                s = user_starts[idx]
                length = user_lengths[idx]
                if length > 1:
                    sub_perm = rng.permutation(length)
                    flat_items[s:s + length] = flat_items[s:s + length][sub_perm]
                    flat_ratings[s:s + length] = flat_ratings[s:s + length][sub_perm]

            _train_epoch(
                user_starts[perm],
                user_lengths[perm],
                flat_items,
                flat_ratings,
                self._mu,
                self._bu,
                self._bi,
                P,
                Q,
                Y,
                user_ids[perm],
                impl_starts[perm],
                impl_lengths[perm],
                flat_impl_items,
                impl_inv_sqrt[perm],
                current_lr,
                self._reg,
                k,
            )
            current_lr *= self._lr_decay
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self._n_epochs} (lr={current_lr:.5f})")

        self._Q = Q
        self._P_hat = P.copy()
        for idx, u in enumerate(active_users):
            impl = user_implicit[u]
            self._P_hat[u] += impl_inv_sqrt[idx] * Y[impl].sum(axis=0)

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return (
            self._mu
            + self._bu[sids]
            + self._bi[pids]
            + np.sum(self._Q[pids] * self._P_hat[sids], axis=1)
        )
