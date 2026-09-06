"""SVD predictor for collaborative filtering.

Implements the SVD model following Surprise's implementation:
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD

References:
    - Koren, Y. "Matrix Factorization Techniques for Recommender Systems"
      https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
    - Funk, S. "Netflix Update: Try This at Home." 2006.
      https://sifter.org/simon/journal/20061211.html
"""
import numpy as np
import pandas as pd
from numba import njit
from typing import Optional, List

from cil.predictors.base import Predictor


@njit(cache=True)
def _train_svd_epoch(
    user_starts: np.ndarray,
    user_lengths: np.ndarray,
    flat_items: np.ndarray,
    flat_ratings: np.ndarray,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    user_ids: np.ndarray,
    lr_bu: float,
    lr_bi: float,
    lr_pu: float,
    lr_qi: float,
    reg_bu: float,
    reg_bi: float,
    reg_pu: float,
    reg_qi: float,
    n_factors: int,
    biased: bool,
) -> None:
    """One epoch of SGD for SVD."""
    n_users = len(user_starts)

    for ui in range(n_users):
        u = user_ids[ui]
        start = user_starts[ui]
        length = user_lengths[ui]

        for ri in range(length):
            idx = start + ri
            i = flat_items[idx]
            r = flat_ratings[idx]

            # predict
            pred = mu
            if biased:
                pred += bu[u] + bi[i]
            for f in range(n_factors):
                pred += P[u, f] * Q[i, f]

            err = r - pred

            # update biases
            if biased:
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

            # update factors
            for f in range(n_factors):
                p_uf = P[u, f]
                q_if = Q[i, f]
                P[u, f] += lr_pu * (err * q_if - reg_pu * p_uf)
                Q[i, f] += lr_qi * (err * p_uf - reg_qi * q_if)


class BiasedSVDPredictor(Predictor):
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        biased: bool = True,
        init_mean: float = 0.0,
        init_std: float = 0.1,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        lr_bu: Optional[float] = None,
        lr_bi: Optional[float] = None,
        lr_pu: Optional[float] = None,
        lr_qi: Optional[float] = None,
        reg_bu: Optional[float] = None,
        reg_bi: Optional[float] = None,
        reg_pu: Optional[float] = None,
        reg_qi: Optional[float] = None,
        lr_decay: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        n_factors : int, optional
            Number of latent factors. Default is 100.
        n_epochs : int, optional
            Number of SGD iterations. Default is 20.
        biased : bool, optional
            Whether to use bias terms (b_u and b_i). Default is True.
        init_mean : float, optional
            Mean for the normal distribution used in factor initialization. 
            Default is 0.
        init_std : float, optional
            Standard deviation for the normal distribution used in factor 
            initialization. Default is 0.1.
        lr_all : float, optional
            Learning rate for all parameters. Default is 0.005.
        reg_all : float, optional
            Regularization term for all parameters. Default is 0.02.
        lr_bu : float, optional
            Learning rate for user biases b_u. Overrides `lr_all`.
        lr_bi : float, optional
            Learning rate for item biases b_i. Overrides `lr_all`.
        lr_pu : float, optional
            Learning rate for user latent factors p_u. Overrides `lr_all`.
        lr_qi : float, optional
            Learning rate for item latent factors q_i. Overrides `lr_all`.
        reg_bu : float, optional
            Regularization for user biases b_u. Overrides `reg_all`.
        reg_bi : float, optional
            Regularization for item biases b_i. Overrides `reg_all`.
        reg_pu : float, optional
            Regularization for user latent factors p_u. Overrides `reg_all`.
        reg_qi : float, optional
            Regularization for item latent factors q_i. Overrides `reg_all`.
        lr_decay : float, optional
            Multiplicative learning rate decay applied after each epoch. 
            A value of 1.0 means no decay. Default is 1.0.
        seed : int, optional
        """
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._biased = biased
        self._init_mean = init_mean
        self._init_std = init_std
        self._lr_decay = lr_decay
        self._seed = seed

        # per-parameter lr and reg, falling back to lr_all / reg_all
        self._lr_bu = lr_bu if lr_bu is not None else lr_all
        self._lr_bi = lr_bi if lr_bi is not None else lr_all
        self._lr_pu = lr_pu if lr_pu is not None else lr_all
        self._lr_qi = lr_qi if lr_qi is not None else lr_all
        self._reg_bu = reg_bu if reg_bu is not None else reg_all
        self._reg_bi = reg_bi if reg_bi is not None else reg_all
        self._reg_pu = reg_pu if reg_pu is not None else reg_all
        self._reg_qi = reg_qi if reg_qi is not None else reg_all

        self._mu: float = 0.0
        self._bu: Optional[np.ndarray] = None
        self._bi: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: Optional[pd.DataFrame] = None,
    ) -> None:
        rng = np.random.default_rng(self._seed)
        n_users = num_scientists
        n_items = num_papers
        n_factors = self._n_factors

        # initialization
        self._mu = float(train_df["rating"].mean())
        self._bu = np.zeros(n_users, dtype=np.float64)
        self._bi = np.zeros(n_items, dtype=np.float64)
        self._P = rng.normal(self._init_mean, self._init_std, (n_users, n_factors)).astype(np.float64)
        self._Q = rng.normal(self._init_mean, self._init_std, (n_items, n_factors)).astype(np.float64)

        grouped = train_df.groupby("sid")
        active_users: np.ndarray = np.array(list(grouped.groups.keys()), dtype=np.int64)
        n_active = len(active_users)

        # flattening
        user_starts = np.zeros(n_active, dtype=np.int64)
        user_lengths = np.zeros(n_active, dtype=np.int64)
        flat_items_list: List[np.ndarray] = []
        flat_ratings_list: List[np.ndarray] = []

        offset = 0
        for idx, u in enumerate(active_users):
            grp = grouped.get_group(u)
            u_items = grp["pid"].values.astype(np.int64)
            u_ratings = grp["rating"].values.astype(np.float64)
            user_starts[idx] = offset
            user_lengths[idx] = len(u_items)
            flat_items_list.append(u_items)
            flat_ratings_list.append(u_ratings)
            offset += len(u_items)

        flat_items = np.concatenate(flat_items_list)
        flat_ratings = np.concatenate(flat_ratings_list)

        current_lr_bu = self._lr_bu
        current_lr_bi = self._lr_bi
        current_lr_pu = self._lr_pu
        current_lr_qi = self._lr_qi

        for _ in range(self._n_epochs):
            # shuffling to prevent ordering bias
            perm = rng.permutation(n_active)
            _train_svd_epoch(
                user_starts[perm],
                user_lengths[perm],
                flat_items,
                flat_ratings,
                self._mu,
                self._bu,
                self._bi,
                self._P,
                self._Q,
                active_users[perm],
                current_lr_bu,
                current_lr_bi,
                current_lr_pu,
                current_lr_qi,
                self._reg_bu,
                self._reg_bi,
                self._reg_pu,
                self._reg_qi,
                n_factors,
                self._biased,
            )
            # updating learning rate
            current_lr_bu *= self._lr_decay
            current_lr_bi *= self._lr_decay
            current_lr_pu *= self._lr_decay
            current_lr_qi *= self._lr_decay

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        preds = np.sum(self._P[sids] * self._Q[pids], axis=1) + self._mu
        if self._biased:
            preds += self._bu[sids] + self._bi[pids]
        return preds