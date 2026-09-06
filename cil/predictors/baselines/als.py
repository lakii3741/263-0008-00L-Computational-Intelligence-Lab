"""Alternating Least Squares predictor for collaborative filtering.

Standard regularized matrix factorization via ALS. Decomposes the rating
matrix R ~ U V^T where U (users x k) and V (items x k) are low-rank factor
matrices. The Frobenius-norm-regularized objective is minimized by alternating
between closed-form updates for U (fixing V) and V (fixing U).
"""

from typing import cast

import numpy as np
import pandas as pd

from cil.predictors.base import Predictor


class ALSPredictor(Predictor):
    """Alternating Least Squares baseline.

    Objective::

        min_{U,V} Σ_{observed (u,i)} (r_ui - u_u^T v_i)^2
                  + λ (‖U‖_F^2 + ‖V‖_F^2)

    Each ALS iteration alternates two closed-form steps:

    1. Fix V, solve for each row u_u via ridge regression:
       u_u = (V_Iu^T V_Iu + λ I)^{-1} V_Iu^T r_u
    2. Fix U, solve for each row v_i via ridge regression:
       v_i = (U_Ji^T U_Ji + λ I)^{-1} U_Ji^T r_i

    where I_u is the set of items rated by user u and J_i the set of users
    who rated item i.

    Args:
        k: Latent factor dimensionality.
        reg: Regularization strength (λ).
        n_iters: Number of alternating iterations.
        seed: Random seed for factor initialization.
    """

    def __init__(
        self,
        k: int = 10,
        reg: float = 0.1,
        n_iters: int = 20,
        seed: int = 42,
    ) -> None:
        self._k = k
        self._reg = reg
        self._n_iters = n_iters
        self._seed = seed

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
        reg_I = self._reg * np.eye(k)

        # Initialize factor matrices
        U = rng.normal(0, 0.1, (n_users, k))
        V = rng.normal(0, 0.1, (n_items, k))

        # Build per-user and per-item index structures (sid/pid are ints by construction).
        user_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for u, grp in train_df.groupby("sid"):
            user_data[cast(int, u)] = (
                grp["pid"].values.astype(np.intp),
                grp["rating"].values.astype(np.float64),
            )

        item_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, grp in train_df.groupby("pid"):
            item_data[cast(int, i)] = (
                grp["sid"].values.astype(np.intp),
                grp["rating"].values.astype(np.float64),
            )

        for _ in range(self._n_iters):
            # Fix V, solve for each user vector
            for u in range(n_users):
                if u not in user_data:
                    U[u] = 0.0
                    continue
                items, ratings = user_data[u]
                V_u = V[items]  # (n_rated, k)
                U[u] = np.linalg.solve(V_u.T @ V_u + reg_I, V_u.T @ ratings)

            # Fix U, solve for each item vector
            for i in range(n_items):
                if i not in item_data:
                    V[i] = 0.0
                    continue
                users, ratings = item_data[i]
                U_i = U[users]  # (n_rated, k)
                V[i] = np.linalg.solve(U_i.T @ U_i + reg_I, U_i.T @ ratings)

        self._U = U
        self._V = V

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return np.sum(self._U[sids] * self._V[pids], axis=1)
