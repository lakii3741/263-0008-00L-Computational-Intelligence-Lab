"""Bagging wrapper. Trains the base predictor on a bootstrap sample of the train df."""
import numpy as np
from cf.base import Predictor


class Bagged(Predictor):
    def __init__(self, base_factory, frac=0.8, seed=0):
        self.base_factory = base_factory
        self.frac = frac
        self.seed = seed
        self.p = None

    def fit(self, train_df, n_users, n_items, wishlist=None):
        rng = np.random.default_rng(self.seed)
        n = len(train_df)
        m = int(self.frac * n)
        idx = rng.integers(0, n, size=m)
        sub = train_df.iloc[idx].reset_index(drop=True)
        self.p = self.base_factory()
        self.p.fit(sub, n_users, n_items, wishlist)
        return self

    def predict(self, sids, pids):
        return self.p.predict(sids, pids)
