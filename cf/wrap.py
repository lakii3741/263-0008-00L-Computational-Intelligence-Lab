"""Wrap a Predictor with a Normalizer so we can mix normalizers in the ensemble."""
import numpy as np
from cf.base import Predictor


class Normalized(Predictor):
    def __init__(self, predictor, normalizer):
        self.p = predictor
        self.n = normalizer

    def fit(self, train_df, n_users, n_items, wishlist=None):
        self.n.fit(train_df)
        tn = self.n.transform(train_df)
        self.p.fit(tn, n_users, n_items, wishlist)
        return self

    def predict(self, sids, pids):
        raw = self.p.predict(sids, pids)
        return self.n.inverse(sids, pids, raw)
