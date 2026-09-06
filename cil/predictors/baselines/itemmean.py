"""ItemMean predictor for collaborative filtering.
Always predicts the item mean rating from the training data.
If an item has no ratings in the training data, it falls back to the global mean.
"""
import numpy as np
import pandas as pd
from cil.predictors.base import Predictor
from typing import Optional, Dict, List

class ItemMeanPredictor(Predictor):
    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: Optional[pd.DataFrame] = None,
    ) -> None:
        self.global_mean = float(train_df["rating"].mean())
        self.item_means = train_df.groupby("pid")["rating"].mean()

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        pred = self.item_means.reindex(pids).fillna(self.global_mean)
        return pred.values