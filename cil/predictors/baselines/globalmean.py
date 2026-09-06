"""GlobalMean predictor for collaborative filtering.
Always predicts the global mean rating from the training data.
"""
import numpy as np
import pandas as pd
from cil.predictors.base import Predictor
from typing import Optional, Dict, List

class GlobalMeanPredictor(Predictor):
    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: Optional[pd.DataFrame] = None,
    ) -> None:
        self.global_mean = float(train_df["rating"].mean())

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        
        return self.global_mean * np.ones_like(sids, dtype=np.float64)