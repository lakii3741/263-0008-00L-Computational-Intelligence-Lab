from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Predictor(ABC):
    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None: ...

    @abstractmethod
    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray: ...
