from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Normalizer(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def inverse_transform(
        self, sids: np.ndarray, pids: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray: ...

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)
