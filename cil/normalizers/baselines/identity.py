import numpy as np
import pandas as pd

from cil.normalizers.base import Normalizer


class IdentityNormalizer(Normalizer):
    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def inverse_transform(
        self, sids: np.ndarray, pids: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        return predictions
