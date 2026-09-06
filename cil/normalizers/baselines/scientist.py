"""Per-scientist (row) mean-centering normalizer."""

import numpy as np
import pandas as pd

from cil.normalizers.base import Normalizer


class ScientistNormalizer(Normalizer):
    """Subtract each scientist's mean rating.

    ``fit`` computes the mean rating per scientist from the training data.
    ``transform`` centers each rating by subtracting the scientist's mean.
    ``inverse_transform`` adds the mean back to restore the original scale.

    Scientists unseen during fit fall back to the global mean.
    """

    def __init__(self) -> None:
        self._global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._global_mean = float(train_df["rating"].mean())
        sid_means = train_df.groupby("sid")["rating"].mean()
        max_sid = int(train_df["sid"].max())
        self._means = np.full(max_sid + 1, self._global_mean)
        self._means[sid_means.index.values] = sid_means.values

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["rating"] = df["rating"].values - self._means[df["sid"].values]
        return out

    def inverse_transform(
        self, sids: np.ndarray, pids: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        return predictions + self._means[sids]
