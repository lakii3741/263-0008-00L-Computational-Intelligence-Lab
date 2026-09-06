"""Per-paper (column) mean-centering normalizer."""

import numpy as np
import pandas as pd

from cil.normalizers.base import Normalizer


class PaperNormalizer(Normalizer):
    """Subtract each paper's mean rating.

    ``fit`` computes the mean rating per paper from the training data.
    ``transform`` centers each rating by subtracting the paper's mean.
    ``inverse_transform`` adds the mean back to restore the original scale.

    Papers unseen during fit fall back to the global mean.
    """

    def __init__(self) -> None:
        self._global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._global_mean = float(train_df["rating"].mean())
        pid_means = train_df.groupby("pid")["rating"].mean()
        max_pid = int(train_df["pid"].max())
        self._means = np.full(max_pid + 1, self._global_mean)
        self._means[pid_means.index.values] = pid_means.values

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["rating"] = df["rating"].values - self._means[df["pid"].values]
        return out

    def inverse_transform(
        self, sids: np.ndarray, pids: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        return predictions + self._means[pids]
