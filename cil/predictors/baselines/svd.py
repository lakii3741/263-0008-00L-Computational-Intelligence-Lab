import numpy as np
import pandas as pd

from cil.data_loader import DataLoader
from cil.predictors.base import Predictor


class SVDPredictor(Predictor):
    def __init__(self, k: int = 2, fill: float = 3.0) -> None:
        self._k = k
        self._fill = fill

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        mat = DataLoader.to_matrix(train_df, num_scientists, num_papers)
        mat = np.nan_to_num(mat, nan=self._fill)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        self._recon = (U[:, : self._k] * S[: self._k]) @ Vh[: self._k]

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        return self._recon[sids, pids]
