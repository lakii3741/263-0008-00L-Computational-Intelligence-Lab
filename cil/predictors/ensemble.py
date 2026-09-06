"""General-purpose ensemble predictor that averages over member models."""

import numpy as np
import pandas as pd

from cil.normalizers import Normalizer
from cil.predictors.base import Predictor


class EnsemblePredictor(Predictor):
    """Uniform average over multiple (predictor, normalizer) pairs.

    Each member operates in its own normalized space.  At prediction time
    every member's output is inverse-transformed back to the original
    rating scale and clipped to [1, 5] before averaging.

    Because the ensemble already handles de-normalization internally,
    pair it with ``IdentityNormalizer`` when using the ``Evaluator``.
    """

    def __init__(self, members: list[tuple[Predictor, Normalizer]]) -> None:
        self._members = members

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        for predictor, normalizer in self._members:
            train_norm = normalizer.fit_transform(train_df)
            predictor.fit(
                train_norm, num_scientists, num_papers, wishlist=wishlist
            )

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        avg = np.zeros(len(sids), dtype=np.float64)
        for predictor, normalizer in self._members:
            raw = predictor.predict(sids, pids)
            preds = normalizer.inverse_transform(sids, pids, raw)
            avg += np.clip(preds, 1.0, 5.0)
        avg /= len(self._members)
        return avg
