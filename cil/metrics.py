"""Rating-prediction metrics.

Every metric follows the :class:`Metric` protocol: take two 1-D float arrays
``y_true`` and ``y_pred`` of equal length and return a single ``float`` score.
Lower is better for all metrics defined here.
"""

from typing import Protocol

import numpy as np


class Metric(Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))
