import numpy as np
import pyximport
pyximport.install(
    setup_args={"include_dirs": np.get_include()},
    reload_support=True
)

from cil.predictors.base import Predictor
from cil.predictors.baselines import (
    ALSPredictor,
    IterativeSVDPredictor,
    SVDPredictor,
    SVDppPredictor,
    BiasedSVDPredictor,
    KNNBasicPredictor,
    KNNWithMeansPredictor,
    KNNWithZScorePredictor,
    GlobalMeanPredictor,
    ItemMeanPredictor,
    UserMeanPredictor,
    SlopeOnePredictor,
    NeuMFPredictor,
    DMFPredictor,
    EASEPredictor
)
from cil.predictors.implicit import (
    BPRPredictor,
    ItemPopPredictor,
    IALSPredictor,
    EASEBinaryPredictor
)

from cil.predictors.ensemble import EnsemblePredictor
from cil.predictors.knn_item import ItemKNNPredictor

__all__ = [
    "Predictor",
    # naive baselines
    "GlobalMeanPredictor",
    "ItemMeanPredictor",
    "UserMeanPredictor",
    # memory-based models
    "KNNBasicPredictor",
    "KNNWithMeansPredictor",
    "KNNWithZScorePredictor",
    "ItemKNNPredictor",
    "SlopeOnePredictor",
    # Matrix Factorization based models
    "ALSPredictor",
    "EnsemblePredictor",
    "IterativeSVDPredictor",
    "SVDPredictor",
    "BiasedSVDPredictor",
    "SVDppPredictor",
    # Neural models
    "NeuMFPredictor",
    "DMFPredictor",
    "EASEPredictor",
    # Implicit models
    "ItemPopPredictor",
    "BPRPredictor",
    "IALSPredictor",
    "EASEBinaryPredictor"
]
