import numpy as np
import pyximport
pyximport.install(
    setup_args={"include_dirs": np.get_include()},
    reload_support=True
)

from cil.predictors.baselines.als import ALSPredictor
from cil.predictors.baselines.iterative_svd import IterativeSVDPredictor
from cil.predictors.baselines.svd import SVDPredictor
from cil.predictors.baselines.biasedsvd import BiasedSVDPredictor
from cil.predictors.baselines.svdpp import SVDppPredictor
from cil.predictors.baselines.slopeone import SlopeOnePredictor
from cil.predictors.knn_item import ItemKNNPredictor

from cil.predictors.baselines.knn_basic import KNNBasicPredictor
from cil.predictors.baselines.knn_with_mean import KNNWithMeansPredictor
from cil.predictors.baselines.knn_with_zscore import KNNWithZScorePredictor

from cil.predictors.baselines.globalmean import GlobalMeanPredictor
from cil.predictors.baselines.itemmean import ItemMeanPredictor
from cil.predictors.baselines.usermean import UserMeanPredictor
from cil.predictors.baselines.neumf import NeuMFPredictor
from cil.predictors.baselines.dmf import DMFPredictor
from cil.predictors.baselines.ease import EASEPredictor

__all__ = [
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
    "IterativeSVDPredictor", 
    "SVDPredictor", 
    "BiasedSVDPredictor",
    "SVDppPredictor",
    # Neural models
    "NeuMFPredictor",
    "DMFPredictor",
    "EASEPredictor"
    ]
