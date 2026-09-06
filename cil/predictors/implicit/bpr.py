from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lenskit.data import from_interactions_df
from lenskit.flexmf._implicit import FlexMFImplicitConfig, FlexMFImplicitScorer
from lenskit.pipeline import RecPipelineBuilder
from lenskit.training import TrainingOptions
from cil.predictors.base import Predictor


class BPRPredictor(Predictor):
    """Bayesian Personalized Ranking for implicit scientist-paper interactions.

    Combines observed interactions (train_df) with wishlist items as
    additional positive signal before fitting BPR via LensKit's FlexMF.

    Parameters
    ----------
    n_factors : int
        Latent factor dimensionality.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    reg : float
        L2 regularisation strength.
    neg_count : int
        Negative samples drawn per positive interaction per step.
    random_seed : int
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 30,
        lr: float = 0.05,
        reg: float = 0.01,
        neg_count: int = 1,
        random_seed: int = 42,
        rating_threshold: float = 10.0,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.neg_count = neg_count
        self.random_seed = random_seed
        self.rating_threshold = rating_threshold

        self._pipeline = None
        self._num_scientists: int | None = None
        self._num_papers: int | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        """Train BPR on interactions, optionally augmented with wishlist.

        Parameters
        ----------
        train_df : pd.DataFrame
            Interactions with columns ``sid``, ``pid``, and
            optionally ``rating``. Ratings are binarised: rows at or above
            ``rating_threshold`` are kept as positives; rows below are dropped.
            If ``rating_threshold`` is ``None``, all rows are treated as
            positives regardless of rating.
        num_scientists : int
            Total number of scientists (defines embedding table size).
        num_papers : int
            Total number of papers (defines embedding table size).
        wishlist : pd.DataFrame | None
            Optional additional positive signal with columns ``sid``
            and ``pid``. Wishlist items are always treated as positives
            (no threshold applied). Merged and de-duplicated with train_df
            before fitting.
        """
        self._num_scientists = num_scientists
        self._num_papers = num_papers

        interactions = self._merge_interactions(train_df, wishlist, self.rating_threshold)
        dataset = from_interactions_df(
            interactions.rename(columns={"sid": "user", "pid": "item"}),
            users=list(range(num_scientists)),
            items=list(range(num_papers)),
        )

        config = FlexMFImplicitConfig(
            preset="bpr",
            embedding_size=self.n_factors,
            epochs=self.n_epochs,
            learning_rate=self.lr,
            regularization=self.reg,
            negative_count=self.neg_count,
        )
        scorer = FlexMFImplicitScorer(config=config)
        builder = RecPipelineBuilder()
        builder.scorer(scorer)
        builder.ranker(n=num_papers)
        self._pipeline = builder.build("BPR")
        self._pipeline.train(dataset, options=TrainingOptions(rng=self.random_seed))
        return self

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """Predict preference scores for (scientist, paper) pairs.

        Parameters
        ----------
        sids : np.ndarray of int, shape (N,)
            Scientist indices.
        pids : np.ndarray of int, shape (N,)
            Paper indices.

        Returns
        -------
        np.ndarray of float, shape (N,)
            BPR preference scores (higher = more preferred).
        """
        self._check_fitted()

        scorer = self._pipeline.component("scorer")
        model = scorer.model  # FlexMFModel (torch.nn.Module)
        U = model.u_embed.weight.detach().cpu().numpy()  # (n_scientists, k)
        V = model.i_embed.weight.detach().cpu().numpy()  # (n_papers, k)

        return np.einsum("nk,nk->n", U[sids], V[pids])

    @staticmethod
    def _merge_interactions(
        train_df: pd.DataFrame,
        wishlist: pd.DataFrame | None,
        rating_threshold: float | None,
    ) -> pd.DataFrame:
        """Binarise ratings and merge with wishlist, dropping duplicates.

        Rows in train_df with a rating below ``rating_threshold`` are treated
        as negatives and excluded. Wishlist items are always kept as positives.
        """
        positives = train_df.copy()
        if rating_threshold is not None and "rating" in positives.columns:
            positives = positives[positives["rating"] >= rating_threshold]

        frames = [positives[["sid", "pid"]]]
        if wishlist is not None and not wishlist.empty:
            frames.append(wishlist[["sid", "pid"]])
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")