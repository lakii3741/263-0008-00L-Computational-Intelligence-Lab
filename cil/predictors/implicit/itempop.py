import numpy as np
import pandas as pd

from lenskit.basic import PopScorer, PopConfig
from lenskit.data import from_interactions_df
from lenskit.training import TrainingOptions

from cil.predictors.base import Predictor


class ItemPopPredictor(Predictor):
    """Popularity-based predictor using LensKit's PopScorer.

    Scores every (scientist, paper) pair by the paper's global popularity
    (interaction count quantile by default). User-agnostic — the same score
    is returned for every scientist. Useful as a non-personalised baseline.

    Parameters
    ----------
    score : {"quantile", "rank", "count"}
        How to compute popularity. ``"quantile"`` (default) gives scores in
        [0, 1]; ``"rank"`` gives integer ranks; ``"count"`` gives raw counts.
    rating_threshold : float | None
        If set, only interactions with ``rating >= rating_threshold`` are
        counted as positive signal. Wishlist items are always counted.
    """

    def __init__(
        self,
        score: str = "quantile",
        rating_threshold: float = 10.0,
    ) -> None:
        self.score = score
        self.rating_threshold = rating_threshold

        self._item_scores: np.ndarray | None = None  # (n_papers,)

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        """Fit popularity scores from interaction counts.

        Parameters
        ----------
        train_df : pd.DataFrame
            Interactions with columns ``sid``, ``pid``, and
            optionally ``rating``. Rows below ``rating_threshold`` are dropped.
        num_scientists : int
            Ignored — popularity is user-agnostic.
        num_papers : int
            Total number of papers (defines score vector size).
        wishlist : pd.DataFrame | None
            Optional extra positive interactions, always counted regardless
            of any rating threshold.
        """
        interactions = self._merge_interactions(train_df, wishlist)
        dataset = from_interactions_df(
            interactions.rename(columns={"sid": "user", "pid": "item"}),
            users=list(range(num_scientists)),
            items=list(range(num_papers)),
        )

        scorer = PopScorer(config=PopConfig(score=self.score))
        scorer.train(dataset, TrainingOptions())
        self._item_scores = scorer.item_scores.copy()  # (n_papers,)
        return self

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """Return popularity scores for (scientist, paper) pairs.

        Scores depend only on ``pids`` — all scientists receive the same value
        for a given paper.

        Parameters
        ----------
        sids : np.ndarray of int, shape (N,)
            Scientist indices (ignored).
        pids : np.ndarray of int, shape (N,)
            Paper indices.

        Returns
        -------
        np.ndarray of float32, shape (N,)
        """
        if self._item_scores is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        return self._item_scores[pids]
    

    def _merge_interactions(
        self,
        train_df: pd.DataFrame,
        wishlist: pd.DataFrame | None,
    ) -> pd.DataFrame:
        positives = train_df.copy()
        if self.rating_threshold is not None and "rating" in positives.columns:
            positives = positives[positives["rating"] >= self.rating_threshold]

        frames = [positives[["sid", "pid"]]]
        if wishlist is not None and not wishlist.empty:
            frames.append(wishlist[["sid", "pid"]])
        return pd.concat(frames, ignore_index=True).drop_duplicates()