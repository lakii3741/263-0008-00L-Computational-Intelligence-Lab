import numpy as np
import pandas as pd

from lenskit.als import ImplicitMFScorer
from lenskit.als._implicit import ImplicitMFConfig
from lenskit.data import from_interactions_df
from lenskit.training import TrainingOptions

from cil.predictors.base import Predictor


class IALSPredictor(Predictor):
    """Implicit ALS (iALS) predictor using LensKit's ImplicitMFScorer.

    Unlike explicit ALS which minimises rating prediction error, iALS treats
    all unobserved interactions as implicit negatives with a confidence weight,
    optimising over the full user-item matrix. This makes it complementary to
    both BPR (pairwise ranking) and explicit ALS (observed ratings only).

    Parameters
    ----------
    n_factors : int
        Latent factor dimensionality.
    n_epochs : int
        Number of ALS iterations.
    reg : float
        L2 regularisation strength.
    weight : float
        Confidence weight for positive (observed) interactions. Higher values
        make the model trust positives more strongly relative to negatives.
    rating_threshold : float | None
        If set, only interactions with ``rating >= rating_threshold`` are
        treated as positive signal. Wishlist items are always positive.
    random_seed : int
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        reg: float = 0.1,
        weight: float = 40.0,
        rating_threshold: float = 10.0,
        random_seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.weight = weight
        self.rating_threshold = rating_threshold
        self.random_seed = random_seed

        self._U: np.ndarray | None = None  # (n_scientists, n_factors)
        self._V: np.ndarray | None = None  # (n_papers, n_factors)

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        """Train iALS on binarised interactions.

        Parameters
        ----------
        train_df : pd.DataFrame
            Interactions with columns ``sid``, ``pid``, and
            optionally ``rating``. Rows below ``rating_threshold`` are dropped.
        num_scientists : int
            Total number of scientists (defines embedding table size).
        num_papers : int
            Total number of papers (defines embedding table size).
        wishlist : pd.DataFrame | None
            Optional extra positive interactions, always included regardless
            of any rating threshold.
        """
        interactions = self._merge_interactions(train_df, wishlist)
        dataset = from_interactions_df(
            interactions.rename(columns={"sid": "user", "pid": "item"}),
            users=list(range(num_scientists)),
            items=list(range(num_papers)),
        )

        config = ImplicitMFConfig(
            embedding_size=self.n_factors,
            epochs=self.n_epochs,
            regularization=self.reg,
            weight=self.weight,
            use_ratings=False,  # treat all interactions as binary
        )
        scorer = ImplicitMFScorer(config=config)
        scorer.train(dataset, TrainingOptions(rng=self.random_seed))

        # Store embeddings as plain numpy arrays for fast inference
        self._U = scorer.user_embeddings  # (n_scientists, n_factors)
        self._V = scorer.item_embeddings  # (n_papers, n_factors)
        return self

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """Predict iALS preference scores for (scientist, paper) pairs.

        Parameters
        ----------
        sids : np.ndarray of int, shape (N,)
            Scientist indices.
        pids : np.ndarray of int, shape (N,)
            Paper indices.

        Returns
        -------
        np.ndarray of float, shape (N,)
        """
        self._check_fitted()
        return np.einsum("nk,nk->n", self._U[sids], self._V[pids])

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

    def _check_fitted(self) -> None:
        if self._U is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")