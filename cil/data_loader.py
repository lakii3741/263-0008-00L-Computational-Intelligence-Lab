import math
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = Path(data_dir)

    @cached_property
    def ratings(self) -> pd.DataFrame:
        df = pd.read_csv(self._data_dir / "train_ratings.csv")
        return df.astype({"sid": int, "pid": int})

    @cached_property
    def wishlist(self) -> pd.DataFrame:
        df = pd.read_csv(self._data_dir / "train_tbr.csv")
        return df.astype({"sid": int, "pid": int})

    @cached_property
    def submission_pairs(self) -> pd.DataFrame:
        df = pd.read_csv(self._data_dir / "sample_submission.csv")
        parts = df["sid_pid"].str.split("_", expand=True)
        return pd.DataFrame({"sid": parts[0].astype(int), "pid": parts[1].astype(int)})

    @cached_property
    def num_scientists(self) -> int:
        return int(self.ratings["sid"].max()) + 1

    @cached_property
    def num_papers(self) -> int:
        return int(self.ratings["pid"].max()) + 1

    def split(
        self,
        val_fraction: float = 0.25,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Coverage-preserving train/val split.

        Guarantees every sid and pid appears at least once in train.
        Equivalent to a single fold of :meth:`kfold_split`.
        """
        df = self.ratings.reset_index(drop=True)

        # Core: first row per sid + first row per pid
        sid_idx = df.groupby("sid").head(1).index
        pid_idx = df.groupby("pid").head(1).index
        core_idx = sid_idx.union(pid_idx)

        remaining = df.drop(index=core_idx)

        target_train = math.ceil((1 - val_fraction) * len(df))
        add_n = min(max(0, target_train - len(core_idx)), len(remaining))

        if add_n == 0:
            train_df = df.loc[core_idx].reset_index(drop=True)
            val_df = remaining.reset_index(drop=True)
        else:
            extra_train = remaining.sample(n=add_n, random_state=seed)
            val_df = remaining.drop(index=extra_train.index).reset_index(drop=True)
            train_df = pd.concat([df.loc[core_idx], extra_train], ignore_index=True)

        return train_df, val_df

    def _create_kfolds(
        self, df: pd.DataFrame, n_folds: int, seed: int
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        df = df.reset_index(drop=True)

        # Core: first row per sid + first row per pid → always train
        sid_idx = df.groupby("sid").head(1).index
        pid_idx = df.groupby("pid").head(1).index
        core_idx = sid_idx.union(pid_idx)
        core_df = df.loc[core_idx]

        remaining = df.drop(index=core_idx).sample(frac=1, random_state=seed)
        fold_ids = np.arange(len(remaining)) % n_folds

        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(n_folds):
            val_df = remaining.iloc[fold_ids == i].reset_index(drop=True)
            train_extra = remaining.iloc[fold_ids != i]
            train_df = pd.concat([core_df, train_extra], ignore_index=True)
            folds.append((train_df, val_df))
        return folds

    def kfold_split(
        self,
        n_folds: int = 5,
        seed: int = 42,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Coverage-preserving k-fold split.

        The core set (first occurrence of every sid and pid) is always
        placed in training.  The remaining observations are shuffled and
        partitioned into *n_folds* roughly equal groups; for each fold
        one group is validation and the rest (plus core) are training.
        """
        df = self.ratings.reset_index(drop=True)
        folds = self._create_kfolds(df=df, n_folds=n_folds, seed=seed)

        return folds

    def k_fold_held_out_split(
        self,
        n_folds: int = 5,
        seed: int = 42,
        holdout_fraction: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame]]]:
        """Train/holdout split with k-fold CV on the train partition.

        The holdout is drawn once via :meth:`split` and never appears in any
        fold. The remaining train rows are split into ``n_folds`` coverage-
        preserving folds so hyperparameter search on the folds never sees the
        holdout rows.

        Returns:
            train_df: Union of all fold train + val rows (``1 - holdout_fraction`` of data).
            holdout_df: Held-out partition, untouched during CV.
            folds: ``n_folds`` ``(train_fold, val_fold)`` pairs drawn from ``train_df``.
        """
        train_df, holdout_df = self.split(val_fraction=holdout_fraction, seed=seed)
        folds = self._create_kfolds(df=train_df, n_folds=n_folds, seed=seed)
        return train_df, holdout_df, folds

    @staticmethod
    def to_matrix(df: pd.DataFrame, num_scientists: int, num_papers: int) -> np.ndarray:
        """Pivot triplet DataFrame to dense (num_scientists, num_papers) matrix. Missing = NaN."""
        mat = df.pivot(index="sid", columns="pid", values="rating")
        mat = mat.reindex(index=range(num_scientists), columns=range(num_papers))
        return mat.values.astype(np.float64)
