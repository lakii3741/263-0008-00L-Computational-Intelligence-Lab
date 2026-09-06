from pathlib import Path
import numpy as np
import pandas as pd


class Data:
    def __init__(self, root="data"):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.ratings = pd.read_csv(root / "train_ratings.csv").astype({"sid": int, "pid": int, "rating": float})
        self.wishlist = pd.read_csv(root / "train_tbr.csv").astype({"sid": int, "pid": int})
        sub = pd.read_csv(root / "sample_submission.csv")
        sp = sub["sid_pid"].str.split("_", expand=True)
        self.test_sids = sp[0].astype(int).to_numpy()
        self.test_pids = sp[1].astype(int).to_numpy()
        self._sub_template = sub
        self.n_users = int(max(self.ratings["sid"].max(), self.wishlist["sid"].max(), self.test_sids.max())) + 1
        self.n_items = int(max(self.ratings["pid"].max(), self.wishlist["pid"].max(), self.test_pids.max())) + 1

    def submission_template(self):
        return self._sub_template.copy()


def cv_splits(ratings, n_folds=5, seed=42):
    """Coverage-preserving K-fold split. Each fold returns (train_df, val_df).
    The smallest train fold is guaranteed to cover every sid and pid present in ratings."""
    rng = np.random.default_rng(seed)
    n = len(ratings)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    splits = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        train_df = ratings.iloc[train_idx].reset_index(drop=True)
        val_df = ratings.iloc[val_idx].reset_index(drop=True)

        # repair coverage: any sid/pid missing from train is moved over from val
        missing_sids = set(ratings["sid"].unique()) - set(train_df["sid"].unique())
        missing_pids = set(ratings["pid"].unique()) - set(train_df["pid"].unique())
        if missing_sids or missing_pids:
            mask_s = val_df["sid"].isin(missing_sids)
            mask_p = val_df["pid"].isin(missing_pids)
            keep_s = val_df[mask_s].drop_duplicates("sid")
            keep_p = val_df[mask_p].drop_duplicates("pid")
            move = pd.concat([keep_s, keep_p]).drop_duplicates()
            train_df = pd.concat([train_df, move], ignore_index=True)
            val_df = val_df.drop(index=move.index).reset_index(drop=True)

        splits.append((train_df, val_df))
    return splits
