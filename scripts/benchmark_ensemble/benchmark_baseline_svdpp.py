"""Single SVD++ + wishlist, evaluated on Wendelin's holdout split.

Same model as baseline_svdpp.py but the train/holdout partition comes from
cil.data_loader.DataLoader so the RMSE is comparable to the other baselines
the team runs on the same loader. The holdout is touched once at the end.
"""

import numpy as np
from cf import Data
from cf.eval import rmse
from cf.svdpp import SVDpp
from cil.data_loader import DataLoader


def main():
    data = Data("data")

    loader = DataLoader("data")
    train_df, holdout_df, _ = loader.k_fold_held_out_split(
        n_folds=5, seed=42, holdout_fraction=0.2
    )
    train_df = train_df.reset_index(drop=True).astype(
        {"sid": int, "pid": int, "rating": float}
    )
    print(f"train rows: {len(train_df)}  holdout rows: {len(holdout_df)}")

    model = SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05,
                  seed=42, include_wishlist=True)
    model.fit(train_df, data.n_users, data.n_items, data.wishlist)

    sids = holdout_df["sid"].to_numpy()
    pids = holdout_df["pid"].to_numpy()
    y_holdout = holdout_df["rating"].to_numpy(dtype=np.float64)
    pred = np.clip(model.predict(sids, pids), 1.0, 5.0)
    print(f"single SVD++ holdout RMSE: {rmse(y_holdout, pred):.4f}")


if __name__ == "__main__":
    main()
