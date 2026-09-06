"""Single SVD++ baseline.

Trains a biased SVD++ on 80% of the ratings, reports held-out RMSE on the
remaining 20%, fits an isotonic calibration on the held-out predictions,
then refits on the full data, applies the held-out isotonic to the test
predictions, and writes the Kaggle CSV.

This is the strongest single model in our toolbox and is the floor we try to
beat with ensembling. Reported with and without isotonic so the table is
strictly apples-to-apples with the stacking rows.
"""

import numpy as np
from cf import Data, make_submission
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic
from cf.svdpp import SVDpp


SPLIT_SEED = 42  # same split as the stacking scripts


def main():
    data = Data("data")
    print(f"loading data: n_users={data.n_users}, n_items={data.n_items}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(data.ratings))
    n_tr = int(0.8 * len(data.ratings))
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    tr_df = data.ratings.iloc[tr_idx].reset_index(drop=True)
    va_df = data.ratings.iloc[va_idx].reset_index(drop=True)
    print(f"80/20 split: train={len(tr_df)}, val={len(va_df)}")

    model = SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05,
                  seed=42, include_wishlist=True)
    model.fit(tr_df, data.n_users, data.n_items, data.wishlist)
    pv = np.clip(
        model.predict(va_df["sid"].to_numpy(), va_df["pid"].to_numpy()),
        1.0, 5.0,
    )
    y_va = va_df["rating"].to_numpy()
    print(f"single SVD++ held-out RMSE: {rmse(y_va, pv):.4f}")

    ir = fit_isotonic(pv, y_va)
    print(f"with isotonic: {rmse(y_va, apply_isotonic(ir, pv)):.4f}")

    print("refitting on full data for the Kaggle submission...")
    final = SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05,
                  seed=42, include_wishlist=True)
    final.fit(data.ratings, data.n_users, data.n_items, data.wishlist)
    test_pred = np.clip(
        final.predict(data.test_sids, data.test_pids),
        1.0, 5.0,
    )
    test_pred = np.clip(apply_isotonic(ir, test_pred), 1.0, 5.0)

    sub = data.submission_template()
    sub["rating"] = test_pred
    sub.to_csv("predictions_baseline.csv", index=False)
    print("wrote predictions_baseline.csv")


if __name__ == "__main__":
    main()
