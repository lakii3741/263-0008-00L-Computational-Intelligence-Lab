"""HGBR over 39 base members (no row features), evaluated on Wendelin's holdout split.

Same recipe as gbm_stacking.py but using cil.data_loader's train/holdout
partition. The inner 80/20 used for HGBR early stopping and isotonic stays
inside the train portion; holdout is touched once for RMSE.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cf import Data
from cf.bases import train_bases
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic
from cil.data_loader import DataLoader


def main():
    data = Data("data")

    loader = DataLoader("data")
    train_df, holdout_df, _ = loader.k_fold_held_out_split(
        n_folds=5, seed=42, holdout_fraction=0.2
    )

    data.ratings = train_df.reset_index(drop=True).astype(
        {"sid": int, "pid": int, "rating": float}
    )
    data.test_sids = holdout_df["sid"].to_numpy()
    data.test_pids = holdout_df["pid"].to_numpy()
    y_holdout = holdout_df["rating"].to_numpy(dtype=np.float64)
    print(f"train rows: {len(data.ratings)}  holdout rows: {len(holdout_df)}")

    print("training 39 base members via 5-fold CV on the train partition...")
    Z, P, y, _ = train_bases(data, level="mid")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    gbm = HistGradientBoostingRegressor(
        max_iter=600, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    )
    gbm.fit(Z[tr], y[tr])
    pv = np.clip(gbm.predict(Z[va]), 1.0, 5.0)
    print(f"GBM inner-val RMSE: {rmse(y[va], pv):.4f}  ({gbm.n_iter_} trees)")

    ir = fit_isotonic(pv, y[va])

    gbm_full = HistGradientBoostingRegressor(
        max_iter=gbm.n_iter_, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=42, early_stopping=False,
    )
    gbm_full.fit(Z, y)
    holdout_pred = np.clip(gbm_full.predict(P), 1.0, 5.0)
    holdout_pred = np.clip(apply_isotonic(ir, holdout_pred), 1.0, 5.0)
    print(f"GBM holdout RMSE: {rmse(y_holdout, holdout_pred):.4f}")


if __name__ == "__main__":
    main()
