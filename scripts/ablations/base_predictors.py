"""47 base members + GBM + context features, evaluated on Wendelin's holdout split.

Ablation over ensemble components:
  no_svdpp          remove all SVD++ members
  no_knn            remove ItemKNN members
  no_bagged_svdpp   remove bagged SVD++ members
  no_ease           remove EASE members
  no_cold_start     remove cold-start members (IterSVD, SVDImpute, BiasModel)
"""

import os
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic
from cil.data_loader import DataLoader

LEVELS = [
    "no_svdpp",
    "no_knn",
    "no_bagged_svdpp",
    "no_ease",
    "no_cold_start",
]


def main(level):
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

    print(f"training base members using 5-fold CV on the train partition (level={level})...")
    Z, P, y, _ = train_bases(data, level=level)

    F_tr = row_features(data, data.ratings["sid"].to_numpy(),
                              data.ratings["pid"].to_numpy())
    F_te = row_features(data, data.test_sids, data.test_pids)
    Z_aug = np.concatenate([Z, F_tr], axis=1)
    P_aug = np.concatenate([P, F_te], axis=1)
    print(f"final stack: {Z.shape[1]} base preds + {F_tr.shape[1]} context features")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    gbm = HistGradientBoostingRegressor(
        max_iter=2000, max_depth=7, learning_rate=0.03,
        l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
    )
    gbm.fit(Z_aug[tr], y[tr])
    pv = np.clip(gbm.predict(Z_aug[va]), 1.0, 5.0)
    print(f"final inner-val RMSE: {rmse(y[va], pv):.4f}  ({gbm.n_iter_} trees)")

    ir = fit_isotonic(pv, y[va])

    gbm_full = HistGradientBoostingRegressor(
        max_iter=gbm.n_iter_, max_depth=7, learning_rate=0.03,
        l2_regularization=1.0, random_state=42, early_stopping=False,
    )
    gbm_full.fit(Z_aug, y)
    holdout_pred = np.clip(gbm_full.predict(P_aug), 1.0, 5.0)
    holdout_pred = np.clip(apply_isotonic(ir, holdout_pred), 1.0, 5.0)
    print(f"final holdout RMSE: {rmse(y_holdout, holdout_pred):.4f}")

    output_dir = "results/ablations"
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/components_{level}.npz"
    np.savez(file_path, y_holdout=y_holdout, holdout_pred=holdout_pred)
    print(f"results saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=LEVELS)
    args = parser.parse_args()
    main(args.level)