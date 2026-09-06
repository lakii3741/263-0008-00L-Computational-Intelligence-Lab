"""HistGradientBoosting stacker with per-(user, item) context features.

Same 39 base predictions as gbm_stacking.py, but the meta-learner now also
sees row-level context: how many ratings the user/item has, their mean
rating in train, the rating std, wishlist activity, and a binary
'has_wish' indicator. The boosted tree can use these to weight bases
differently per row.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic


def main():
    data = Data("data")
    print("training 39 base members via 5-fold CV...")
    Z, P, y, _ = train_bases(data, level="mid")

    F_tr = row_features(data, data.ratings["sid"].to_numpy(),
                              data.ratings["pid"].to_numpy())
    F_te = row_features(data, data.test_sids, data.test_pids)
    Z_aug = np.concatenate([Z, F_tr], axis=1)
    P_aug = np.concatenate([P, F_te], axis=1)
    print(f"stacking {Z.shape[1]} base preds + {F_tr.shape[1]} context features")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    gbm = HistGradientBoostingRegressor(
        max_iter=800, max_depth=7, learning_rate=0.04,
        l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
    )
    gbm.fit(Z_aug[tr], y[tr])
    pv = np.clip(gbm.predict(Z_aug[va]), 1.0, 5.0)
    print(f"GBM+context held-out RMSE: {rmse(y[va], pv):.4f}  ({gbm.n_iter_} trees)")

    gbm_full = HistGradientBoostingRegressor(
        max_iter=gbm.n_iter_, max_depth=7, learning_rate=0.04,
        l2_regularization=1.0, random_state=42, early_stopping=False,
    )
    gbm_full.fit(Z_aug, y)
    test_pred = np.clip(gbm_full.predict(P_aug), 1.0, 5.0)

    ir = fit_isotonic(pv, y[va])
    print(f"with isotonic: {rmse(y[va], apply_isotonic(ir, pv)):.4f}")
    test_pred = np.clip(apply_isotonic(ir, test_pred), 1.0, 5.0)

    sub = data.submission_template()
    sub["rating"] = test_pred
    out_path = Path("results") / "kaggle" / "predictions_gbm_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
