"""HistGradientBoosting stacker over 39 base predictions.

Same idea as linear_ensemble.py but the meta-learner is a tree boosting
model instead of a weighted sum. Tree splits express conditional
combinations of base predictions that NNLS / Ridge cannot.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cf import Data
from cf.bases import train_bases
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic


def main():
    data = Data("data")
    print("training 39 base members via 5-fold CV...")
    Z, P, y, _ = train_bases(data, level="mid")
    print(f"done: {Z.shape[1]} base predictors over {len(y)} rows")

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
    print(f"GBM held-out RMSE: {rmse(y[va], pv):.4f}  ({gbm.n_iter_} trees)")

    # refit on the full data with the iteration count chosen by early stopping
    gbm_full = HistGradientBoostingRegressor(
        max_iter=gbm.n_iter_, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=42, early_stopping=False,
    )
    gbm_full.fit(Z, y)
    test_pred = np.clip(gbm_full.predict(P), 1.0, 5.0)

    ir = fit_isotonic(pv, y[va])
    print(f"with isotonic: {rmse(y[va], apply_isotonic(ir, pv)):.4f}")
    test_pred = np.clip(apply_isotonic(ir, test_pred), 1.0, 5.0)

    sub = data.submission_template()
    sub["rating"] = test_pred
    out_path = Path("results") / "kaggle" / "predictions_gbm.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
