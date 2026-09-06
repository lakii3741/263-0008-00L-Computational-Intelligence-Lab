"""NNLS over 39 base members, evaluated on Wendelin's holdout split.

Same recipe as linear_ensemble.py but using cil.data_loader's train/holdout
partition. NNLS is fit on the inner 80% of train, isotonic on the inner 20%,
then refit on full train and evaluated once on holdout.
"""

import numpy as np
from cf import Data
from cf.bases import train_bases
from cf.ensemble import fit_nnls
from cf.calibrate import fit_isotonic, apply_isotonic
from cf.eval import rmse
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
    Z, P, y, names = train_bases(data, level="mid")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    w = fit_nnls(Z[tr], y[tr])
    pv = np.clip(Z[va] @ w, 1.0, 5.0)
    print(f"NNLS inner-val RMSE: {rmse(y[va], pv):.4f}")
    for n, wi in zip(names, w):
        if wi > 0.005:
            print(f"  {n:<35} {wi:.4f}")

    ir = fit_isotonic(pv, y[va])

    w_full = fit_nnls(Z, y)
    holdout_pred = np.clip(P @ w_full, 1.0, 5.0)
    holdout_pred = np.clip(apply_isotonic(ir, holdout_pred), 1.0, 5.0)
    print(f"NNLS holdout RMSE: {rmse(y_holdout, holdout_pred):.4f}")


if __name__ == "__main__":
    main()
