"""
Ablation study: Metalearners
    - mean:  simple average of base predictions
    - ridge: ridge regression stacking
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import RidgeCV

from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic
from cil.data_loader import DataLoader


# =================================
# Metalearner implementations
# =================================

def fit_mean(_Z_tr, _y_tr):
    """Returns a stateless callable that averages base predictions."""
    def predict(Z):
        return Z[:, : _Z_tr.shape[1]].mean(axis=1)   # ignores context cols
    return predict


def fit_ridge(Z_tr, y_tr):
    alphas = np.logspace(-3, 5, 30)
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(Z_tr, y_tr)
    print(f"  ridge best alpha: {model.alpha_:.4g}")
    return model.predict


METALEARNERS = {
    "mean":  fit_mean,
    "ridge": fit_ridge,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Metalearner ablation study")
    parser.add_argument(
        "--metalearner",
        choices=list(METALEARNERS),
        required=True,
        help="Which metalearner to use for stacking",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metalearner_name = args.metalearner

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

    print("training 47 base members via 5-fold CV on the train partition...")
    Z, P, y, _ = train_bases(data, level="full")

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

    print(f"fitting metalearner: {metalearner_name}")
    fit_fn = METALEARNERS[metalearner_name]
    predict_fn = fit_fn(Z_aug[tr], y[tr])

    pv = np.clip(predict_fn(Z_aug[va]), 1.0, 5.0)
    print(f"inner-val RMSE ({metalearner_name}): {rmse(y[va], pv):.4f}")

    ir = fit_isotonic(pv, y[va])

    predict_full = fit_fn(Z_aug, y)

    holdout_pred = np.clip(predict_full(P_aug), 1.0, 5.0)
    holdout_pred = np.clip(apply_isotonic(ir, holdout_pred), 1.0, 5.0)
    print(f"holdout RMSE ({metalearner_name}): {rmse(y_holdout, holdout_pred):.4f}")

    output_dir = "results/ablations"
    file_path = f"{output_dir}/metalearners_{metalearner_name}.npz"
    os.makedirs(output_dir, exist_ok=True)
    np.savez(file_path, y_holdout=y_holdout, holdout_pred=holdout_pred)
    print(f"results saved to {file_path}")


if __name__ == "__main__":
    main()