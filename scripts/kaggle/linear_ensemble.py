"""NNLS-blended ensemble of 39 classical CF predictors.

Trains 39 base members (SVD++ family, ItemKNN, EASE, IterSVD, plus a few
bagged/EASE-binary variants) via 5-fold CV, splits the OOF rows into
80/20, fits non-negative least-squares on the 80%, reports held-out
RMSE on the 20%, applies isotonic post-calibration learned on the 20%
predictions, then refits NNLS on the full OOF for the final test
submission.

This is the standard "average a bunch of models" recipe -- our linear
stacking baseline. Compared apples-to-apples against gbm_stacking.py
(same 39 base members, only the stacker class differs).
"""
from pathlib import Path
import numpy as np
from cf import Data
from cf.bases import train_bases
from cf.ensemble import fit_nnls
from cf.calibrate import fit_isotonic, apply_isotonic
from cf.eval import rmse


def main():
    data = Data("data")
    print("training 14 base members via 5-fold CV...")
    Z, P, y, names = train_bases(data, level="mid")
    print(f"done: {Z.shape[1]} members, {len(y)} OOF rows")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    w = fit_nnls(Z[tr], y[tr])
    pv = np.clip(Z[va] @ w, 1.0, 5.0)
    print(f"NNLS held-out RMSE: {rmse(y[va], pv):.4f}")
    for n, wi in zip(names, w):
        if wi > 0.005:
            print(f"  {n:<35} {wi:.4f}")

    ir = fit_isotonic(pv, y[va])
    print(f"with isotonic: {rmse(y[va], apply_isotonic(ir, pv)):.4f}")

    # final submission: refit NNLS on the full OOF, apply held-out isotonic
    w_full = fit_nnls(Z, y)
    test_pred = np.clip(P @ w_full, 1.0, 5.0)
    test_pred = np.clip(apply_isotonic(ir, test_pred), 1.0, 5.0)

    sub = data.submission_template()
    sub["rating"] = test_pred
    out_path = Path("results") / "kaggle" / "predictions_linear.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
