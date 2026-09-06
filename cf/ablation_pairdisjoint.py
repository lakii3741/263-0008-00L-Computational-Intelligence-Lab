"""Ablation: does the pair-disjointness R ∩ W = ∅ buy us anything?

The hypothesis was: with-wishlist SVD++ and no-wishlist SVD++ should produce
decorrelated residuals, so combining them in the stacker gives more lift than
either alone. This script tests that by training a HGBR stacker on three
configurations of base members from the 'full' library:

    agnostic_only:    KNN + EASE + IterSVD + SVDImpute + bias + EASE-binary
                      (everything that doesn't pick a wishlist channel)
    withish + agnostic:  + the SVD++ family that uses wishlist in Y-aggregator
    nowish + agnostic:   + the SVD++ family that does NOT use wishlist
    both + agnostic:     all of the above

If the disjointness hypothesis holds, 'both' should comfortably beat each of
the single-channel options. Empirically (see report appendix) it does NOT --
HGBR can extract roughly equivalent signal from either channel.

Runtime: ~30 min for base training + 4 HGBR fits.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic


def classify(name: str) -> str:
    if "nowish" in name:
        return "nowish"
    if name.startswith("svdpp_") or name.startswith("bag_svdpp_"):
        return "withish"
    return "agnostic"


def run_hgbr(Z_aug, y, tr, va, label):
    print(f"\n=== {label} (dim={Z_aug.shape[1]}) ===", flush=True)
    gb = HistGradientBoostingRegressor(
        max_iter=2000, max_depth=7, learning_rate=0.03,
        l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
    )
    gb.fit(Z_aug[tr], y[tr])
    pv = np.clip(gb.predict(Z_aug[va]), 1, 5)
    val_rmse = rmse(y[va], pv)
    cal_rmse = rmse(y[va], apply_isotonic(fit_isotonic(pv, y[va]), pv))
    print(f"  val: {val_rmse:.4f}  +iso: {cal_rmse:.4f}  n_iter_={gb.n_iter_}", flush=True)
    return val_rmse, cal_rmse


def main():
    data = Data("data")
    print("training the full base library (47 members)...")
    Z, _, y, names = train_bases(data, level="full")
    print(f"trained {len(names)} members")

    cats = [classify(n) for n in names]
    by_cat = {c: [j for j, cc in enumerate(cats) if cc == c]
              for c in ("withish", "nowish", "agnostic")}
    for c, ix in by_cat.items():
        print(f"  {c:>9}: {len(ix)} members")

    F_tr = row_features(data, data.ratings["sid"].to_numpy(),
                              data.ratings["pid"].to_numpy())

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    configs = {
        "agnostic_only":             by_cat["agnostic"],
        "withish_svdpp + agnostic":  by_cat["withish"] + by_cat["agnostic"],
        "nowish_svdpp + agnostic":   by_cat["nowish"]  + by_cat["agnostic"],
        "both_svdpp + agnostic":     by_cat["withish"] + by_cat["nowish"] + by_cat["agnostic"],
    }

    results = {}
    for label, ix in configs.items():
        Z_aug = np.concatenate([Z[:, ix], F_tr], axis=1)
        results[label] = run_hgbr(Z_aug, y, tr, va, label)

    print("\n" + "=" * 60)
    print(f"{'Config':<35} {'val':>8} {'+iso':>8}")
    print("-" * 60)
    for label, (v, c) in results.items():
        print(f"{label:<35} {v:>8.4f} {c:>8.4f}")


if __name__ == "__main__":
    main()
