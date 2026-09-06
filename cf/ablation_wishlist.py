"""Ablation: does the wishlist channel actually help score?

Compares two HGBR stackers on the same 80/20 split:

    RATING-ONLY: drops every member or feature that uses the wishlist
        - members: no-wishlist SVD++ + KNN + EASE + IterSVD + SVDImpute + bias
        - row features: u/i_log_count, u/i_mean, u/i_std (NO has_wish, NO wish counts)
    FULL: every member of the 'full' library + all 9 row features.

If RATING-ONLY ≈ FULL, the wishlist adds nothing and we should not center the
report on it. We measured ~0.001 RMSE difference -- useful but not the headline.

Runtime: ~30 min for base training + 2 HGBR fits.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic


def classify_rating_only(name: str) -> bool:
    """True iff this member uses ONLY rating data, never wishlist."""
    n = name.lower()
    if "nowish" in n: return True
    if "ease_bin" in n: return False
    if n.startswith("bag_svdpp"): return False
    if n.startswith("svdpp_"): return False
    return n.startswith(("knn_", "ease_", "itersvd", "svdimpute", "bias"))


def row_features_no_wish(data, sids, pids):
    """Same as row_features but drops the 3 wishlist-related columns."""
    full = row_features(data, sids, pids)  # 9 columns; last 3 are wishlist-related
    return full[:, :6]


def run_hgbr(Z_aug, y, tr, va, label):
    print(f"\n=== {label} (dim={Z_aug.shape[1]}) ===", flush=True)
    gb = HistGradientBoostingRegressor(
        max_iter=2500, max_depth=7, learning_rate=0.025,
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

    rating_only_idx = [j for j, n in enumerate(names) if classify_rating_only(n)]
    print(f"  rating-only members: {len(rating_only_idx)} of {len(names)}")
    print(f"    {[names[j] for j in rating_only_idx]}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    # rating-only: drop wishlist members AND wishlist row features
    F_tr_ro = row_features_no_wish(data, data.ratings["sid"].to_numpy(),
                                         data.ratings["pid"].to_numpy())
    Z_ro = np.concatenate([Z[:, rating_only_idx], F_tr_ro], axis=1)
    v_ro, c_ro = run_hgbr(Z_ro, y, tr, va, "RATING-ONLY (no wishlist info anywhere)")

    # full: every member, every row feature
    F_tr = row_features(data, data.ratings["sid"].to_numpy(),
                              data.ratings["pid"].to_numpy())
    Z_full = np.concatenate([Z, F_tr], axis=1)
    v_f, c_f = run_hgbr(Z_full, y, tr, va, f"FULL (all {len(names)} members + 9 row feats)")

    print("\n" + "=" * 60)
    print(f"{'Config':<45} {'val':>8} {'+iso':>8}")
    print("-" * 60)
    print(f"{'RATING-ONLY':<45} {v_ro:>8.4f} {c_ro:>8.4f}")
    print(f"{'FULL':<45} {v_f:>8.4f} {c_f:>8.4f}")
    print(f"{'Δ from wishlist':<45} {v_ro - v_f:>+8.4f} {c_ro - c_f:>+8.4f}")


if __name__ == "__main__":
    main()
