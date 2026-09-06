"""Random hyperparameter sweep for the HGBR + features stacker.

The best architecture from our apples-to-apples table is HGBR over the 39
base predictions augmented with 9 row-context features (i.e. the
gbm_stacking_with_features.py setup). This script samples 20 random
hyperparameter configurations from a sensible grid, evaluates each on the
same 80/20 split (seed 42), and prints a sorted summary.

Runtime: ~25 min for base training + ~30 min for the 20 HGBR fits = ~1 hour.
"""
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cf import Data
from cf.bases import train_bases, row_features
from cf.eval import rmse
from cf.calibrate import fit_isotonic, apply_isotonic


N_SAMPLES = 20
SAMPLE_SEED = 7

GRID = {
    "max_iter":          [600, 1000, 1500, 2000, 2500],
    "max_depth":         [5, 6, 7, 8, 9],
    "learning_rate":     [0.02, 0.03, 0.04, 0.05, 0.06],
    "l2_regularization": [0.5, 1.0, 2.0, 5.0],
}


def sample_configs(rng, n):
    seen = set()
    out = []
    while len(out) < n:
        cfg = {k: rng.choice(v).item() for k, v in GRID.items()}
        key = tuple(cfg.items())
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def main():
    data = Data("data")
    print("training 39 base members via 5-fold CV...")
    Z, _, y, _ = train_bases(data, level="mid")
    F_tr = row_features(data, data.ratings["sid"].to_numpy(),
                              data.ratings["pid"].to_numpy())
    Z_aug = np.concatenate([Z, F_tr], axis=1)
    print(f"sweeping HGBR over {Z.shape[1]} preds + {F_tr.shape[1]} context features")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, va = perm[:n_tr], perm[n_tr:]

    cfgs = sample_configs(np.random.default_rng(SAMPLE_SEED), N_SAMPLES)
    results = []
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        gbm = HistGradientBoostingRegressor(
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25,
            **cfg,
        )
        gbm.fit(Z_aug[tr], y[tr])
        pv = np.clip(gbm.predict(Z_aug[va]), 1.0, 5.0)
        v = rmse(y[va], pv)
        ir = fit_isotonic(pv, y[va])
        v_iso = rmse(y[va], apply_isotonic(ir, pv))
        elapsed = time.time() - t0
        results.append((v, v_iso, gbm.n_iter_, cfg, elapsed))
        print(f"[{i+1:>2}/{N_SAMPLES}] val={v:.4f} +iso={v_iso:.4f} "
              f"trees={gbm.n_iter_:>4} ({elapsed:>4.0f}s) "
              f"depth={cfg['max_depth']} lr={cfg['learning_rate']} "
              f"l2={cfg['l2_regularization']} max_iter={cfg['max_iter']}",
              flush=True)

    results.sort(key=lambda r: r[1])
    print("\n=== top 5 by held-out + isotonic ===")
    for v, v_iso, n_iter_, cfg, _ in results[:5]:
        print(f"  val={v:.4f} +iso={v_iso:.4f} trees={n_iter_:>4}  {cfg}")


if __name__ == "__main__":
    main()
