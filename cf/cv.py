import time
import numpy as np
import pandas as pd
from cf.eval import evaluate


def _coverage_split(ratings, n_folds=5, seed=42):
    """Yield (train_df, val_df) tuples. val_df keeps original ratings index so callers
    can scatter OOF predictions back."""
    rng = np.random.default_rng(seed)
    n = len(ratings)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    all_sids = set(ratings["sid"].unique())
    all_pids = set(ratings["pid"].unique())

    for k in range(n_folds):
        val_pos = folds[k]
        train_pos = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        tr = ratings.iloc[train_pos]
        va = ratings.iloc[val_pos]

        miss_s = all_sids - set(tr["sid"].unique())
        miss_p = all_pids - set(tr["pid"].unique())
        if miss_s or miss_p:
            ks = va[va["sid"].isin(miss_s)].drop_duplicates("sid")
            kp = va[va["pid"].isin(miss_p)].drop_duplicates("pid")
            move = pd.concat([ks, kp]).drop_duplicates()
            tr = pd.concat([tr, move])
            va = va.drop(index=move.index)
        yield tr.reset_index(drop=True), va


def cross_validate(data, build_fn, n_folds=5, seed=42, verbose=True):
    """build_fn(train_df, wishlist) -> fitted predictor."""
    fold_scores = []
    t0 = time.time()
    for k, (tr, va) in enumerate(_coverage_split(data.ratings, n_folds, seed)):
        t = time.time()
        pred = build_fn(tr, data.wishlist)
        s = evaluate(pred, va)
        fold_scores.append(s)
        if verbose:
            print(f"  fold {k+1}/{n_folds}: RMSE={s:.4f}  ({time.time()-t:.1f}s)", flush=True)
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    if verbose:
        print(f"  CV: {mean:.4f} ± {std:.4f}  total {time.time()-t0:.1f}s", flush=True)
    return mean, std, fold_scores


def oof_predictions(data, build_fn, n_folds=5, seed=42, verbose=True):
    """Returns oof vector aligned to data.ratings rows in order, plus the targets."""
    n = len(data.ratings)
    oof = np.full(n, np.nan)
    t0 = time.time()
    for k, (tr, va) in enumerate(_coverage_split(data.ratings, n_folds, seed)):
        t = time.time()
        pred = build_fn(tr, data.wishlist)
        p = pred.predict(va["sid"].to_numpy(), va["pid"].to_numpy())
        oof[va.index.to_numpy()] = np.clip(p, 1.0, 5.0)
        if verbose:
            print(f"  fold {k+1}/{n_folds} predicted ({time.time()-t:.1f}s)", flush=True)
    if verbose:
        print(f"  OOF total {time.time()-t0:.1f}s", flush=True)
    return oof, data.ratings["rating"].to_numpy(dtype=np.float64)
