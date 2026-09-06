"""End-to-end demo exercising every component of the cil harness.

Covers: DataLoader (split, kfold_split, k_fold_held_out_split, to_matrix), all
normalizers, all predictors (SVD, IterativeSVD, ALS, SVD++, ItemKNN, KNNBasic,
KNNWithMeans, EnsemblePredictor), CVResult, Evaluator (cross_validate with both
auto-split and pre-computed folds, evaluate, make_submission), benchmark,
param_sweep, and tune_with_holdout / TuneResult.
"""

import os
import tempfile

import numpy as np

from cil.metrics import rmse, mae
from cil import (
    CVResult,
    DataLoader,
    EnsemblePredictor,
    Evaluator,
    IdentityNormalizer,
    ItemKNNPredictor,
    TuneResult,
    benchmark,
    param_sweep,
    tune_with_holdout,
)
from cil.normalizers.baselines import PaperNormalizer, ScientistNormalizer
from cil.predictors.baselines import (
    ALSPredictor,
    IterativeSVDPredictor,
    KNNBasicPredictor,
    KNNWithMeansPredictor,
    SVDPredictor,
    SVDppPredictor,
)

# == 1. DataLoader =============================================================

loader = DataLoader("data")
print(f"Scientists: {loader.num_scientists}, Papers: {loader.num_papers}")
print(f"Ratings:    {len(loader.ratings)}")
print(f"Wishlist:   {len(loader.wishlist)}")
print(f"Submission: {len(loader.submission_pairs)} pairs")

# Splitting
train_df, val_df = loader.split()
print(f"\nTrain/val split: {len(train_df)} / {len(val_df)}")

folds = loader.kfold_split(n_folds=3)
print(f"3-fold split:    {[len(v) for _, v in folds]} val sizes")

train_partition, holdout_df, holdout_folds = loader.k_fold_held_out_split(
    n_folds=3, holdout_fraction=0.2
)
print(
    f"Holdout split:   train {len(train_partition)} / holdout {len(holdout_df)}, "
    f"{len(holdout_folds)} folds with val sizes {[len(v) for _, v in holdout_folds]}"
)

# to_matrix
mat = DataLoader.to_matrix(train_df, loader.num_scientists, loader.num_papers)
print(f"Rating matrix:   {mat.shape}, {np.isnan(mat).sum()} missing")

# == 2. Normalizers ============================================================

print("\n--- Normalizers ---")
for name, NormCls in [
    ("Identity", IdentityNormalizer),
    ("Scientist", ScientistNormalizer),
    ("Paper", PaperNormalizer),
]:
    norm = NormCls()
    normed = norm.fit_transform(train_df)
    sids = val_df["sid"].values[:5]
    pids = val_df["pid"].values[:5]
    dummy_preds = np.full(5, 3.0)
    restored = norm.inverse_transform(sids, pids, dummy_preds)
    print(
        f"  {name:10s}: train mean rating {normed['rating'].mean():.3f}, "
        f"inverse_transform sample {restored[:3].round(3)}"
    )

# == 3. Individual predictors ==================================================

print("\n--- Predictors (single train/val split) ---")
normalizer = IdentityNormalizer()
train_norm = normalizer.fit_transform(train_df)
evaluator = Evaluator(normalizer)

predictors = {
    "SVD(k=2)": SVDPredictor(k=2),
    "IterSVD(k=2)": IterativeSVDPredictor(k=2, n_iters=5),
    "ALS(k=2)": ALSPredictor(k=2, reg=0.1, n_iters=5),
    "SVD++(k=2)": SVDppPredictor(k=2, n_epochs=2),
    "ItemKNN(k=10)": ItemKNNPredictor(k=10, shrinkage=50.0),
    "KNNBasic(k=10)": KNNBasicPredictor(k=10, shrinkage=50.0),
    "KNNWithMeans(k=10)": KNNWithMeansPredictor(k=10, shrinkage=50.0),
}

active_metrics = {"RMSE": rmse, "MAE": mae}

for name, pred in predictors.items():
    pred.fit(
        train_norm, loader.num_scientists, loader.num_papers, wishlist=loader.wishlist
    )
    results = evaluator.evaluate(pred, val_df, active_metrics)
    preds = pred.predict(val_df["sid"].values, val_df["pid"].values)
    metrics_str = ", ".join(
        [f"{name}: {val:.4f}" for name, val in results.scores.items()]
    )
    print(
        f"  {name:18s}: {metrics_str}  "
        f"pred range [{preds.min():.2f}, {preds.max():.2f}]"
    )

# == 4. EnsemblePredictor ======================================================

print("\n--- EnsemblePredictor ---")
ensemble = EnsemblePredictor(
    [
        (SVDPredictor(k=2), IdentityNormalizer()),
        (ALSPredictor(k=2, reg=0.1, n_iters=5), ScientistNormalizer()),
        (SVDppPredictor(k=2, n_epochs=2), PaperNormalizer()),
    ]
)
ensemble.fit(
    train_df, loader.num_scientists, loader.num_papers, wishlist=loader.wishlist
)

result = Evaluator(IdentityNormalizer()).evaluate(ensemble, val_df, active_metrics)
scores_str = ", ".join([f"{name}: {val:.4f}" for name, val in result.scores.items()])
print(f"  Ensemble scores: {scores_str}")

# == 5. Cross-validation & CVResult ===========================================

print("\n--- Cross-validation (3-fold, SVD k=2) ---")
result = Evaluator.cross_validate(
    loader,
    normalizer_factory=IdentityNormalizer,
    predictor_factory=lambda: SVDPredictor(k=2),
    n_folds=3,
)
assert isinstance(result, CVResult)
print(f"  auto-split: {result}")

# Reuse folds from k_fold_held_out_split so CV happens only on the train partition.
result_pre_split = Evaluator.cross_validate(
    loader,
    normalizer_factory=IdentityNormalizer,
    predictor_factory=lambda: SVDPredictor(k=2),
    folds=holdout_folds,
)
assert isinstance(result_pre_split, CVResult)
print(f"  pre-split:  {result_pre_split}")

# == 6. Benchmark ==============================================================

print("\n--- Benchmark ---")
results = benchmark(
    loader,
    {
        "SVD k=2": (lambda: SVDPredictor(k=2), IdentityNormalizer),
        "ALS k=2": (lambda: ALSPredictor(k=2, reg=0.1, n_iters=5), IdentityNormalizer),
    },
    n_folds=3,
    metrics=active_metrics,
)
assert len(results) == 2
for name, r in results.items():
    assert isinstance(r, CVResult)
    assert all(len(scores) == 3 for scores in result.fold_scores.values())

# == 7. Parameter sweep ========================================================

print("\n--- Parameter sweep (SVD, k in [2, 5]) ---")
sweep_results = param_sweep(
    loader,
    SVDPredictor,
    param_grid={"k": [2, 5]},
    normalizer_factory=IdentityNormalizer,
    n_folds=3,
)
assert len(sweep_results) == 2

# == 8. tune_with_holdout =====================================================

print("\n--- tune_with_holdout (SVD, k in [2, 5] × {identity, scientist}) ---")
tune_result = tune_with_holdout(
    data_loader=loader,
    predictor_cls=SVDPredictor,
    search_space={"k": [2, 5]},
    normalizers={"identity": IdentityNormalizer, "scientist": ScientistNormalizer},
    metrics=active_metrics,
    n_folds=2,
    holdout_fraction=0.2,
)
assert isinstance(tune_result, TuneResult)
print(f"  best: norm={tune_result.best_normalizer}, params={tune_result.best_params}")
print(f"  CV scores: {tune_result.cv_scores}")
print(f"  Holdout scores: {tune_result.holdout_scores}")
print(
    f"  oof predictions: {len(tune_result.oof_predictions)} rows, "
    f"holdout predictions: {len(tune_result.holdout_predictions)} rows"
)

# == 9. Submission generation ==================================================

print("\n--- Submission ---")
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "submission.csv")
    normalizer = IdentityNormalizer()
    train_norm = normalizer.fit_transform(loader.ratings)
    pred = SVDPredictor(k=2)
    pred.fit(
        train_norm, loader.num_scientists, loader.num_papers, wishlist=loader.wishlist
    )
    Evaluator(normalizer).make_submission(pred, loader, path)

    import pandas as pd

    sub = pd.read_csv(path)
    print(f"  Rows: {len(sub)}, Columns: {list(sub.columns)}")
    print(f"  Rating range: [{sub['rating'].min():.2f}, {sub['rating'].max():.2f}]")
    assert len(sub) == len(loader.submission_pairs)
    assert list(sub.columns) == ["sid_pid", "rating"]
    assert sub["rating"].between(1.0, 5.0).all()

print("\n=== ALL CHECKS PASSED ===")
