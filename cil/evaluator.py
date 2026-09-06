from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cil.data_loader import DataLoader
from cil.normalizers import Normalizer
from cil.predictors import Predictor
from cil.metrics import Metric, rmse


@dataclass
class CVResult:
    """Result of k-fold cross-validation for multiple metrics."""

    fold_scores: dict[str, list[float]]
    oof_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def mean(self) -> dict[str, float]:
        return {name: float(np.mean(vals)) for name, vals in self.fold_scores.items()}

    @property
    def std(self) -> dict[str, float]:
        return {name: float(np.std(vals)) for name, vals in self.fold_scores.items()}

    def __repr__(self) -> str:
        stats_str = ", ".join(
            [f"{k}={self.mean[k]:.4f} (±{self.std[k]:.4f})" for k in self.fold_scores]
        )
        num_folds = len(next(iter(self.fold_scores.values())))
        return f"CVResult({stats_str}, folds={num_folds})"


@dataclass
class EvalResult:
    scores: dict[str, float]
    predictions: pd.DataFrame


class Evaluator:
    def __init__(self, normalizer: Normalizer) -> None:
        self._normalizer = normalizer

    @classmethod
    def cross_validate(
        cls,
        data_loader: DataLoader,
        normalizer_factory: Callable[[], Normalizer],
        predictor_factory: Callable[[], Predictor],
        metrics: dict[str, Metric] | None = None,
        n_folds: int = 5,
        seed: int = 42,
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    ) -> CVResult:
        """K-fold cross validation.

        For each fold a fresh normalizer and predictor are created via
        the provided factories, fitted on the training split, and
        evaluated on the held-out validation split.

        If ``folds`` is given, it is used verbatim (``n_folds`` and ``seed`` are
        ignored for splitting). Pass pre-computed folds when running CV on a
        subset of the data — e.g. the train partition of a train/holdout split.
        """
        active_metrics = metrics if metrics is not None else {"RMSE": rmse}
        if folds is None:
            folds = data_loader.kfold_split(n_folds=n_folds, seed=seed)
        n_folds = len(folds)
        fold_scores: dict[str, list[float]] = {name: [] for name in active_metrics}
        oof_chunks: list[pd.DataFrame] = []

        for i, (train_df, val_df) in enumerate(folds):
            normalizer = normalizer_factory()
            train_norm = normalizer.fit_transform(train_df)

            predictor = predictor_factory()
            predictor.fit(
                train_norm,
                data_loader.num_scientists,
                data_loader.num_papers,
                wishlist=data_loader.wishlist,
            )

            eval_result = cls(normalizer).evaluate(predictor, val_df, active_metrics)

            for name, score in eval_result.scores.items():
                fold_scores[name].append(score)
            oof_chunks.append(eval_result.predictions)

            score_line = ", ".join(
                [f"{n}: {s:.4f}" for n, s in eval_result.scores.items()]
            )
            print(f"  Fold {i + 1}/{n_folds}: {score_line}")

        oof_predictions = pd.concat(oof_chunks, ignore_index=True)
        result = CVResult(fold_scores=fold_scores, oof_predictions=oof_predictions)

        stats_summary = "  ".join(
            [
                f"Mean {name}: {result.mean[name]:.4f} (±{result.std[name]:.4f})"
                for name in result.mean
            ]
        )
        print(stats_summary)
        return result

    def evaluate(
        self,
        predictor: Predictor,
        val_df: pd.DataFrame,
        metrics: dict[str, Metric] | None = None,
        bootstrap: bool = False,
        n_iterations: int = 10_000,
        ci: float = 95.0,
        seed: int = 42,
    ) -> EvalResult:
        active_metrics = metrics if metrics is not None else {"RMSE": rmse}

        sids = val_df["sid"].values
        pids = val_df["pid"].values

        preds = predictor.predict(sids, pids)
        preds = self._normalizer.inverse_transform(sids, pids, preds)
        preds = np.clip(preds, 1.0, 5.0)

        scores = {
            name: metric_fn(val_df["rating"].values.astype(np.float64), preds)
            for name, metric_fn in active_metrics.items()
        }

        if bootstrap:
                    rng = np.random.default_rng(seed)
                    y_true = val_df["rating"].values.astype(np.float64)
                    n_size = len(y_true)
                    boot_stats = {name: [] for name in active_metrics}

                    for _ in range(n_iterations):
                        indices = rng.choice(n_size, size=n_size, replace=True)
                        resampled_true = y_true[indices]
                        resampled_pred = preds[indices]
                        for name, metric_fn in active_metrics.items():
                            boot_stats[name].append(metric_fn(resampled_true, resampled_pred))

                    for name, stats in boot_stats.items():
                        lower_bound = np.percentile(stats, (100 - ci) / 2)
                        upper_bound = np.percentile(stats, 100 - (100 - ci) / 2)
                        scores[f"bootstrap_{name}_mean"] = float(np.mean(stats))
                        scores[f"bootstrap_{name}_lower"] = float(lower_bound)
                        scores[f"bootstrap_{name}_upper"] = float(upper_bound)
                        scores[f"bootstrap_{name}_error"] = float((upper_bound - lower_bound) / 2)

        predictions = pd.DataFrame(
            {
                "sid": sids,
                "pid": pids,
                "rating_true": val_df["rating"].values.astype(np.float64),
                "rating_pred": preds,
            }
        )

        return EvalResult(scores=scores, predictions=predictions)

    def make_submission(
        self,
        predictor: Predictor,
        data_loader: DataLoader,
        path: str | Path = "submission.csv",
    ) -> None:
        sp = data_loader.submission_pairs
        sids = sp["sid"].values
        pids = sp["pid"].values
        preds = predictor.predict(sids, pids)
        preds = self._normalizer.inverse_transform(sids, pids, preds)
        preds = np.clip(preds, 1.0, 5.0)

        out = pd.DataFrame(
            {
                "sid_pid": [f"{s}_{p}" for s, p in zip(sids, pids)],
                "rating": preds,
            }
        )
        out.to_csv(path, index=False)
