"""Benchmark and parameter-sweep utilities for predictor evaluation."""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from tabulate import tabulate

from cil.data_loader import DataLoader
from cil.evaluator import CVResult, Evaluator
from cil.normalizers import Normalizer
from cil.predictors import Predictor
from cil.metrics import Metric, rmse


@dataclass
class TuneResult:
    """Output of :func:`tune_with_holdout`: winning config + CV/holdout scores + predictions.

    Attributes:
        best_params: Full kwargs (fixed + winning grid point) used for the final fit.
        best_normalizer: Name of the normalizer paired with the winning config.
        cv_scores: Mean of each metric across CV folds, for the winning config.
        holdout_scores: Metric values on the held-out set after refitting on the full train.
        oof_predictions: Out-of-fold predictions from the winning config, columns
            ``sid, pid, rating_true, rating_pred``. One row per non-core train row (the
            coverage-preserving core — first row per sid/pid — is always in training and
            therefore never scored out-of-fold). Suitable as stacking features.
        holdout_predictions: Predictions on the holdout (same columns as ``oof_predictions``).
    """

    best_params: dict
    best_normalizer: str
    cv_scores: dict[str, float]
    holdout_scores: dict[str, float]
    oof_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame


def benchmark(
    data_loader: DataLoader,
    configs: dict[str, tuple[Callable[[], Predictor], Callable[[], Normalizer]]],
    n_folds: int = 5,
    seed: int = 42,
    metrics: dict[str, Metric] | None = None,
) -> dict[str, CVResult]:
    """Cross-validate named (predictor_factory, normalizer_factory) pairs.

    Args:
        data_loader: DataLoader instance.
        configs: ``{name: (predictor_factory, normalizer_factory), ...}``
        n_folds: Number of CV folds.
        seed: Random seed for splitting.
        metrics: Optional dict of metric name to Metric. If None, defaults to {"RMSE": rmse}.

    Returns:
        ``{name: CVResult, ...}``
    """
    metrics = metrics if metrics is not None else {"RMSE": rmse}
    results: dict[str, CVResult] = {}
    for name, (pred_factory, norm_factory) in configs.items():
        print(f"{name}:")
        results[name] = Evaluator.cross_validate(
            data_loader,
            normalizer_factory=norm_factory,
            predictor_factory=pred_factory,
            n_folds=n_folds,
            seed=seed,
            metrics=metrics,
        )

    _print_table(results)
    return results


def param_sweep(
    data_loader: DataLoader,
    predictor_cls: type[Predictor],
    param_grid: dict[str, list],
    normalizer_factory: Callable[[], Normalizer],
    fixed_params: dict | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, CVResult]:
    """Grid search over predictor hyperparameters.

    Args:
        data_loader: DataLoader instance.
        predictor_cls: Predictor class to instantiate.
        param_grid: ``{param_name: [value1, value2, ...], ...}``
        normalizer_factory: Normalizer factory for all runs.
        fixed_params: Extra keyword arguments passed to every predictor.
        n_folds: Number of CV folds.
        seed: Random seed for splitting.

    Returns:
        ``{label: CVResult, ...}``
    """
    fixed = fixed_params or {}
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs: dict[str, tuple[Callable[[], Predictor], Callable[[], Normalizer]]] = {}
    for combo in product(*values):
        swept = dict(zip(keys, combo))
        params = {**fixed, **swept}
        label = ", ".join(f"{k}={v}" for k, v in swept.items())
        configs[label] = (lambda p=params: predictor_cls(**p), normalizer_factory)

    return benchmark(data_loader, configs, n_folds=n_folds, seed=seed)


def tune_with_holdout(
    data_loader: DataLoader,
    predictor_cls: type[Predictor],
    search_space: dict[str, list],
    normalizers: Mapping[str, Callable[[], Normalizer]],
    fixed_params: dict | None = None,
    metrics: dict[str, Metric] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    selection_metric: str = "RMSE",
) -> TuneResult:
    """Grid-search via k-fold CV on the train partition, score the winner on a held-out set.

    The holdout is sampled once and never touched during tuning. The search runs over
    the cartesian product of ``normalizers`` × ``search_space``: for each (normalizer,
    grid point) pair the predictor (instantiated with ``fixed_params`` merged with the
    grid point) is cross-validated on the train folds with that normalizer. The pair
    with the best mean ``selection_metric`` is refit on the full train partition and
    scored once on the holdout. Winner's per-fold validation predictions are
    concatenated into ``oof_predictions`` so downstream stackers can use them as
    features without leaking labels.

    Args:
        data_loader: Source of ratings.
        predictor_cls: Predictor class to tune.
        search_space: ``{param_name: [values]}``. Empty dict means a single config
            consisting of ``fixed_params`` only.
        normalizers: ``{name: factory}`` mapping. Each factory is called to build a
            fresh normalizer for every fold and for the final refit. Pass a single-
            entry dict to fix the normalizer; pass multiple entries to compare them.
        fixed_params: Extra kwargs merged into every predictor instantiation. Overridden
            by values from ``search_space`` if the same key appears in both.
        metrics: ``{name: metric_fn}``. Defaults to ``{"RMSE": rmse}``.
        n_folds: K in k-fold CV.
        seed: Random seed for the train/holdout split and the CV split.
        holdout_fraction: Fraction of ratings reserved for final evaluation.
        selection_metric: Name of the metric whose CV mean is minimized to pick the
            winning config. Must be a key in ``metrics``.

    Returns:
        :class:`TuneResult` — see its docstring for field semantics.
    """
    if not normalizers:
        raise ValueError("normalizers must contain at least one entry")
    fixed = dict(fixed_params or {})
    active_metrics = metrics if metrics is not None else {"RMSE": rmse}
    if selection_metric not in active_metrics:
        raise ValueError(
            f"selection_metric {selection_metric!r} not in metrics {list(active_metrics)}"
        )

    if search_space:
        keys, values = zip(*search_space.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
    else:
        combinations = [{}]

    train_df, holdout_df, folds = data_loader.k_fold_held_out_split(
        n_folds=n_folds,
        seed=seed,
        holdout_fraction=holdout_fraction,
    )
    n_runs = len(normalizers) * len(combinations)
    print(
        f"Train/holdout split: {len(train_df)} train / {len(holdout_df)} holdout. "
        f"Searching {n_runs} config(s) "
        f"({len(normalizers)} normalizer × {len(combinations)} grid) "
        f"over {n_folds}-fold CV on train."
    )

    best_score = np.inf
    best_combo: dict = {}
    best_normalizer_name: str = ""
    best_cv_scores: dict[str, float] = {}
    best_oof: pd.DataFrame = pd.DataFrame()

    for norm_name, norm_factory in normalizers.items():
        for combo in combinations:
            params = {**fixed, **combo}
            grid_label = (
                ", ".join(f"{k}={v}" for k, v in combo.items()) or "<no grid params>"
            )
            print(f"\n[{predictor_cls.__name__}] norm={norm_name}, {grid_label}")
            cv_result = Evaluator.cross_validate(
                data_loader,
                normalizer_factory=norm_factory,
                predictor_factory=lambda p=params: predictor_cls(**p),
                metrics=active_metrics,
                folds=folds,
            )
            score = cv_result.mean[selection_metric]
            if score < best_score:
                best_score = score
                best_combo = combo
                best_normalizer_name = norm_name
                best_cv_scores = dict(cv_result.mean)
                best_oof = cv_result.oof_predictions

    best_params = {**fixed, **best_combo}
    print(
        f"\nBest {predictor_cls.__name__}: norm={best_normalizer_name}, {best_combo} "
        f"(CV {selection_metric}={best_score:.4f}). Refitting on full train partition..."
    )

    normalizer = normalizers[best_normalizer_name]()
    train_norm = normalizer.fit_transform(train_df)
    predictor = predictor_cls(**best_params)
    predictor.fit(
        train_norm,
        data_loader.num_scientists,
        data_loader.num_papers,
        wishlist=data_loader.wishlist,
    )
    holdout_eval = Evaluator(normalizer).evaluate(predictor, holdout_df, active_metrics, bootstrap=True)
    holdout_summary = ", ".join(f"{n}={v:.4f}" for n, v in holdout_eval.scores.items())
    print(f"Holdout: {holdout_summary}")

    return TuneResult(
        best_params=best_params,
        best_normalizer=best_normalizer_name,
        cv_scores=best_cv_scores,
        holdout_scores=holdout_eval.scores,
        oof_predictions=best_oof,
        holdout_predictions=holdout_eval.predictions,
    )


def _print_table(results: dict[str, CVResult]) -> None:
    metrics = []
    for r in results.values():
        for m in r.mean.keys():
            if m not in metrics:
                metrics.append(m)

    headers = ["Config"] + metrics

    rows = []
    for name, r in results.items():
        row = [name]
        for m in metrics:
            if m in r.mean:
                row.append(f"{r.mean[m]:.4f} (±{r.std[m]:.4f})")
            else:
                row.append("N/A")
        rows.append(row)

    rows.sort(key=lambda r: float(r[1].split()[0]) if r[1] != "N/A" else float("inf"))
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
