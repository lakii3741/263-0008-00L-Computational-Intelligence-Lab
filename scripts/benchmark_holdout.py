# ruff: noqa: E402
# (BLAS thread-count env vars below must be set before numpy is imported, which
#  forces the rest of the imports to follow that block.)
"""Tune one or more predictors via k-fold CV on the train partition + held-out scoring.

For each selected model, reads ``config/<model>.yaml`` (schema below), runs a grid
search via ``cil.benchmark.tune_with_holdout``, pickles the winning config plus
CV/holdout scores and OOF/holdout predictions to ``<results_dir>/<model>_results.pkl``,
also writes a JSON sibling ``<results_dir>/<model>_results.json`` with the scalar
fields for easy inspection (predictions DataFrames remain only in the pickle), and
prints a final comparison table sorted by holdout score on the chosen metric.

Models are tuned in parallel via a process pool (one model per worker, capped at
``min(num_models, os.cpu_count())``). BLAS thread counts are pinned to 1 in workers
to avoid CPU oversubscription. Each worker's stdout is streamed line-by-line to
the parent through a shared queue and printed with a ``[<model>]`` prefix, so
progress from every model is visible in real time (lines from different models
interleave in the console but are unambiguously tagged).

YAML schema:
    search_space:     dict[str, list]  # cartesian-product grid; may be empty or omitted
    fixed_params:     dict             # kwargs held constant across all grid points
    normalizer:       str | list[str]  # identity | paper | scientist (default: identity).
                                       # A list searches over normalizers as an extra axis.
    n_folds:          int              # default: 5
    holdout_fraction: float            # default: 0.2
    seed:             int              # default: 42
    selection_metric: str              # metric key driving "best config" pick (default: RMSE)

Each pickle holds a dict:
    {
        "model": str,
        "best_params": dict,
        "best_normalizer": str,
        "cv_scores": {metric: float},
        "holdout_scores": {metric: float},
        "oof_predictions": DataFrame[sid, pid, rating_true, rating_pred],
        "holdout_predictions": DataFrame[sid, pid, rating_true, rating_pred],
    }
The JSON sibling holds the same dict minus the two predictions DataFrames.

Examples:
    uv run python scripts/benchmark_holdout.py                            # all models
    uv run python scripts/benchmark_holdout.py --models itemknn svd als   # subset
    uv run python scripts/benchmark_holdout.py --models svd               # one model
    uv run python scripts/benchmark_holdout.py --sort_by MAE              # sort key
"""

import os

# Pin BLAS thread counts BEFORE numpy is (transitively) imported, so workers
# spawned by ProcessPoolExecutor — which re-import this module — inherit
# single-threaded BLAS and don't oversubscribe the CPU under the model-level pool.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import contextlib
import json
import multiprocessing as mp
import pickle
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tabulate import tabulate

from cil import DataLoader, TuneResult, tune_with_holdout
from cil.metrics import mae, rmse
from cil.normalizers import IdentityNormalizer, Normalizer
from cil.normalizers.baselines import PaperNormalizer, ScientistNormalizer
from cil.predictors import Predictor
from cil.predictors.baselines import (
    GlobalMeanPredictor,
    ItemMeanPredictor,
    UserMeanPredictor,
    SlopeOnePredictor,
    ALSPredictor,
    ItemKNNPredictor,
    IterativeSVDPredictor,
    KNNBasicPredictor,
    KNNWithMeansPredictor,
    KNNWithZScorePredictor,
    SVDppPredictor,
    SVDPredictor,
    BiasedSVDPredictor,
    NeuMFPredictor,
    DMFPredictor,
    EASEPredictor,
)

MODEL_MAP: dict[str, type[Predictor]] = {
    "globalmean": GlobalMeanPredictor,
    "itemmean": ItemMeanPredictor,
    "usermean": UserMeanPredictor,
    "slopeone": SlopeOnePredictor,
    "ease": EASEPredictor,
    "itemknn": ItemKNNPredictor,
    "svd": SVDPredictor,
    "iter_svd": IterativeSVDPredictor,
    "biasedsvd": BiasedSVDPredictor,
    "als": ALSPredictor,
    "svdpp": SVDppPredictor,
    "knnbasic_item": KNNBasicPredictor,
    "knn_with_means": KNNWithMeansPredictor,
    "knn_with_zscore": KNNWithZScorePredictor,
    "neumf": NeuMFPredictor,
    "neumf_wl": NeuMFPredictor,
    "dmf": DMFPredictor,
    "dmf_wl": DMFPredictor,
}

NORMALIZER_MAP: dict[str, type[Normalizer]] = {
    "identity": IdentityNormalizer,
    "paper": PaperNormalizer,
    "scientist": ScientistNormalizer,
}

METRICS = {"RMSE": rmse, "MAE": mae}


def tune_one(model: str, data_dir: str) -> TuneResult:
    """Run `tune_with_holdout` for a single model using its `config/<model>.yaml`."""
    config_path = Path("config") / f"{model}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    norm_field = config.get("normalizer", "identity")
    norm_names = [norm_field] if isinstance(norm_field, str) else list(norm_field)
    unknown = [n for n in norm_names if n not in NORMALIZER_MAP]
    if unknown:
        raise ValueError(
            f"Unknown normalizer(s) {unknown!r}; expected from {sorted(NORMALIZER_MAP)}"
        )
    normalizers = {n: NORMALIZER_MAP[n] for n in norm_names}

    return tune_with_holdout(
        data_loader=DataLoader(data_dir),
        predictor_cls=MODEL_MAP[model],
        search_space=config.get("search_space") or {},
        normalizers=normalizers,
        fixed_params=config.get("fixed_params") or {},
        metrics=METRICS,
        n_folds=int(config.get("n_folds", 5)),
        seed=int(config.get("seed", 42)),
        holdout_fraction=float(config.get("holdout_fraction", 0.2)),
        selection_metric=config.get("selection_metric", "RMSE"),
    )


def save_result(result: TuneResult, model: str, results_dir: str) -> Path:
    """Pickle a TuneResult to ``<results_dir>/<model>_results.pkl`` and return the path."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "best_params": result.best_params,
        "best_normalizer": result.best_normalizer,
        "cv_scores": result.cv_scores,
        "holdout_scores": result.holdout_scores,
        "oof_predictions": result.oof_predictions,
        "holdout_predictions": result.holdout_predictions,
    }
    out_path = output_dir / f"{model}_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path


def _jsonable(obj: object) -> object:
    """JSON ``default`` hook converting numpy scalars to native Python."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")


def save_result_json(result: TuneResult, model: str, results_dir: str) -> Path:
    """Write the JSON-serializable summary of a TuneResult.

    Output: ``<results_dir>/<model>_results.json`` with the same fields as the
    pickle except the two predictions DataFrames (which stay only in the pickle
    since they are bulky stacking features, not human-inspection material).
    """
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "best_params": result.best_params,
        "best_normalizer": result.best_normalizer,
        "cv_scores": result.cv_scores,
        "holdout_scores": result.holdout_scores,
    }
    out_path = output_dir / f"{model}_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_jsonable)
    return out_path


_LOG_QUEUE: Any = None  # populated per-worker by _worker_init


class _PrefixedQueueWriter:
    """File-like writer that buffers stdout into newline-terminated lines and pushes
    ``[<prefix>] <line>`` onto a shared log queue for parent-side draining.
    """

    def __init__(self, queue: Any, prefix: str) -> None:
        self._queue = queue
        self._prefix = prefix
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._queue.put(f"[{self._prefix}] {line}\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._queue.put(f"[{self._prefix}] {self._buf}")
            self._buf = ""


def _worker_init(queue: Any) -> None:
    """Pool initializer: stash the parent-side log queue in a module-level global."""
    global _LOG_QUEUE  # noqa: PLW0603
    _LOG_QUEUE = queue


def _drain_log_queue(queue: Any) -> None:
    """Parent-side drain: print tagged items from the queue until a ``None`` sentinel arrives."""
    while True:
        item = queue.get()
        if item is None:
            return
        sys.stdout.write(item)
        sys.stdout.flush()


def _tune_and_save(
    model: str, data_dir: str, results_dir: str
) -> tuple[str, str, dict[str, float], dict[str, float], dict]:
    """Worker entrypoint: tune one model and persist its artifacts.

    Stdout is streamed line-by-line to the shared parent-side log queue with a
    ``[<model>]`` prefix so progress is visible live; lines from different
    models interleave in the parent console but are unambiguously tagged.
    """
    assert _LOG_QUEUE is not None, "Worker not initialized with log queue"
    writer = _PrefixedQueueWriter(_LOG_QUEUE, model)
    try:
        with contextlib.redirect_stdout(writer):
            print(f"=== {model} starting ===")
            result = tune_one(model, data_dir)
            pkl_path = save_result(result, model, results_dir)
            json_path = save_result_json(result, model, results_dir)
            print(f"Saved {pkl_path}")
            print(f"Saved {json_path}")
    finally:
        writer.flush()
    return (
        model,
        result.best_normalizer,
        result.cv_scores,
        result.holdout_scores,
        result.best_params,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_MAP),
        default=sorted(MODEL_MAP),
        help="Subset of models to benchmark (default: all).",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--sort_by",
        type=str,
        default="RMSE",
        choices=sorted(METRICS),
        help="Metric to sort the final comparison table by (ascending).",
    )
    args = parser.parse_args()

    n_workers = min(len(args.models), os.cpu_count() or 1)
    rows: list[tuple[str, str, dict[str, float], dict[str, float], dict]] = []
    with mp.Manager() as manager:
        log_queue = manager.Queue()
        drain = threading.Thread(
            target=_drain_log_queue, args=(log_queue,), daemon=True
        )
        drain.start()
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(log_queue,),
            ) as pool:
                futures = {
                    pool.submit(_tune_and_save, m, args.data_dir, args.results_dir): m
                    for m in args.models
                }
                for fut in as_completed(futures):
                    rows.append(fut.result())
        finally:
            log_queue.put(None)
            drain.join()

    metric_names = list(METRICS.keys())
    headers = (
        ["Model", "Normalizer"]
        + [f"CV {m}" for m in metric_names]
        + [f"Bootstrapped Holdout {m} (± 95% CI)" for m in metric_names]
        + ["Best params"]
    )
    
    table = []
    for model, normalizer, cv_scores, holdout_scores, best_params in rows:
        holdout_cells = []
        for m in metric_names:
            point = holdout_scores.get(m, float("nan"))
            err = holdout_scores.get(f"bootstrap_{m}_error")
            if err is not None:
                holdout_cells.append(f"{point:.4f} ± {err:.4f}")
            else:
                holdout_cells.append(f"{point:.4f}")
        table.append(
            [model, normalizer]
            + [f"{cv_scores.get(m, float('nan')):.4f}" for m in metric_names]
            + holdout_cells
            + [", ".join(f"{k}={v}" for k, v in best_params.items())]
        )
    holdout_col = 2 + len(metric_names) + metric_names.index(args.sort_by)
    table.sort(key=lambda r: float(r[holdout_col].split()[0]))

    print(
        f"\n{'=' * 70}\n=== Comparison (sorted by Holdout {args.sort_by})\n{'=' * 70}"
    )
    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
