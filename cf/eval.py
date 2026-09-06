import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(predictor, val_df):
    """predictor is a callable (sids, pids) -> ratings, or has .predict(sids, pids)."""
    pred_fn = predictor if callable(predictor) else predictor.predict
    preds = pred_fn(val_df["sid"].to_numpy(), val_df["pid"].to_numpy())
    preds = np.clip(preds, 1.0, 5.0)
    return rmse(val_df["rating"].to_numpy(), preds)


def make_submission(predictor, data, path):
    pred_fn = predictor if callable(predictor) else predictor.predict
    sub = data.submission_template()
    preds = pred_fn(data.test_sids, data.test_pids)
    sub["rating"] = np.clip(preds, 1.0, 5.0)
    sub.to_csv(path, index=False)
    return path
