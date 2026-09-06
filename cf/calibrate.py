import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(val_pred, val_true, lo=1.0, hi=5.0):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=lo, y_max=hi)
    ir.fit(val_pred, val_true)
    return ir


def apply_isotonic(ir, preds, lo=1.0, hi=5.0):
    return np.clip(ir.predict(preds), lo, hi)
