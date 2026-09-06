import numpy as np
from scipy.optimize import nnls


def fit_nnls(Z, y):
    """Z: (N, M) OOF predictions, y: (N,) targets. Returns weights w >= 0 minimizing ||Z w - y||^2."""
    w, _ = nnls(Z, y)
    return w


def fit_nnls_normalized(Z, y, sum_to_one=False):
    w = fit_nnls(Z, y)
    if sum_to_one and w.sum() > 0:
        w = w / w.sum()
    return w


def blend(member_preds, weights):
    """member_preds: (N, M); weights: (M,). Returns (N,)."""
    return member_preds @ weights
