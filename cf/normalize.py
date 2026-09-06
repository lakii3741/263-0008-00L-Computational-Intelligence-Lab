import numpy as np


class Identity:
    def fit(self, df):
        return self

    def transform(self, df):
        return df

    def inverse(self, sids, pids, preds):
        return preds


class GlobalMean:
    def fit(self, df):
        self.mu = float(df["rating"].mean())
        return self

    def transform(self, df):
        out = df.copy()
        out["rating"] = df["rating"].to_numpy() - self.mu
        return out

    def inverse(self, sids, pids, preds):
        return preds + self.mu


class ScientistMean:
    """Per-user mean centering with shrinkage toward global."""

    def __init__(self, shrink=10.0):
        self.shrink = shrink

    def fit(self, df):
        sids = df["sid"].to_numpy()
        r = df["rating"].to_numpy(dtype=np.float64)
        self.mu = float(r.mean())
        n_users = int(sids.max()) + 1
        sums = np.zeros(n_users)
        cnts = np.zeros(n_users)
        np.add.at(sums, sids, r - self.mu)
        np.add.at(cnts, sids, 1.0)
        self.bias = sums / (cnts + self.shrink)
        return self

    def transform(self, df):
        out = df.copy()
        sids = df["sid"].to_numpy()
        out["rating"] = df["rating"].to_numpy() - self.mu - self.bias[sids]
        return out

    def inverse(self, sids, pids, preds):
        b = np.zeros_like(preds)
        valid = sids < len(self.bias)
        b[valid] = self.bias[sids[valid]]
        return preds + self.mu + b


class PaperMean:
    def __init__(self, shrink=25.0):
        self.shrink = shrink

    def fit(self, df):
        pids = df["pid"].to_numpy()
        r = df["rating"].to_numpy(dtype=np.float64)
        self.mu = float(r.mean())
        n_items = int(pids.max()) + 1
        sums = np.zeros(n_items)
        cnts = np.zeros(n_items)
        np.add.at(sums, pids, r - self.mu)
        np.add.at(cnts, pids, 1.0)
        self.bias = sums / (cnts + self.shrink)
        return self

    def transform(self, df):
        out = df.copy()
        pids = df["pid"].to_numpy()
        out["rating"] = df["rating"].to_numpy() - self.mu - self.bias[pids]
        return out

    def inverse(self, sids, pids, preds):
        b = np.zeros_like(preds)
        valid = pids < len(self.bias)
        b[valid] = self.bias[pids[valid]]
        return preds + self.mu + b
