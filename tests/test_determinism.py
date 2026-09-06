import numpy as np

from cil import DataLoader
from cil.predictors.baselines import ALSPredictor, SVDppPredictor


def _fit(predictor, train_df, n_users, n_papers):
    predictor.fit(train_df, n_users, n_papers)
    sids = train_df["sid"].values[:1000]
    pids = train_df["pid"].values[:1000]
    return predictor.predict(sids, pids)


def test_als_bit_exact(train_df, n_users, n_papers):
    pred_a = ALSPredictor(k=8, reg=0.1, n_iters=3, seed=42)
    pred_b = ALSPredictor(k=8, reg=0.1, n_iters=3, seed=42)

    out_a = _fit(pred_a, train_df, n_users, n_papers)
    out_b = _fit(pred_b, train_df, n_users, n_papers)

    np.testing.assert_array_equal(pred_a._U, pred_b._U)
    np.testing.assert_array_equal(pred_a._V, pred_b._V)
    np.testing.assert_array_equal(out_a, out_b)
    print("ALSPredictor: bit-exact reproducibility PASSED")


def test_svdpp_bit_exact(train_df, n_users, n_papers):
    pred_a = SVDppPredictor(k=8, lr=0.007, reg=0.05, n_epochs=2, seed=42)
    pred_b = SVDppPredictor(k=8, lr=0.007, reg=0.05, n_epochs=2, seed=42)

    out_a = _fit(pred_a, train_df, n_users, n_papers)
    out_b = _fit(pred_b, train_df, n_users, n_papers)

    np.testing.assert_array_equal(pred_a._Q, pred_b._Q)
    np.testing.assert_array_equal(pred_a._P_hat, pred_b._P_hat)
    np.testing.assert_array_equal(pred_a._bu, pred_b._bu)
    np.testing.assert_array_equal(pred_a._bi, pred_b._bi)
    np.testing.assert_array_equal(out_a, out_b)
    print("SVDppPredictor: bit-exact reproducibility PASSED")


def test_seed_actually_matters(train_df, n_users, n_papers):
    """Sanity check: different seeds must produce different factors."""
    pred_42 = ALSPredictor(k=8, reg=0.1, n_iters=3, seed=42)
    pred_43 = ALSPredictor(k=8, reg=0.1, n_iters=3, seed=43)
    pred_42.fit(train_df, n_users, n_papers)
    pred_43.fit(train_df, n_users, n_papers)

    assert not np.array_equal(pred_42._U, pred_43._U), (
        "Different seeds produced identical factors"
    )
    print("Seed propagation: different seeds → different factors PASSED")


if __name__ == "__main__":
    loader = DataLoader("data")
    train_df, _ = loader.split(seed=42)
    n_users, n_papers = loader.num_scientists, loader.num_papers

    test_als_bit_exact(train_df, n_users, n_papers)
    test_svdpp_bit_exact(train_df, n_users, n_papers)
    test_seed_actually_matters(train_df, n_users, n_papers)

    print("\n=== ALL DETERMINISM TESTS PASSED ===")
