"""Test that transform followed by inverse_transform is the identity for all normalizers."""

import numpy as np

from cil import DataLoader
from cil.normalizers.baselines import (
    IdentityNormalizer,
    PaperNormalizer,
    ScientistNormalizer,
)

ALL_NORMALIZERS = [IdentityNormalizer, PaperNormalizer, ScientistNormalizer]


def test_round_trip():
    loader = DataLoader("data")
    train_df, val_df = loader.split(seed=42)

    for cls in ALL_NORMALIZERS:
        norm = cls()
        norm.fit(train_df)

        for label, df in [("train", train_df), ("val", val_df)]:
            original = df["rating"].values.astype(np.float64)
            transformed = norm.transform(df)
            recovered = norm.inverse_transform(
                df["sid"].values, df["pid"].values, transformed["rating"].values
            )
            np.testing.assert_allclose(
                recovered,
                original,
                atol=1e-10,
                err_msg=f"{cls.__name__} round-trip failed on {label} split",
            )

        print(f"{cls.__name__}: round-trip PASSED")


if __name__ == "__main__":
    test_round_trip()
    print("\n=== ALL NORMALIZER TESTS PASSED ===")
