"""
Split the feature table by season into train / validation / test.

CP4, CP5 and CP6 all import from here, so every model is trained and
judged on exactly the same plays.
"""

import pandas as pd
import polars as pl

from src.features import FEATURES_PATH, FEATURE_COLS, LABEL_COL

# The split boundaries live in ONE place. Change them here, and every
# model that imports this file picks up the change automatically.
TRAIN_END = 2023     # train on 2010 through 2023
VAL_SEASON = 2024    # tune and compare on 2024
TEST_SEASON = 2025   # touch ONCE, in CP6


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames split by season."""
    # scikit-learn works with pandas, so convert from polars here.
    df = pl.read_parquet(FEATURES_PATH).to_pandas()

    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    return train, val, test


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the label (y).

    X and y are the standard ML names: X is the inputs (a table),
    y is what we're trying to predict (one column).
    """
    return df[FEATURE_COLS], df[LABEL_COL]