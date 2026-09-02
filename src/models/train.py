"""Dataset splitting utilities used before model training."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
TARGET_COLUMN = "SalePrice"


def split_features_and_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the regression target without mutating data."""
    if target_column not in data.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")
    if data[target_column].isna().any():
        raise ValueError("The target column contains missing values.")

    return data.drop(columns=[target_column]), data[target_column].copy()


def create_train_test_split(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split before learned transformations.

    Feature engineering, imputation, encoding, scaling, and model fitting must
    be fitted with ``X_train`` and ``y_train`` only. ``X_test`` and ``y_test``
    are reserved for final evaluation.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")

    features, target = split_features_and_target(data, target_column)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
