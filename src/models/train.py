"""Dataset splitting utilities used before model training."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessing_pipeline


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
TARGET_COLUMN = "SalePrice"

# A compact, domain-informed set for the untuned linear benchmark. It avoids
# unstable coefficient estimates caused by a very wide, sparse feature matrix.
BASELINE_FEATURE_COLUMNS = (
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "LotArea",
    "TotalSF",
    "TotalBathrooms",
    "HouseAge",
    "YearsSinceRemodel",
    "TotalOutdoorSpace",
    "OverallQualGrLivArea",
    "Neighborhood",
    "MSZoning",
    "BldgType",
    "HouseStyle",
    "KitchenQual",
    "ExterQual",
    "Foundation",
    "GarageType",
    "BsmtQual",
)


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


def build_regression_pipeline(
    features: pd.DataFrame,
    estimator: RegressorMixin,
    selected_feature_columns: tuple[str, ...] | None = None,
) -> Pipeline:
    """Combine feature processing and a regressor into one fitted artifact."""
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessing_pipeline(features, selected_feature_columns),
            ),
            ("model", estimator),
        ]
    )


def train_baseline_and_linear_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit the untuned baseline and linear-regression candidates on training data."""
    candidates: dict[str, RegressorMixin] = {
        "DummyRegressor (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
    }
    return {
        name: build_regression_pipeline(
            X_train,
            estimator,
            selected_feature_columns=BASELINE_FEATURE_COLUMNS,
        ).fit(X_train, y_train)
        for name, estimator in candidates.items()
    }


def train_model_comparison_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit untuned linear and tree-based candidates on one training partition."""
    candidates: dict[str, tuple[RegressorMixin, tuple[str, ...] | None]] = {
        "LinearRegression": (LinearRegression(), BASELINE_FEATURE_COLUMNS),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=-1),
            None,
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE),
            None,
        ),
    }
    return {
        name: build_regression_pipeline(
            X_train,
            estimator,
            selected_feature_columns=selected_columns,
        ).fit(X_train, y_train)
        for name, (estimator, selected_columns) in candidates.items()
    }
