"""Raw-data validation and deterministic cleaning for Ames housing data.

Learned transformations such as median imputation and one-hot encoding do not
belong here.  They are fitted on training data only in the modelling pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.engineering import add_engineered_features


TARGET_COLUMN = "SalePrice"
IDENTIFIER_COLUMN = "Id"
HIGH_CARDINALITY_THRESHOLD = 50

# These raw components are represented by interpretable engineered aggregates.
# They remain in the processed dataset, but are excluded from the model matrix
# to avoid exact linear dependencies such as TotalSF = basement + floor areas.
REDUNDANT_AFTER_ENGINEERING = {
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
}


@dataclass(frozen=True)
class DataValidationReport:
    """Summary of data-quality checks that do not change the source data."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_values: dict[str, int]
    constant_columns: list[str]
    high_cardinality_categoricals: dict[str, int]
    negative_numeric_values: dict[str, int]
    impossible_values: dict[str, int]
    iqr_outliers: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation for a report or log."""
        return asdict(self)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read a raw CSV without altering it."""
    return pd.read_csv(path)


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Read processed data while preserving categorical dwelling-class codes."""
    return pd.read_csv(path, dtype={"MSSubClass": "string"})


def normalize_model_input(features: pd.DataFrame) -> pd.DataFrame:
    """Apply schema normalizations needed by both training and inference."""
    normalized = features.copy().where(features.notna(), np.nan)
    if "MSSubClass" in normalized.columns:
        normalized["MSSubClass"] = normalized["MSSubClass"].astype("string")
    return normalized


def validate_housing_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
) -> DataValidationReport:
    """Inspect common quality risks in the Ames training data.

    The function reports observations; it deliberately does not remove
    potential outliers because their legitimacy needs domain and EDA review.
    """
    if target_column not in data.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")

    missing_values = (
        data.isna().sum().loc[lambda values: values > 0].sort_values(ascending=False).to_dict()
    )
    constant_columns = data.nunique(dropna=False).loc[lambda values: values <= 1].index.tolist()

    categorical = data.select_dtypes(exclude="number")
    high_cardinality = (
        categorical.nunique(dropna=True)
        .loc[lambda values: values > high_cardinality_threshold]
        .sort_values(ascending=False)
        .to_dict()
    )

    numeric = data.select_dtypes(include="number")
    negative_values = (
        (numeric < 0).sum().loc[lambda values: values > 0].sort_values(ascending=False).to_dict()
    )
    impossible_values = _find_impossible_values(data, target_column)
    iqr_outliers = _count_iqr_outliers(numeric.drop(columns=[target_column], errors="ignore"))

    return DataValidationReport(
        rows=len(data),
        columns=len(data.columns),
        duplicate_rows=int(data.duplicated().sum()),
        missing_values={column: int(count) for column, count in missing_values.items()},
        constant_columns=constant_columns,
        high_cardinality_categoricals={column: int(count) for column, count in high_cardinality.items()},
        negative_numeric_values={column: int(count) for column, count in negative_values.items()},
        impossible_values=impossible_values,
        iqr_outliers=iqr_outliers,
    )


def clean_housing_data(data: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Apply only deterministic, non-learned cleaning steps.

    Exact duplicate records are removed, the identifier is excluded from model
    features, and ``MSSubClass`` is represented as categorical data because its
    values encode dwelling classes rather than numeric magnitude. Missing values
    are intentionally retained for the train-fitted preprocessing pipeline.
    """
    if target_column not in data.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")

    cleaned = data.drop_duplicates().copy()
    cleaned = cleaned.drop(columns=[IDENTIFIER_COLUMN], errors="ignore")

    if "MSSubClass" in cleaned.columns:
        cleaned["MSSubClass"] = cleaned["MSSubClass"].astype("Int64").astype("string")

    return cleaned


def save_processed_data(data: pd.DataFrame, path: str | Path) -> Path:
    """Write cleaned data to the processed layer, never to the raw layer."""
    destination = Path(path)
    if "raw" in destination.parts:
        raise ValueError("Processed data must not be written under a raw-data path.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return destination


def build_preprocessing_pipeline(
    features: pd.DataFrame,
    selected_feature_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Create an unfitted pipeline for raw Ames property features.

    ``features`` supplies only the input schema. Fitted statistics, including
    numerical medians, category levels, and scaling parameters, are learned
    later when this pipeline is fitted on ``X_train`` only.
    """
    _require_feature_columns(features)
    feature_sample = add_engineered_features(features.iloc[:0].copy())
    if selected_feature_columns is not None:
        missing_columns = sorted(set(selected_feature_columns).difference(feature_sample.columns))
        if missing_columns:
            missing_display = ", ".join(missing_columns)
            raise ValueError(f"Unknown selected feature columns: {missing_display}.")
        feature_sample = feature_sample.loc[:, list(selected_feature_columns)]
    numerical_columns = [
        column
        for column in feature_sample.select_dtypes(include="number").columns
        if column not in REDUNDANT_AFTER_ENGINEERING
    ]
    categorical_columns = feature_sample.select_dtypes(exclude="number").columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("input_normalization", FunctionTransformer(normalize_model_input, validate=False)),
            (
                "feature_engineering",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocessor", column_transformer),
        ]
    )


def _find_impossible_values(data: pd.DataFrame, target_column: str) -> dict[str, int]:
    """Check dataset constraints whose violations are unambiguously invalid."""
    checks: dict[str, pd.Series] = {
        target_column: data[target_column].le(0),
    }
    if "MoSold" in data:
        checks["MoSold"] = ~data["MoSold"].between(1, 12)
    for column in ("OverallQual", "OverallCond"):
        if column in data:
            checks[column] = ~data[column].between(1, 10)
    if {"YearBuilt", "YearRemodAdd"}.issubset(data.columns):
        checks["YearRemodAdd_before_YearBuilt"] = data["YearRemodAdd"] < data["YearBuilt"]

    return {name: int(mask.fillna(False).sum()) for name, mask in checks.items() if mask.fillna(False).any()}


def _count_iqr_outliers(numeric_data: pd.DataFrame) -> dict[str, int]:
    """Flag observations beyond 1.5 IQR for investigation, not deletion."""
    outlier_counts: dict[str, int] = {}
    for column in numeric_data.columns:
        series = numeric_data[column].dropna()
        first_quartile, third_quartile = series.quantile([0.25, 0.75])
        interquartile_range = third_quartile - first_quartile
        if interquartile_range == 0:
            continue
        count = ((series < first_quartile - 1.5 * interquartile_range) | (series > third_quartile + 1.5 * interquartile_range)).sum()
        if count:
            outlier_counts[column] = int(count)
    return outlier_counts


def _require_feature_columns(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("Cannot build a preprocessing pipeline from an empty feature set.")
    if TARGET_COLUMN in features.columns:
        raise ValueError("Features must not include the target column 'SalePrice'.")
