"""Consistent regression evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@dataclass(frozen=True)
class RegressionMetrics:
    """Metrics in the original dollar scale of the house-price target."""

    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_regression_metrics(
    y_true: pd.Series,
    predictions: pd.Series | Any,
) -> RegressionMetrics:
    """Calculate MAE, RMSE, and R² for a set of regression predictions."""
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=float(root_mean_squared_error(y_true, predictions)),
        r2=float(r2_score(y_true, predictions)),
    )


def evaluate_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate fitted models on one common holdout set."""
    results = []
    for name, model in models.items():
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            predictions = model.predict(X_test)
        if not np.isfinite(predictions).all():
            raise ValueError(f"Model '{name}' produced non-finite predictions.")
        metrics = calculate_regression_metrics(y_test, predictions)
        results.append({"Model": name, **metrics.as_dict()})

    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
