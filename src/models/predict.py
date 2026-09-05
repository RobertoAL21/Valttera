"""Serialization and inference helpers for the complete house-price pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def save_pipeline(pipeline: Any, path: str | Path) -> Path:
    """Persist a complete fitted preprocessing-and-model pipeline with Joblib."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def load_pipeline(path: str | Path) -> Any:
    """Load a previously serialized fitted pipeline."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {source}")
    return joblib.load(source)


def predict_prices(pipeline: Any, properties: pd.DataFrame) -> np.ndarray:
    """Predict prices from raw property fields using the embedded pipeline."""
    predictions = np.asarray(pipeline.predict(properties), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Pipeline produced non-finite price predictions.")
    return predictions


def predict_single_price(pipeline: Any, property_data: Mapping[str, object]) -> float:
    """Predict one price from a raw property record."""
    prediction = predict_prices(pipeline, pd.DataFrame([property_data]))
    if len(prediction) != 1:
        raise ValueError("Expected exactly one prediction for one property record.")
    return float(prediction[0])
