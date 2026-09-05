"""Shared fixtures for API tests."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data.preprocessing import load_processed_data
from src.models.train import split_features_and_target


@pytest.fixture
def client() -> TestClient:
    """Start the application lifespan so the model loads as it would in production."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_prediction_payload() -> dict[str, object]:
    """Build one schema-valid raw property record from the processed dataset."""
    features, _ = split_features_and_target(load_processed_data("data/processed/train_clean.csv"))
    payload = {
        column: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for column, value in features.iloc[0].items()
    }
    payload["MSSubClass"] = int(payload["MSSubClass"])
    return payload
