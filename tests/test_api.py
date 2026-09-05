"""Integration tests for public API behavior."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient


def test_health_endpoint_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_a_finite_positive_price(
    client: TestClient,
    valid_prediction_payload: dict[str, object],
) -> None:
    response = client.post("/predict", json=valid_prediction_payload)

    assert response.status_code == 200
    predicted_price = response.json()["predicted_price"]
    assert isinstance(predicted_price, float)
    assert math.isfinite(predicted_price)
    assert predicted_price > 0


def test_predict_rejects_invalid_property_value(
    client: TestClient,
    valid_prediction_payload: dict[str, object],
) -> None:
    response = client.post(
        "/predict",
        json={**valid_prediction_payload, "LotArea": -1},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Request validation failed."
    assert body["errors"] == [
        {"field": "body.LotArea", "message": "Input should be greater than 0"}
    ]
