"""FastAPI application serving the serialized house-price pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status

from app.schemas import ApplicationInfo, HealthResponse, PredictionResponse, PropertyFeatures
from src.models.predict import load_pipeline, predict_single_price


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "house_price_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline once per application process."""
    try:
        app.state.model = load_pipeline(MODEL_PATH)
        app.state.model_load_error = None
    except (FileNotFoundError, OSError, ValueError) as error:
        app.state.model = None
        app.state.model_load_error = str(error)
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predict Ames residential property sale prices using a trained ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=ApplicationInfo, tags=["Service"])
def root() -> ApplicationInfo:
    """Return basic service metadata."""
    return ApplicationInfo(name=app.title, version=app.version, docs_url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health(request: Request) -> HealthResponse:
    """Report whether the application loaded its prediction artifact."""
    model_loaded = request.app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict a residential property sale price",
)
def predict(property_features: PropertyFeatures, request: Request) -> PredictionResponse:
    """Predict one price from validated raw property attributes."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable.",
        )

    try:
        predicted_price = predict_single_price(
            model,
            property_features.model_dump(by_alias=True),
        )
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The property data could not be used for prediction.",
        ) from error

    return PredictionResponse(predicted_price=round(predicted_price, 2))
