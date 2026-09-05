"""FastAPI application serving the serialized house-price pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import ApplicationInfo, HealthResponse, PredictionResponse, PropertyFeatures
from src.config import MODEL_PATH
from src.models.predict import load_pipeline, predict_single_price


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline once per application process."""
    try:
        app.state.model = load_pipeline(MODEL_PATH)
        app.state.model_load_error = None
    except Exception as error:
        logger.exception("Unable to load the prediction model during application startup.")
        app.state.model = None
        app.state.model_load_error = str(error)
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predict Ames residential property sale prices using a trained ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Return concise, field-level validation errors without echoing input values."""
    errors = [
        {
            "field": ".".join(str(location) for location in issue["loc"]),
            "message": issue["msg"],
        }
        for issue in error.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Request validation failed.", "errors": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, error: Exception) -> JSONResponse:
    """Prevent unexpected internal details from being exposed to API clients."""
    logger.exception("Unexpected prediction service error.", exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred."},
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The property data could not be used for prediction.",
        ) from error

    return PredictionResponse(predicted_price=round(predicted_price, 2))
