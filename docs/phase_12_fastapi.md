# Phase 12 — FastAPI application

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name, version, and Swagger documentation link. |
| `GET` | `/health` | Whether the serialized model loaded successfully. |
| `POST` | `/predict` | Validates a raw Ames property record and returns `predicted_price`. |

## Application lifecycle

FastAPI's lifespan hook loads `models/house_price_pipeline.joblib` once when each application process starts. The prediction endpoint retrieves that in-memory pipeline; it does not load a file per request.

`PropertyFeatures` describes all raw fields expected by the model, rejects unexpected fields, and applies basic domain validation such as positive areas, quality score ranges, and a valid sale month. The Pydantic schema also generates the interactive request documentation at `/docs`.

## Run locally

```bash
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs`.
