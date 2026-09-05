# Valttera — House Price Prediction

An end-to-end Machine Learning project that predicts Ames, Iowa residential sale prices from raw property attributes. It demonstrates a production-oriented workflow: data validation, reusable feature engineering, leakage-safe pipelines, model comparison/tuning, FastAPI serving, tests, and Docker packaging.

## Results

The selected model is a tuned `GradientBoostingRegressor` evaluated on the fixed holdout split.

| MAE | RMSE | R² |
| ---: | ---: | ---: |
| $16,263.20 | $27,266.73 | 0.903 |

The model is less reliable for high-value properties: the highest actual-price quartile has roughly $31.4k mean absolute error. It is an estimate, not a professional appraisal.

## Architecture

```text
raw CSV → validation/cleaning → processed data
        → feature engineering + preprocessing pipeline
        → tuned Gradient Boosting pipeline (.joblib)
        → FastAPI /predict endpoint
```

```text
app/                    FastAPI application and Pydantic schemas
data/raw/               Immutable source dataset (not tracked)
data/processed/         Derived cleaned dataset (not tracked)
docs/                   Phase decisions and results
models/                 Serialized deployment pipeline (not tracked)
notebooks/              EDA, feature engineering, model experiments
scripts/                Reproducible prepare/train commands
src/                    Reusable data, feature, model, and config logic
tests/                  Pytest suite
```

## Requirements

- Python 3.10–3.12 (Docker uses Python 3.11)
- The Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) training file, saved as `data/raw/train.csv`

## Setup and reproduction

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download `train.csv` from Kaggle into `data/raw/`, then run:

```bash
python -m scripts.prepare_data
python -m scripts.train_pipeline
```

The training command performs the bounded 20-candidate, five-fold `RandomizedSearchCV` on training data, refits the selected pipeline on all labeled rows, and writes `models/house_price_pipeline.joblib`.

## Run the API

```bash
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger documentation. The API accepts the raw Ames property fields and returns:

```json
{
  "predicted_price": 201860.14
}
```

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Model-load status |
| `POST` | `/predict` | Validated property-price prediction |

The serialized pipeline loads once at startup. It performs type normalization, feature engineering, imputation, encoding, scaling, and prediction as one operation.

## Testing

```bash
python -m pytest
```

The suite covers API health/prediction/validation behavior, preprocessing, and feature engineering.

## Docker

Ensure Docker Desktop is running, then:

```bash
docker build -t house-price-api .
docker run --rm -p 8000:8000 house-price-api
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs). The Docker image includes only serving code and the model artifact; raw data, notebooks, and tests are excluded.

## Key engineering decisions

- Raw data is never overwritten; derived data is stored separately.
- Learned steps are inside Scikit-learn pipelines and are fitted on training data only.
- `MSSubClass` is treated as categorical, and missing amenities remain distinguishable from zero measurements.
- Unknown API categories do not break predictions (`OneHotEncoder(handle_unknown="ignore")`).
- The complete pipeline—not a bare estimator—is serialized with Joblib.
- API errors are validated and safe for clients; server details are logged without stack traces in responses.

Further detail for each project phase is available in [`docs/`](docs/).
