# Phase 11 — Model serialization

## Deployment artifact

`models/house_price_pipeline.joblib` contains one fitted Scikit-learn pipeline with:

1. Input type normalization (`MSSubClass` is always categorical).
2. Deterministic feature engineering.
3. Numerical imputation and scaling.
4. Categorical imputation and one-hot encoding.
5. The tuned `GradientBoostingRegressor`.

Saving the full pipeline prevents a common production bug: preprocessing training data one way and API inputs another way. The API will pass raw property fields to this artifact; it must not manually reproduce any transformation.

The input-normalization step also converts JSON `null` values to pandas missing values before imputation. This ensures optional API fields use the same missing-value treatment as the training data.

## Refit decision

Hyperparameters were selected using training-only cross-validation. After final evaluation, the selected pipeline is cloned and refitted on all 1,460 labeled rows, giving the deployment artifact the largest permitted training set. The fixed test partition is not used to choose parameters.

## Inference helpers

`src/models/predict.py` owns artifact loading and prediction:

- `save_pipeline()` persists the fitted pipeline.
- `load_pipeline()` loads it once and gives a clear error if absent.
- `predict_prices()` handles a DataFrame of raw property records.
- `predict_single_price()` handles one property dictionary and returns a Python float.
