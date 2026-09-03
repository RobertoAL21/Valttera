# Phase 6 — Unified preprocessing pipeline

## Pipeline flow

```text
Raw property features
  -> deterministic feature engineering
  -> numerical branch: median imputation -> standard scaling
  -> categorical branch: explicit missing category -> one-hot encoding
  -> model-ready numerical matrix
```

`build_preprocessing_pipeline()` returns an unfitted Scikit-learn `Pipeline`. Its `features` argument supplies the schema only. During the next phase, it will be fitted on `X_train` only.

## Numerical features

Numerical fields—including the engineered numeric fields—use `SimpleImputer(strategy="median")`. The median is robust to the skew and outliers identified in EDA. `StandardScaler` then centers and scales those fields. Scaling is important for linear regression and harmless for the tree-based models in this small project because all models will receive the same prepared inputs.

## Categorical features

Categorical fields use a constant `"Missing"` value before `OneHotEncoder(handle_unknown="ignore")`.

- A constant missing category preserves the fact that a value was unavailable or an amenity absent.
- `handle_unknown="ignore"` ensures the API can receive a valid new category without failing at prediction time; its unseen encoded columns simply remain zero.

`sparse_output=False` intentionally produces a dense matrix. The Ames data has manageable categorical cardinality, and `GradientBoostingRegressor` requires dense input.

## Leakage protection

The pipeline does not call `fit` in this phase. Fitting median values, scaler statistics, and known category levels occurs later with `pipeline.fit(X_train)`. The fitted object can then transform `X_test` or a future API request without learning anything from it.
