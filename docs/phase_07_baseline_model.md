# Phase 7 — Baseline and linear regression

## Why a baseline comes first

`DummyRegressor(strategy="mean")` predicts the training-set mean sale price for every property. It ignores all property features. A useful house-price model must beat this simple reference on the same held-out rows.

`LinearRegression` is the first real model. It is a transparent, fast candidate that estimates a weighted linear combination of the processed features. It is intentionally untuned in this phase.

For this benchmark, Linear Regression uses a compact, domain-informed subset: quality, living/total area, garage capacity, lot area, bathroom/age/renovation measures, and a small set of meaningful categoricals such as neighborhood and kitchen quality. This is not hyperparameter tuning; it is a deliberate baseline design that avoids unstable coefficients from an unnecessarily wide sparse matrix. Tree models will be assessed with the full common feature pipeline in the next phase.

Both estimators use the exact same feature-engineering and preprocessing pipeline. This is the only fair comparison: a better score should come from the model's use of property information, not from different data preparation.

The categorical encoder drops one reference category per feature. With an intercept-based linear regression, retaining every category dummy creates an exact linear dependency; removing one dummy preserves the information and improves numerical stability.

## Evaluation metrics

| Metric | Meaning | Preferred direction |
| --- | --- | --- |
| MAE | Average absolute dollar prediction error | Lower |
| RMSE | Error that penalizes large misses more heavily | Lower |
| R² | Fraction of variation explained relative to predicting the mean | Higher |

We do not use classification accuracy because price is a continuous quantity; a prediction is not simply correct or incorrect.

## Holdout discipline

Both phase-7 candidates are evaluated on the same 292-row test partition. These are benchmark results, not a reason to repeatedly optimize against the test set. Hyperparameter selection in later phases will use cross-validation on the training partition; the final report will evaluate the selected pipeline once on the holdout set.
