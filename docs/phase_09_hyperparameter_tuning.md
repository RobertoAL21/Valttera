# Phase 9 — Gradient Boosting hyperparameter tuning

## Why Gradient Boosting

In the untuned comparison, Gradient Boosting had the lowest RMSE and highest R². Its MAE was close to the best Random Forest MAE, making it the strongest candidate for a bounded search.

## Search design

`RandomizedSearchCV` samples 20 combinations and uses five-fold cross-validation on `X_train`/`y_train` only. It ranks candidates by negative RMSE, so the search prefers a lower RMSE without looking at the held-out test target values.

| Hyperparameter | Values considered | Purpose |
| --- | --- | --- |
| `n_estimators` | 100, 150, 200, 250, 300 | Number of sequential boosting stages. |
| `learning_rate` | 0.03, 0.05, 0.08, 0.10 | Contribution of each stage; lower values often need more stages. |
| `max_depth` | 2, 3, 4 | Complexity of individual trees and feature interactions. |
| `min_samples_leaf` | 1, 2, 4 | Minimum samples in a leaf; higher values can reduce overfitting. |
| `subsample` | 0.8, 1.0 | Fraction of rows used per stage; 0.8 adds stochastic regularization. |

This is intentionally a small, explainable search. It is not an exhaustive sweep, and it avoids tuning unrelated or excessively broad parameters.

## Comparison rule

After cross-validation chooses its best configuration, we compare its held-out metrics to the untuned Gradient Boosting model. Tuning is retained only if it improves the relevant error measures without creating a disproportionate complexity cost.

## Result

The best five-fold cross-validation RMSE was **$28,531.52** with:

```python
{
    "model__subsample": 0.8,
    "model__n_estimators": 250,
    "model__min_samples_leaf": 1,
    "model__max_depth": 3,
    "model__learning_rate": 0.05,
}
```

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Untuned Gradient Boosting | $17,515.24 | $29,444.36 | 0.89 |
| Tuned Gradient Boosting | $16,263.20 | $27,266.73 | 0.90 |

The tuned candidate reduces typical error by about $1,252 and RMSE by about $2,178 on the held-out partition. It is therefore the candidate carried forward to final evaluation.
