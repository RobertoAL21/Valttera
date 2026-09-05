# Phase 8 — Untuned model comparison

## Candidates

| Model | Strength | Limitation |
| --- | --- | --- |
| Linear Regression | Fast and easy to explain; provides a strong transparent benchmark. | Assumes additive linear relationships and can be sensitive to multicollinearity. |
| Random Forest Regressor | Captures non-linear effects and interactions with little preparation. | Larger artifact, slower inference/training, and less direct global interpretation. |
| Gradient Boosting Regressor | Often achieves strong tabular-data accuracy by sequentially correcting errors. | Sequential training is less parallel, and it needs careful tuning later. |

The random forest and gradient boosting pipelines use all non-redundant engineered/raw features. The linear candidate uses the deliberately compact baseline feature set from Phase 7 to keep coefficient estimates stable. All candidates use the same train/test partition and fit their own preprocessing steps on `X_train` only.

## Evaluation decision

We compare MAE, RMSE, and R² on the same test partition. Selection will not rely on R² alone:

- MAE communicates typical dollar error.
- RMSE makes large misses visible.
- R² gives useful context relative to a mean prediction.
- Training cost, inference cost, and interpretability matter for a portfolio API as well.

The comparison is deliberately untuned. Phase 9 will tune only the strongest promising candidate using cross-validation on the training partition.
