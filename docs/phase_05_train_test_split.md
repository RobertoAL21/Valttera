# Phase 5 — Train/test split

## Split contract

The 1,460 rows in `data/processed/train_clean.csv` are split once using `random_state=42` and `test_size=0.20`.

| Partition | Rows | Purpose |
| --- | ---: | --- |
| `X_train`, `y_train` | 1,168 | Fit preprocessing, feature engineering, and candidate models. |
| `X_test`, `y_test` | 292 | Final, held-out evaluation only. |

`X` contains 79 predictor columns and `y` is `SalePrice`. The target is removed from `X` before splitting.

## Why the order matters

Some future steps learn from the data: a median imputer learns medians, an encoder learns category levels, a scaler learns means and standard deviations, and a model learns relationships with price. If any of those are fitted before the split, information from the future test rows influences training. That is data leakage and makes reported test metrics too optimistic.

The next preprocessing pipeline will therefore fit only on `X_train`. It will apply the learned transformations to `X_test` without refitting. The deterministic feature formulas from Phase 4 are also designed to become part of that same pipeline.
