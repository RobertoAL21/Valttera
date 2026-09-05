# Phase 10 — Final model evaluation

The selected candidate is the tuned Gradient Boosting pipeline. Evaluation uses the fixed test partition and reports MAE, RMSE, R², residual diagnostics, and tree feature importance.

Residuals are defined as `actual_price - predicted_price`:

- Positive residual: the model underpredicted the sale price.
- Negative residual: the model overpredicted the sale price.

The accompanying final-evaluation section in `notebooks/03_model_experiments.ipynb` includes three required checks:

1. Actual versus predicted price, with a perfect-prediction diagonal.
2. Residual distribution and residuals versus predictions.
3. The top 15 tree feature importances.

Feature importance is diagnostic rather than causal. It measures a feature's contribution to split-based error reduction in this specific fitted model; correlated fields may share or obscure one another's importance.

## Final holdout metrics

| MAE | RMSE | R² |
| ---: | ---: | ---: |
| $16,263.20 | $27,266.73 | 0.903 |

The mean signed residual is about **+$593**, so there is little average directional bias. However, average absolute error varies considerably by actual-price quartile:

| Actual-price quartile | Mean absolute error |
| --- | ---: |
| $35,311–$127,000 | $11,798.99 |
| $127,000–$154,150 | $9,225.82 |
| $154,150–$209,175 | $12,552.06 |
| $209,175–$755,000 | $31,405.41 |

The model is much less reliable for expensive homes. Its largest error underpredicts a $755,000 home by about $226,941. This is consistent with the right-tailed target distribution and the small number of luxury-home examples; it is a material limitation to disclose to API consumers.

The most influential transformed features are `OverallQualGrLivArea`, `TotalSF`, `OverallQual`, `BsmtFinSF1`, and `HouseAge`. These are plausible property attributes, but importance is not evidence of causality.
