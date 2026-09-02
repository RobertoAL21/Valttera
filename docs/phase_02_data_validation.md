# Phase 2 — Raw-data validation and deterministic cleaning

## Source and preservation

`data/raw/train.csv` contains the Ames/Kaggle training dataset. It is treated as immutable. The processed output, `data/processed/train_clean.csv`, is created from it; no raw file is overwritten.

## Validation findings

| Check | Result | Decision |
| --- | ---: | --- |
| Rows / columns | 1,460 / 81 | Expected Ames training shape. |
| Exact duplicate rows | 0 | No rows removed for the current source. The reusable cleaner still removes exact duplicates if a future ingestion contains them. |
| Missing target (`SalePrice`) | 0 | The training target is usable. |
| Constant columns | 0 | No columns are removed for this reason. |
| High-cardinality categorical columns (>50 levels) | 0 | One-hot encoding remains practical for this dataset. The highest is `Neighborhood` with 25 levels. |
| Impossible values | 0 | Checked positive price, month range 1–12, quality ranges 1–10, and renovation year not preceding construction. |
| Negative numeric values | 0 | No impossible negative numeric measurements found. |

## Missing values

There are 19 columns with missing values. The most incomplete are `PoolQC` (1,453), `MiscFeature` (1,406), `Alley` (1,369), `Fence` (1,179), `MasVnrType` (872), and `FireplaceQu` (690). In this dataset many missing categorical values mean that an amenity is absent, rather than that the source is corrupt. We retain them now and will handle them explicitly in a train-fitted pipeline during Phase 6.

## Type correction and cleaning decisions

- `MSSubClass` is cast to a string because it is a dwelling-type code, not a continuous measurement.
- `Id` is dropped from the processed modelling dataset. It is merely an identifier and could encourage a spurious ordering relationship.
- Exact duplicate rows are removed defensively.
- No missing-value imputation, outlier removal, scaling, or encoding occurs here. Those transformations either learn from data or need a modelling decision; doing them on the full dataset now would risk leakage.

## Suspicious values and outliers

The reusable validator reports 1.5-IQR outlier counts for each numerical feature. These are flags for EDA, not automatic deletion: unusually large homes, lots, or prices may be genuine and economically meaningful. We will inspect their relationship with `SalePrice` in Phase 3 before making any decision.
