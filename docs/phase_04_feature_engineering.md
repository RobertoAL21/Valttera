# Phase 4 — Feature engineering

## Principles

The features in this phase are deterministic, interpretable, and computable from a property's input attributes. The function does not fit statistics, encode categories, or impute missing values. Those learned operations belong to the training-only preprocessing pipeline in Phase 6.

| Feature | Definition | Why it may help |
| --- | --- | --- |
| `TotalSF` | Basement + first floor + second floor area | Represents the property's usable scale more completely than one area field. |
| `HouseAge` | Sale year − construction year | Buyers commonly value newer construction differently from older homes. |
| `YearsSinceRemodel` | Sale year − remodel year | Captures renovation recency. |
| `TotalBathrooms` | Full baths + 0.5 × half baths, including basement baths | Captures utility better than separate bathroom counts. |
| `TotalOutdoorSpace` | Deck and porch areas combined | Represents usable outdoor amenity space. |
| `OverallQualGrLivArea` | Overall quality × above-ground living area | Lets a linear model express that additional space may be valued differently by quality tier. |
| `HasRemodel` | Remodel year is after construction year | Separates renovated homes from never-remodeled homes. |
| `HasGarage` / `HasBasement` | Positive garage/basement area | Explicit absence indicators can complement area values. |

These features are retained alongside their component fields. Model comparison—not intuition alone—will determine whether they improve generalization.
